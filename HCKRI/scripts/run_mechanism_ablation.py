"""Paired mechanism controls using frozen A-G long-window settings."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from power_risk import load_experiment_config, load_risk_dataset
from power_risk.mechanism_ablations import MechanismRiskNet
from power_risk.reporting import write_csv, write_markdown_table
from run_ablation_study import ABLATION_VARIANTS, run_ablation
from run_learning_ablation import (CASES, METRICS, paired_loaders, save_json,
                                   sha256, validate_predictions)
from train_ton_iot_risk import choose_device, set_seed

VARIANTS = ["Full", "Single-Scale-5", "No-CNN-Residual", "Fixed-Source-Gate",
            "No-Source-Context", "Final-Stage-Only"]
DESCRIPTIONS = {
    "Full": "Original A-G Full forward pass.",
    "Single-Scale-5": "Replace kernels [3,5,7] with [5,5,5]; keep three branches and parameter count.",
    "No-CNN-Residual": "Remove only the input-to-CNN skip; preserve CNN normalization and all other residuals.",
    "Fixed-Source-Gate": "Replace adaptive sigmoid gate with constant one; preserve source addition.",
    "No-Source-Context": "Remove additional source context from gate and Transformer path; retain input source embedding.",
    "Final-Stage-Only": "Zero early pooled features at final concatenation; keep CNN/LSTM in the sequential path.",
}
EXTRA = ["unknown_flag_recall", "known_flag_rate", "unknown_flag_precision", "unknown_flag_f1"]


def make_config(dataset, seed, source_layout, psad_encoding="legacy"):
    filename, pattern = CASES[dataset]
    cfg = load_experiment_config(ROOT / "configs" / filename)
    if source_layout == "multiview" and dataset != "psad":
        view = load_experiment_config(ROOT / "configs" / (
            "ton_iot_multiview_risk.json" if dataset == "ton_iot"
            else "edge_iiot_multisource_risk.json"))
        # Change only the source layout; retain A-G model, optimizer and window settings.
        for name in ["data_dir", "file_glob", "max_rows_per_source", "align_sources_by_rank",
                     "include_source_names", "exclude_source_names", "max_rows_per_file"]:
            setattr(cfg.data, name, getattr(view.data, name))
        pattern = cfg.data.file_glob
    cfg.data.data_dir = str(ROOT / cfg.data.data_dir)
    cfg.data.file_glob = pattern
    cfg.data.auto_download = False
    cfg.data.allow_synthetic_fallback = False
    cfg.data.seed = cfg.train.seed = seed
    if dataset == "psad":
        cfg.data.split_group_column = "time"
        cfg.data.excluded_feature_columns = ["marker", "time", "dataset_part"]
        cfg.data.numeric_observed_only = psad_encoding == "observed"
    return cfg


def extra_metrics(folder):
    with np.load(folder / "predictions.npz") as saved:
        true = saved["unknown_attack_mask"].astype(bool)
        pred = saved["unknown_flags"].astype(bool)
        tp = int((true & pred).sum())
        fp = int((~true & pred).sum())
        fn = int((true & ~pred).sum())
        return {"unknown_flag_recall": tp / max(int(true.sum()), 1),
                "known_flag_rate": fp / max(int((~true).sum()), 1),
                "unknown_flag_precision": tp / max(tp + fp, 1),
                "unknown_flag_f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def refresh_tables(out, rows):
    keys = [*METRICS, *EXTRA]
    write_csv(out / "results.csv", rows,
              ["dataset", "variant", "run", "seed", "parameters", *keys, "seconds"])
    summary, differences = [], []
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        full = {r["seed"]: r for r in rows if r["dataset"] == dataset and r["variant"] == "Full"}
        for variant in VARIANTS:
            subset = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            if not subset:
                continue
            summary.append({"dataset": dataset, "variant": variant, "runs": len(subset),
                **{k + suffix: float(func([r[k] for r in subset]))
                   for k in keys for suffix, func in [("_mean", np.mean),
                    ("_std", lambda v: np.std(v, ddof=1) if len(v) > 1 else 0.0)]}})
            if variant != "Full":
                delta = [full[r["seed"]]["f1"] - r["f1"] for r in subset if r["seed"] in full]
                if delta:
                    differences.append({"dataset": dataset, "variant": variant, "pairs": len(delta),
                        "full_minus_ablation_f1_mean_pp": float(np.mean(delta) * 100),
                        "delta_std_pp": float(np.std(delta, ddof=1) * 100) if len(delta) > 1 else 0.0,
                        "full_wins": sum(d > 0 for d in delta), "ablation_wins": sum(d < 0 for d in delta)})
    if summary:
        columns = ["dataset", "variant", "runs", *[k + suffix for k in keys for suffix in ["_mean", "_std"]]]
        write_csv(out / "summary.csv", summary, columns)
        write_markdown_table(out / "summary.md", summary,
                             ["dataset", "variant", "runs", *[k + "_mean" for k in keys]])
    if differences:
        columns = list(differences[0])
        write_csv(out / "paired_deltas.csv", differences, columns)
        write_markdown_table(out / "paired_deltas.md", differences, columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--source-layout", choices=["ag", "multiview"], default="ag")
    parser.add_argument("--psad-encoding", choices=["legacy", "observed"], default="legacy")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260900)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1:
        parser.error("runs and threads must be positive")
    if "Full" not in args.variants:
        parser.error("Include Full for paired comparisons")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(args.threads)
    source_paths = [Path(__file__), ROOT / "src/power_risk/mechanism_ablations.py",
        ROOT / "src/power_risk/ablations.py", ROOT / "src/power_risk/data.py",
        ROOT / "src/power_risk/config.py", ROOT / "src/power_risk/metrics.py",
        ROOT / "scripts/run_ablation_study.py", ROOT / "scripts/run_learning_ablation.py",
        ROOT / "scripts/train_ton_iot_risk.py"]
    source_paths += [ROOT / "configs" / filename for filename, _ in CASES.values()]
    source_paths += [ROOT / "configs" / filename for filename in
                    ["ton_iot_multiview_risk.json", "edge_iiot_multisource_risk.json"]]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    manifest_path = out / "manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in {"source_sha256": hashes, "source_layout": args.source_layout,
                           "base_seed": args.base_seed, "psad_encoding": args.psad_encoding,
                           "threads": args.threads}.items():
            if previous[key] != value:
                raise ValueError(f"Cannot resume after changing {key}; use a new output folder")
    manifest = {"started": datetime.now().isoformat(), "status": "running", "command": sys.argv,
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_sha256": hashes, "source_layout": args.source_layout, "base_seed": args.base_seed,
        "psad_encoding": args.psad_encoding, "threads": args.threads,
        "runs_requested": args.runs, "variants": args.variants, "definitions": DESCRIPTIONS,
        "notes": ["P/R/F1 and FPR retain A-G risk-class macro averaging (classes 1..4).",
            "Legacy udr is UARC: fraction of held-out attacks predicted as any risk, not explicit unknown recall.",
            "PSAD always uses grouped original-event time IDs and excludes marker/time/dataset_part.",
            "PSAD results are not directly comparable with legacy row-split A-G results.",
            "A-G source layout has one source for TON/Edge; multiview is supplementary and uses different input data.",
            "Training order uses an independent seeded generator; library inference never shuffles training.",
            "No test-based tuning, outcome filtering or result targeting is performed."]}
    save_json(manifest_path, manifest)
    snapshot = out / "source_snapshot"
    for path in source_paths:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
    rows = []
    try:
        for dataset in args.datasets:
            for run_id in range(1, args.runs + 1):
                seed = args.base_seed + run_id - 1
                cfg = make_config(dataset, seed, args.source_layout, args.psad_encoding)
                folder = out / dataset / f"run_{run_id:02d}"
                folder.mkdir(parents=True, exist_ok=args.resume)
                files = sorted(Path(cfg.data.data_dir).rglob(cfg.data.file_glob))
                expected = 4 if dataset == "psad" else (3 if args.source_layout == "multiview" else 1)
                if len(files) != expected:
                    raise ValueError(f"Expected {expected} input files for {dataset}: {files}")
                inputs = [{"path": str(p), "sha256": sha256(p)} for p in files]
                inputs_path = folder / "inputs.json"
                if args.resume and inputs_path.exists():
                    if json.loads(inputs_path.read_text(encoding="utf-8")) != inputs:
                        raise ValueError("Input data changed; refusing to reuse results")
                save_json(inputs_path, inputs)
                save_json(folder / "config.json", asdict(cfg))
                print(f"LOAD {dataset} seed={seed} layout={args.source_layout}", flush=True)
                bundle = load_risk_dataset(cfg.data)
                assert not bundle.used_synthetic_data
                assert not bundle.train_y_unknown.any() and not bundle.val_y_unknown.any()
                assert bundle.test_y_unknown.any()
                assert not set(cfg.data.excluded_feature_columns) & set(bundle.split_summary["encoded_columns"])
                audit = {"records": bundle.split_summary, "sources": bundle.source_names,
                    "feature_dim": bundle.feature_dim, "train_windows": len(bundle.train_x),
                    "labeled_train_windows": int(bundle.train_labeled_mask.sum()),
                    "val_windows": len(bundle.val_x), "test_windows": len(bundle.test_x),
                    "unknown_test_windows": int(bundle.test_y_unknown.sum())}
                save_json(folder / "split_audit.json", audit)
                print(json.dumps({k: v for k, v in audit.items() if k != "records"}), flush=True)
                for variant in args.variants:
                    destination = folder / variant
                    result_path = destination / "result.json"
                    if args.resume and result_path.exists():
                        row = json.loads(result_path.read_text(encoding="utf-8"))
                        validate_predictions(destination, row)
                        rows.append(row)
                        refresh_tables(out, rows)
                        print(f"REUSE {dataset} {variant} seed={seed}", flush=True)
                        continue
                    variant_cfg = copy.deepcopy(cfg)
                    if variant == "Single-Scale-5":
                        variant_cfg.model.cnn_kernel_sizes = [5, 5, 5]
                    set_seed(seed)
                    factory = partial(MechanismRiskNet, mechanism=variant)
                    model = factory(bundle.feature_dim, bundle.n_sources, bundle.n_classes, variant_cfg.model)
                    parameters = sum(p.numel() for p in model.parameters())
                    del model
                    # Parameter counting must not consume initialization RNG.
                    set_seed(seed)
                    print(f"START {dataset} {variant} seed={seed} parameters={parameters}", flush=True)
                    started = time.perf_counter()
                    metrics = run_ablation(variant_cfg, bundle, paired_loaders(variant_cfg, bundle),
                        choose_device(variant_cfg), ABLATION_VARIANTS["Full"], verbose=True,
                        artifact_dir=destination, model_factory=factory)
                    validate_predictions(destination, metrics)
                    row = {"dataset": dataset, "variant": variant, "run": run_id, "seed": seed,
                        "parameters": parameters, **{k: float(metrics[k]) for k in METRICS},
                        **extra_metrics(destination), "seconds": time.perf_counter() - started}
                    save_json(destination / "config.json", asdict(variant_cfg))
                    save_json(result_path, row)
                    rows.append(row)
                    refresh_tables(out, rows)
                    print("DONE " + json.dumps(row), flush=True)
                del bundle
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["finished"] = datetime.now().isoformat()
        manifest["completed_trainings"] = len(rows)
        save_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
