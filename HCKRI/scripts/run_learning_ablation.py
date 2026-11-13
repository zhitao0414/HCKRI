"""Paired prototype, contrastive and semi-supervised ablations on real data."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from power_risk import load_experiment_config, load_risk_dataset
from power_risk.reporting import write_csv, write_markdown_table
from run_ablation_study import ABLATION_VARIANTS, make_loaders, run_ablation
from train_ton_iot_risk import choose_device, set_seed

VARIANTS = ["Full", "No-PrototypeLibrary", "No-Contrastive", "No-SemiSupervised"]
CASES = {
    "ton_iot": ("ton_iot_ablation_long_window.json", "train_test_network.csv"),
    "edge_iiot": ("edge_iiot_ablation_long_window.json", "ML-EdgeIIoT-dataset.csv"),
    "psad": ("power_system_attack_ablation_long_window.json", "*.csv"),
}
METRICS = ["precision", "recall", "f1", "auc", "fpr", "udr", "accuracy"]


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_loaders(config, bundle):
    loaders = make_loaders(config, bundle)
    loaders["library"] = loaders["train"]
    # Library inference must not consume the training sampler or global RNG.
    return {name: torch.utils.data.DataLoader(
        loader.dataset, batch_size=config.train.batch_size, shuffle=name == "train",
        generator=torch.Generator().manual_seed(config.train.seed + index),
    ) for index, (name, loader) in enumerate(loaders.items())}


def validate_predictions(folder, metrics):
    with np.load(folder / "predictions.npz") as saved:
        y, pred, mask = saved["y_true"], saved["pred"], saved["unknown_attack_mask"]
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1, 2, 3, 4],
                                                    average="macro", zero_division=0)
        fpr = np.mean([np.sum((y != c) & (pred == c)) / np.sum(y != c)
                       for c in [1, 2, 3, 4]])
        measured = {"precision": p, "recall": r, "f1": f, "fpr": fpr,
                    "accuracy": np.mean(y == pred), "udr": np.mean(pred[mask] > 0)}
        for key, value in measured.items():
            np.testing.assert_allclose(metrics[key], value, rtol=0, atol=1e-12)
        matrix = confusion_matrix(y, pred, labels=[0, 1, 2, 3, 4])
        save_json(folder / "confusion_matrix.json", matrix.tolist())
        counts = []
        for c in range(5):
            tp = int(matrix[c, c])
            counts.append({"class_id": c, "TP": tp, "FP": int(matrix[:, c].sum()) - tp,
                           "FN": int(matrix[c].sum()) - tp,
                           "support": int(matrix[c].sum())})
        write_csv(folder / "per_class_counts.csv", counts, ["class_id", "TP", "FP", "FN", "support"])


def refresh_tables(out, rows):
    write_csv(out / "results.csv", rows, ["dataset", "variant", "run", "seed", *METRICS, "seconds"])
    summary = []
    for dataset in CASES:
        for variant in VARIANTS:
            subset = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            if subset:
                summary.append({"dataset": dataset, "variant": variant, "runs": len(subset),
                                **{k: float(np.mean([r[k] for r in subset])) for k in METRICS}})
    columns = ["dataset", "variant", "runs", *METRICS]
    write_csv(out / "summary.csv", summary, columns)
    write_markdown_table(out / "summary.md", summary, columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260900)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/learning_ablation_1run_20260923")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = {"started": datetime.now().isoformat(), "status": "running",
                "command": sys.argv, "python": sys.version, "platform": platform.platform(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "runs": args.runs, "base_seed": args.base_seed, "variants": VARIANTS,
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                    [Path(__file__), ROOT / "scripts/run_ablation_study.py",
                     ROOT / "scripts/train_ton_iot_risk.py", ROOT / "src/power_risk/data.py",
                     ROOT / "src/power_risk/config.py", ROOT / "src/power_risk/ablations.py",
                     ROOT / "src/power_risk/metrics.py"]}}
    save_json(out / "manifest.json", manifest)
    rows = []
    try:
        for dataset in args.datasets:
            config_name, pattern = CASES[dataset]
            for run_id in range(1, args.runs + 1):
                cfg = load_experiment_config(ROOT / "configs" / config_name)
                cfg.data.data_dir = str(ROOT / cfg.data.data_dir)
                cfg.data.file_glob = pattern
                cfg.data.auto_download = False
                cfg.data.allow_synthetic_fallback = False
                cfg.data.seed = cfg.train.seed = args.base_seed + run_id - 1
                if dataset == "psad":
                    cfg.data.split_group_column = "time"
                    cfg.data.excluded_feature_columns = ["marker", "time", "dataset_part"]
                folder = out / dataset / f"run_{run_id:02d}"
                folder.mkdir(parents=True)
                files = sorted(Path(cfg.data.data_dir).rglob(pattern))
                expected = 4 if dataset == "psad" else 1
                if len(files) != expected:
                    raise ValueError(f"Expected {expected} input files for {dataset}, found {files}")
                save_json(folder / "config.json", asdict(cfg))
                save_json(folder / "inputs.json", [{"path": str(p), "sha256": sha256(p)} for p in files])
                print(f"Loading {dataset}, seed={cfg.train.seed}", flush=True)
                bundle = load_risk_dataset(cfg.data)
                assert not bundle.used_synthetic_data
                assert not bundle.train_y_unknown.any() and not bundle.val_y_unknown.any()
                assert bundle.test_y_unknown.any()
                encoded = bundle.split_summary["encoded_columns"]
                assert not set(cfg.data.excluded_feature_columns) & set(encoded)
                split = {"record_audit": bundle.split_summary, "feature_dim": bundle.feature_dim,
                         "sources": bundle.source_names,
                         "labeled_train": int(bundle.train_labeled_mask.sum()),
                         "unlabeled_train": int((~bundle.train_labeled_mask).sum()),
                         "val": len(bundle.val_x), "test": len(bundle.test_x),
                         "unknown_test": int(bundle.test_y_unknown.sum()),
                         "test_class_counts": np.bincount(bundle.test_y_class, minlength=5).tolist()}
                save_json(folder / "split_audit.json", split)
                print(json.dumps({k: v for k, v in split.items() if k != "record_audit"}), flush=True)
                for variant in VARIANTS:
                    set_seed(cfg.train.seed)
                    destination = folder / variant
                    switches = ABLATION_VARIANTS[variant]
                    print(f"START {dataset} {variant}", flush=True)
                    started = time.perf_counter()
                    metrics = run_ablation(cfg, bundle, paired_loaders(cfg, bundle),
                                           choose_device(cfg), switches, verbose=True,
                                           artifact_dir=destination)
                    validate_predictions(destination, metrics)
                    row = {"dataset": dataset, "variant": variant, "run": run_id,
                           "seed": cfg.train.seed, **{k: float(metrics[k]) for k in METRICS},
                           "seconds": time.perf_counter() - started}
                    save_json(destination / "result.json", {**row, "switches": switches})
                    rows.append(row)
                    refresh_tables(out, rows)
                    print("DONE " + json.dumps(row), flush=True)
                del bundle
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["finished"] = datetime.now().isoformat()
        manifest["completed_trainings"] = len(rows)
        save_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
