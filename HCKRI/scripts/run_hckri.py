"""Portable, audited entry point for the unchanged HCKRI training implementation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from itertools import product
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from power_risk import load_experiment_config, load_risk_dataset
from power_risk.comparison_ablations import ComparisonAblationNet
from run_ablation_study import ABLATION_VARIANTS, make_loaders, run_ablation

VARIANTS = ["Full", *list("ABCDEFGHIJKLMN")]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def switches(variant):
    result = dict(ABLATION_VARIANTS["Full"])
    if variant == "H":
        return dict(ABLATION_VARIANTS["No-PrototypeLibrary"])
    if variant == "I":
        result["use_contrastive"] = False
    if variant == "J":
        result["use_semi"] = False
    return result


def metrics_from_saved(path):
    with np.load(path) as saved:
        y, pred = saved["y_true"], saved["pred"]
        unknown = saved["unknown_attack_mask"].astype(bool)
        flags = saved["unknown_flags"].astype(bool)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=[1, 2, 3, 4], average="macro", zero_division=0)
    tp = int((unknown & flags).sum())
    fp = int((~unknown & flags).sum())
    fn = int((unknown & ~flags).sum())
    result = {"precision": p, "recall": r, "f1": f1,
              "uarc": float(np.mean(pred[unknown] > 0)) if unknown.any() else float("nan"),
              "unknown_flag_recall": tp / (tp + fn) if tp + fn else float("nan"),
              "unknown_flag_precision": tp / (tp + fp) if tp + fp else float("nan"),
              "unknown_flag_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
              "known_flag_rate": float(np.mean(flags[~unknown])) if (~unknown).any() else float("nan")}
    matrix = confusion_matrix(y, pred, labels=list(range(5)))
    return result, matrix


def audit_saved(folder, measured):
    metrics, matrix = metrics_from_saved(folder / "predictions.npz")
    for key in ("precision", "recall", "f1"):
        np.testing.assert_allclose(metrics[key], measured[key], rtol=0, atol=1e-12)
    save_json(folder / "confusion_matrix.json", matrix)
    with (folder / "per_class_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "TP", "FP", "FN", "support"])
        writer.writeheader()
        for c in range(5):
            tp = int(matrix[c, c])
            writer.writerow({"class_id": c, "TP": tp, "FP": int(matrix[:, c].sum()) - tp,
                             "FN": int(matrix[c].sum()) - tp, "support": int(matrix[c].sum())})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["Full"])
    parser.add_argument("--fractions", nargs="+", type=float)
    parser.add_argument("--pseudo-thresholds", nargs="+", type=float)
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicit opt-in for synthetic smoke tests only")
    args = parser.parse_args()
    cfg = load_experiment_config(resolve(args.config))
    cfg.data.data_dir = str(resolve(Path(cfg.data.data_dir)))
    if cfg.data.auto_download:
        parser.error("Set auto_download=false and prepare the dataset separately.")
    if cfg.data.allow_synthetic_fallback and not args.allow_synthetic:
        parser.error("Synthetic fallback requires --allow-synthetic; never use this for paper results.")
    if args.seed is not None:
        cfg.data.seed = cfg.train.seed = args.seed
    if cfg.data.seed != cfg.train.seed:
        parser.error("Use matching data/train seeds or pass --seed.")
    device_name = args.device or cfg.train.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable. Install a compatible PyTorch build or use --device cpu.")
    cfg.train.device = device_name
    fractions = args.fractions or [cfg.data.labeled_fraction]
    thresholds = args.pseudo_thresholds or [cfg.train.pseudo_label_threshold]
    if any(not 0 < f <= 1 for f in fractions) or any(not 0 <= t <= 1 for t in thresholds):
        parser.error("Fractions must lie in (0,1] and thresholds in [0,1].")
    if any(len(v) != len(set(v)) for v in [fractions, thresholds, args.variants]):
        parser.error("Duplicate grid values are not allowed.")
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)
    input_dir = Path(cfg.data.data_dir)
    files = sorted(input_dir.rglob(cfg.data.file_glob)) if input_dir.exists() else []
    manifest = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "device": device_name,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "packages": {n: importlib.metadata.version(n) for n in
                             ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib"]},
                "deterministic_algorithms": True, "seed": cfg.train.seed,
                "input_sha256": {str(p.relative_to(input_dir)): sha256(p) for p in files},
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for base in ["src/power_risk", "scripts"]
                                  for p in sorted((ROOT / base).glob("*.py"))},
                "variants": args.variants, "fractions": fractions, "pseudo_thresholds": thresholds}
    save_json(output / "manifest.json", manifest)
    rows, masks = [], {}
    try:
        bundle = None
        for fraction in fractions:
            fraction_cfg = copy.deepcopy(cfg)
            fraction_cfg.data.labeled_fraction = fraction
            current = load_risk_dataset(fraction_cfg.data)
            if bundle is None:
                bundle = current
                if bundle.n_classes != 5:
                    raise ValueError("This release expects five reference classes.")
                if bundle.train_y_unknown.any() or bundle.val_y_unknown.any():
                    raise AssertionError("Held-out attack types entered train/validation.")
                audit = {"records": bundle.split_summary, "used_synthetic_data": bundle.used_synthetic_data,
                         "source_names": bundle.source_names, "feature_dim": bundle.feature_dim,
                         "class_names": bundle.class_names, "unknown_attack_types": bundle.unknown_attack_types,
                         "train_windows": len(bundle.train_x), "val_windows": len(bundle.val_x),
                         "test_windows": len(bundle.test_x), "unknown_train_windows": int(bundle.train_y_unknown.sum()),
                         "unknown_validation_windows": int(bundle.val_y_unknown.sum()),
                         "unknown_test_windows": int(bundle.test_y_unknown.sum())}
                save_json(output / "split_audit.json", audit)
                manifest["used_synthetic_data"] = bundle.used_synthetic_data
                save_json(output / "manifest.json", manifest)
            else:
                for name in ["train_x", "train_y_class", "train_y_unknown", "val_x", "val_y_class", "test_x", "test_y_class"]:
                    np.testing.assert_array_equal(getattr(bundle, name), getattr(current, name))
            masks[f"fraction_{fraction:g}"] = current.train_labeled_mask.copy()
            np.savez_compressed(output / "labeled_masks.npz", **masks)
            for threshold, variant in product(thresholds, args.variants):
                case = copy.deepcopy(fraction_cfg)
                case.train.pseudo_label_threshold = threshold
                if variant == "K":
                    case.model.cnn_kernel_sizes = [5, 5, 5]
                folder = output / f"labels_{fraction:g}_pseudo_{threshold:g}" / variant
                folder.mkdir(parents=True, exist_ok=False)
                save_json(folder / "config.json", asdict(case))
                random.seed(case.train.seed)
                np.random.seed(case.train.seed)
                torch.manual_seed(case.train.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(case.train.seed)
                started = time.perf_counter()
                measured = run_ablation(case, current, make_loaders(case, current), torch.device(device_name),
                    switches(variant), verbose=True, artifact_dir=folder,
                    model_factory=partial(ComparisonAblationNet, variant=variant))
                checked = audit_saved(folder, measured)
                result = {"variant": variant, "labeled_fraction": fraction, "pseudo_threshold": threshold,
                          "seed": case.train.seed, "used_synthetic_data": current.used_synthetic_data,
                          **{k: v for k, v in measured.items() if k != "pred"}, **checked,
                          "seconds": time.perf_counter() - started}
                save_json(folder / "result.json", result)
                rows.append(clean(result))
                with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(json.dumps(clean(result), ensure_ascii=False, allow_nan=False), flush=True)
        manifest.update(status="completed", completed_trainings=len(rows))
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
