from __future__ import annotations

import copy
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from power_risk import load_experiment_config, load_risk_dataset
from power_risk.ablations import AblationRiskNet
from power_risk.config import ExperimentConfig
from power_risk.reporting import format_markdown_table, write_csv, write_markdown_table
from train_ton_iot_risk import (
    PrototypeRiskLibrary,
    choose_device,
    default_risk_thresholds,
    evaluate_multiclass_predictions,
    make_loader,
    predict,
    select_best_risk_thresholds,
    set_seed,
    update_risk_library,
)


ABLATION_VARIANTS = {
    "Full": {
        "use_cnn": True,
        "use_lstm": True,
        "use_transformer": True,
        "use_gate": True,
        "use_library": True,
        "use_semi": True,
    },
    "No-CNN": {
        "use_cnn": False,
        "use_lstm": True,
        "use_transformer": True,
        "use_gate": True,
        "use_library": True,
        "use_semi": True,
    },
    "No-BiLSTM": {
        "use_cnn": True,
        "use_lstm": False,
        "use_transformer": True,
        "use_gate": True,
        "use_library": True,
        "use_semi": True,
    },
    "No-Transformer": {
        "use_cnn": True,
        "use_lstm": True,
        "use_transformer": False,
        "use_gate": True,
        "use_library": True,
        "use_semi": True,
    },
    "No-GatedFusion": {
        "use_cnn": True,
        "use_lstm": True,
        "use_transformer": True,
        "use_gate": False,
        "use_library": True,
        "use_semi": True,
    },
    "No-RiskLibrary": {
        "use_cnn": True,
        "use_lstm": True,
        "use_transformer": True,
        "use_gate": True,
        "use_library": False,
        "use_semi": True,
    },
    "No-SemiSupervised": {
        "use_cnn": True,
        "use_lstm": True,
        "use_transformer": True,
        "use_gate": True,
        "use_library": True,
        "use_semi": False,
    },
    "No-CNN-BiLSTM": {
        "use_cnn": False,
        "use_lstm": False,
        "use_transformer": True,
        "use_gate": True,
        "use_library": True,
        "use_semi": True,
    },
    "No-CNN-Transformer": {
        "use_cnn": False,
        "use_lstm": True,
        "use_transformer": False,
        "use_gate": True,
        "use_library": True,
        "use_semi": True,
    },
    "No-CNN-GatedFusion": {
        "use_cnn": False,
        "use_lstm": True,
        "use_transformer": True,
        "use_gate": False,
        "use_library": True,
        "use_semi": True,
    },
    "No-BiLSTM-Transformer": {
        "use_cnn": True,
        "use_lstm": False,
        "use_transformer": False,
        "use_gate": True,
        "use_library": True,
        "use_semi": True,
    },
    "No-BiLSTM-GatedFusion": {
        "use_cnn": True,
        "use_lstm": False,
        "use_transformer": True,
        "use_gate": False,
        "use_library": True,
        "use_semi": True,
    },
    "No-Transformer-GatedFusion": {
        "use_cnn": True,
        "use_lstm": True,
        "use_transformer": False,
        "use_gate": False,
        "use_library": True,
        "use_semi": True,
    },
    "No-GatedFusion-RiskLibrary": {
        "use_cnn": True,
        "use_lstm": True,
        "use_transformer": True,
        "use_gate": False,
        "use_library": False,
        "use_semi": True,
    },
    "No-RiskLibrary-SemiSupervised": {
        "use_cnn": True,
        "use_lstm": True,
        "use_transformer": True,
        "use_gate": True,
        "use_library": False,
        "use_semi": False,
    },
    "No-CNN-BiLSTM-Transformer": {
        "use_cnn": False,
        "use_lstm": False,
        "use_transformer": False,
        "use_gate": True,
        "use_library": True,
        "use_semi": True,
    },
    "Shallow-Supervised": {
        "use_cnn": False,
        "use_lstm": False,
        "use_transformer": False,
        "use_gate": False,
        "use_library": False,
        "use_semi": False,
    },
}

# Keep historical variants unchanged; these isolate the requested new controls.
ABLATION_VARIANTS["No-Contrastive"] = {
    **ABLATION_VARIANTS["Full"], "use_contrastive": False,
}
ABLATION_VARIANTS["No-PrototypeLibrary"] = {
    **ABLATION_VARIANTS["Full"], "use_library": False,
    "use_contrastive": False, "use_confidence_fallback": True,
}


class NullRiskLibrary:
    def __init__(self, confidence_fallback: bool = False) -> None:
        self.confidence_fallback = confidence_fallback

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
        del embeddings, labels, mask

    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor | None:
        del embeddings, labels, temperature
        return None

    @torch.no_grad()
    def unknown_flags(
        self,
        embeddings: torch.Tensor,
        confidence: torch.Tensor,
        confidence_threshold: float,
    ) -> torch.Tensor:
        if self.confidence_fallback:
            return confidence < confidence_threshold
        return torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)


def main() -> None:
    config_path = Path("configs/ton_iot_risk.json")
    n_runs = 20
    base_seed = 20260900
    variant_names = list(ABLATION_VARIANTS)

    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        n_runs = int(sys.argv[2])
    if len(sys.argv) > 3:
        base_seed = int(sys.argv[3])
    if len(sys.argv) > 4:
        variant_names = parse_variant_names(sys.argv[4])

    rows: list[dict[str, object]] = []
    used_synthetic = False
    dataset_slug = config_path.stem
    for suffix in ["_full_risk", "_risk_seed9300", "_risk"]:
        dataset_slug = dataset_slug.replace(suffix, "")
    output_stem = Path(f"outputs/{dataset_slug}_ablation_{n_runs}_runs_{variant_slug(variant_names)}")
    columns = ["variant", "run", "seed", "precision", "recall", "f1", "auc", "fpr", "udr"]
    partial_path = output_stem.with_name(output_stem.name + "_partial.csv")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    for variant_name in variant_names:
        for run_id in range(1, n_runs + 1):
            seed = base_seed + run_id - 1
            row, used_synthetic = run_once(config_path, variant_name, run_id, seed)
            rows.append(row)
            append_partial_row(partial_path, row, columns)
            print(
                f"{variant_name:18s} | Run {run_id:02d}/{n_runs} | "
                f"Seed={seed} | "
                f"P={row['precision']:.4f} | "
                f"R={row['recall']:.4f} | "
                f"F1={row['f1']:.4f} | "
                f"AUC={row['auc']:.4f} | "
                f"FPR={row['fpr']:.4f} | "
                f"UDR={row['udr']:.4f}",
                flush=True,
            )

    table_rows = add_average_rows(rows, variant_names)
    summary_rows = make_summary_rows(rows, variant_names)
    write_csv(output_stem.with_suffix(".csv"), table_rows, columns)
    write_markdown_table(
        output_stem.with_suffix(".md"),
        table_rows,
        columns,
        headers=["Variant", "Run", "Seed", "P", "R", "F1", "AUC", "FPR", "UDR"],
    )

    summary_path = Path(f"outputs/{dataset_slug}_ablation_{n_runs}_runs_{variant_slug(variant_names)}_summary.md")
    write_markdown_table(
        summary_path,
        summary_rows,
        ["variant", "precision", "recall", "f1", "auc", "fpr", "udr"],
        headers=["Variant", "Avg P", "Avg R", "Avg F1", "Avg AUC", "Avg FPR", "Avg UDR"],
    )

    print("\nAblation summary")
    print(
        format_markdown_table(
            summary_rows,
            ["variant", "precision", "recall", "f1", "auc", "fpr", "udr"],
            headers=["Variant", "Avg P", "Avg R", "Avg F1", "Avg AUC", "Avg FPR", "Avg UDR"],
        )
    )
    print(f"\nUsed synthetic fallback: {used_synthetic}")
    print(f"Saved: {output_stem.with_suffix('.csv')}")
    print(f"Saved: {output_stem.with_suffix('.md')}")
    print(f"Saved: {summary_path}")
    print(f"Partial rows: {partial_path}")


def append_partial_row(path: Path, row: dict[str, object], columns: list[str]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def parse_variant_names(raw: str) -> list[str]:
    aliases = {name.lower(): name for name in ABLATION_VARIANTS}
    aliases.update(
        {
            "full": "Full",
            "ours": "Full",
            "no_cnn": "No-CNN",
            "nocnn": "No-CNN",
            "no_bilstm": "No-BiLSTM",
            "nobilstm": "No-BiLSTM",
            "no_transformer": "No-Transformer",
            "notransformer": "No-Transformer",
            "no_gate": "No-GatedFusion",
            "nogate": "No-GatedFusion",
            "no_gatedfusion": "No-GatedFusion",
            "no_library": "No-RiskLibrary",
            "nolibrary": "No-RiskLibrary",
            "no_risklibrary": "No-RiskLibrary",
            "no_semi": "No-SemiSupervised",
            "nosemi": "No-SemiSupervised",
            "no_semisupervised": "No-SemiSupervised",
            "no_contrastive": "No-Contrastive",
            "no_prototype": "No-PrototypeLibrary",
            "no_cnn_bilstm": "No-CNN-BiLSTM",
            "nocnnbilstm": "No-CNN-BiLSTM",
            "no_cnn_transformer": "No-CNN-Transformer",
            "nocnntransformer": "No-CNN-Transformer",
            "no_cnn_gatedfusion": "No-CNN-GatedFusion",
            "no_cnn_gate": "No-CNN-GatedFusion",
            "nocnngate": "No-CNN-GatedFusion",
            "no_bilstm_transformer": "No-BiLSTM-Transformer",
            "nobilstmtransformer": "No-BiLSTM-Transformer",
            "no_bilstm_gatedfusion": "No-BiLSTM-GatedFusion",
            "no_bilstm_gate": "No-BiLSTM-GatedFusion",
            "nobilstmgate": "No-BiLSTM-GatedFusion",
            "no_transformer_gatedfusion": "No-Transformer-GatedFusion",
            "no_transformer_gate": "No-Transformer-GatedFusion",
            "notransformergate": "No-Transformer-GatedFusion",
            "no_gatedfusion_risklibrary": "No-GatedFusion-RiskLibrary",
            "no_gate_library": "No-GatedFusion-RiskLibrary",
            "nogatelibrary": "No-GatedFusion-RiskLibrary",
            "no_risklibrary_semisupervised": "No-RiskLibrary-SemiSupervised",
            "no_library_semi": "No-RiskLibrary-SemiSupervised",
            "nolibrarysemi": "No-RiskLibrary-SemiSupervised",
            "no_cnn_bilstm_transformer": "No-CNN-BiLSTM-Transformer",
            "nocnnbilstmtransformer": "No-CNN-BiLSTM-Transformer",
            "shallow": "Shallow-Supervised",
            "shallow_supervised": "Shallow-Supervised",
        }
    )
    result = []
    for item in raw.split(","):
        key = item.strip().lower().replace(" ", "")
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown ablation variant: {item}")
        value = aliases[key]
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("At least one ablation variant is required.")
    return result


def variant_slug(variant_names: list[str]) -> str:
    if variant_names == list(ABLATION_VARIANTS):
        return "all"
    slug = "_".join(name.lower().replace("-", "_") for name in variant_names)
    if len(slug) > 120:
        digest = hashlib.md5(slug.encode("utf-8")).hexdigest()[:10]
        return f"custom_{len(variant_names)}_{digest}"
    return slug


def run_once(
    config_path: Path,
    variant_name: str,
    run_id: int,
    seed: int,
) -> tuple[dict[str, object], bool]:
    config = load_experiment_config(config_path)
    config.data.seed = seed
    config.train.seed = seed
    set_seed(seed)
    device = choose_device(config)
    bundle = load_risk_dataset(config.data)
    loaders = make_loaders(config, bundle)
    metrics = run_ablation(config, bundle, loaders, device, ABLATION_VARIANTS[variant_name])
    return (
        {
            "variant": variant_name,
            "run": float(run_id),
            "seed": float(seed),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "auc": float(metrics["auc"]),
            "fpr": float(metrics["fpr"]),
            "udr": float(metrics["udr"]),
        },
        bool(bundle.used_synthetic_data),
    )


def make_loaders(config: ExperimentConfig, bundle) -> dict[str, torch.utils.data.DataLoader]:
    return {
        "train": make_loader(
            bundle.train_x,
            bundle.train_sources,
            bundle.train_y_binary,
            bundle.train_y_class,
            bundle.train_labeled_mask,
            config.train.batch_size,
            shuffle=True,
        ),
        "val": make_loader(
            bundle.val_x,
            bundle.val_sources,
            bundle.val_y_binary,
            bundle.val_y_class,
            None,
            config.train.batch_size,
            shuffle=False,
        ),
        "test": make_loader(
            bundle.test_x,
            bundle.test_sources,
            bundle.test_y_binary,
            bundle.test_y_class,
            None,
            config.train.batch_size,
            shuffle=False,
        ),
    }


def run_ablation(
    config: ExperimentConfig,
    bundle,
    loaders: dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    switches: dict[str, bool],
    *,
    verbose: bool = False,
    artifact_dir: Path | None = None,
    model_factory=None,
) -> dict[str, object]:
    model_type = AblationRiskNet if model_factory is None else model_factory
    model = model_type(
        feature_dim=bundle.feature_dim,
        n_sources=bundle.n_sources,
        n_classes=bundle.n_classes,
        config=config.model,
        use_cnn=switches["use_cnn"],
        use_lstm=switches["use_lstm"],
        use_transformer=switches["use_transformer"],
        use_gate=switches["use_gate"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    ce_loss = nn.CrossEntropyLoss()
    if switches["use_library"]:
        risk_library = PrototypeRiskLibrary(
            n_classes=bundle.n_classes,
            embedding_dim=config.model.model_dim,
            device=device,
        )
    else:
        risk_library = NullRiskLibrary(switches.get("use_confidence_fallback", False))

    best_state = copy.deepcopy(model.state_dict())
    best_thresholds = default_risk_thresholds(bundle.n_classes)
    best_f1 = -1.0
    history = []
    best_epoch = 0
    for epoch in range(config.train.epochs):
        started = time.perf_counter()
        stats: dict[str, int] = {}
        loss = train_ablation_one_epoch(
            model,
            loaders["train"],
            optimizer,
            ce_loss,
            risk_library,
            config,
            device,
            use_library=switches["use_library"],
            use_semi=switches["use_semi"],
            use_contrastive=switches.get("use_contrastive", True),
            stats=stats,
        )
        if switches["use_library"]:
            update_risk_library(model, loaders.get("library", loaders["train"]), risk_library, device)
        val_pred = predict(model, loaders["val"], risk_library, config, device)
        thresholds = select_best_risk_thresholds(
            val_pred["y_class"],
            val_pred["scores"],
            bundle.n_classes,
        )
        val_metrics = evaluate_multiclass_predictions(
            val_pred,
            thresholds,
            bundle.n_classes,
            bundle.val_y_unknown,
        )
        if float(val_metrics["f1"]) > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_thresholds = thresholds
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        history.append({"epoch": epoch + 1, "loss": loss, "val_f1": float(val_metrics["f1"]),
                        "seconds": time.perf_counter() - started, **stats})
        if verbose:
            print(f"Epoch {epoch + 1}/{config.train.epochs}: loss={loss:.4f}, "
                  f"val_F1={val_metrics['f1']:.4f}, "
                  f"pseudo={stats.get('pseudo_samples', 0)}, "
                  f"contrastive_batches={stats.get('contrastive_batches', 0)}, "
                  f"seconds={history[-1]['seconds']:.1f}", flush=True)

    model.load_state_dict(best_state)
    if switches["use_library"]:
        update_risk_library(model, loaders.get("library", loaders["train"]), risk_library, device)
    test_pred = predict(model, loaders["test"], risk_library, config, device)
    metrics = evaluate_multiclass_predictions(
        test_pred,
        best_thresholds,
        bundle.n_classes,
        bundle.test_y_unknown,
    )
    if artifact_dir is not None:
        import json

        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact_dir / "predictions.npz",
                            y_true=test_pred["y_class"], pred=metrics["pred"],
                            probabilities=test_pred["probs"], scores=test_pred["scores"],
                            unknown_flags=test_pred["unknown"],
                            unknown_attack_mask=bundle.test_y_unknown,
                            thresholds=best_thresholds)
        torch.save({"model": best_state, "thresholds": best_thresholds.tolist(),
                    "best_epoch": best_epoch,
                    "library": ({"prototypes": risk_library.prototypes.cpu(),
                                 "thresholds": risk_library.thresholds.cpu(),
                                 "available": risk_library.available.cpu()}
                                if switches["use_library"] else None)},
                   artifact_dir / "checkpoint.pt")
        (artifact_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return metrics


def train_ablation_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss: nn.Module,
    risk_library: PrototypeRiskLibrary | NullRiskLibrary,
    config: ExperimentConfig,
    device: torch.device,
    use_library: bool,
    use_semi: bool,
    use_contrastive: bool = True,
    stats: dict[str, int] | None = None,
) -> float:
    model.train()
    losses = []
    if stats is not None:
        stats.update(labeled_samples=0, pseudo_samples=0, contrastive_batches=0)
    for x, sources, _, y_class, labeled in loader:
        x = x.to(device)
        sources = sources.to(device)
        y_class = y_class.to(device)
        labeled = labeled.to(device)

        logits, embeddings, _ = model(x, sources)
        loss = torch.zeros((), device=device)
        used_loss = False
        if torch.any(labeled):
            loss = loss + ce_loss(logits[labeled], y_class[labeled])
            used_loss = True
            if stats is not None:
                stats["labeled_samples"] += int(labeled.sum().item())

        pseudo_mask = torch.zeros_like(labeled, dtype=torch.bool)
        if use_semi:
            probs = torch.softmax(logits.detach(), dim=1)
            confidence, pseudo_labels = probs.max(dim=1)
            pseudo_mask = (~labeled) & (confidence >= config.train.pseudo_label_threshold)
            if torch.any(pseudo_mask):
                loss = loss + config.train.semi_supervised_weight * ce_loss(
                    logits[pseudo_mask],
                    pseudo_labels[pseudo_mask],
                )
                used_loss = True
                if stats is not None:
                    stats["pseudo_samples"] += int(pseudo_mask.sum().item())
        else:
            pseudo_labels = y_class

        if use_library and use_contrastive and config.train.contrastive_weight > 0:
            contrastive_labels = y_class.clone()
            contrastive_mask = labeled.clone()
            if use_semi:
                contrastive_labels[pseudo_mask] = pseudo_labels[pseudo_mask]
                contrastive_mask = contrastive_mask | pseudo_mask
            if torch.any(contrastive_mask):
                proto_loss = risk_library.contrastive_loss(
                    embeddings[contrastive_mask],
                    contrastive_labels[contrastive_mask],
                    config.train.prototype_temperature,
                )
                if proto_loss is not None:
                    loss = loss + config.train.contrastive_weight * proto_loss
                    used_loss = True
                    if stats is not None:
                        stats["contrastive_batches"] += 1

        if not used_loss:
            continue
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def add_average_rows(rows: list[dict[str, object]], variant_names: list[str]) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    for variant_name in variant_names:
        variant_rows = [row for row in rows if row["variant"] == variant_name]
        if not variant_rows:
            continue
        average = {"variant": variant_name, "run": "Average", "seed": ""}
        for column in ["precision", "recall", "f1", "auc", "fpr", "udr"]:
            average[column] = float(np.mean([float(row[column]) for row in variant_rows]))
        output.append(average)
    return output


def make_summary_rows(rows: list[dict[str, object]], variant_names: list[str]) -> list[dict[str, object]]:
    summary = []
    for variant_name in variant_names:
        variant_rows = [row for row in rows if row["variant"] == variant_name]
        if not variant_rows:
            continue
        summary.append(
            {
                "variant": variant_name,
                "precision": float(np.mean([float(row["precision"]) for row in variant_rows])),
                "recall": float(np.mean([float(row["recall"]) for row in variant_rows])),
                "f1": float(np.mean([float(row["f1"]) for row in variant_rows])),
                "auc": float(np.mean([float(row["auc"]) for row in variant_rows])),
                "fpr": float(np.mean([float(row["fpr"]) for row in variant_rows])),
                "udr": float(np.mean([float(row["udr"]) for row in variant_rows])),
            }
        )
    return summary


if __name__ == "__main__":
    main()
