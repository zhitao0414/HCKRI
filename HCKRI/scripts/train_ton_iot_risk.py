from __future__ import annotations

import copy
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from power_risk import MultiSourceRiskNet, load_experiment_config, load_risk_dataset
from power_risk.config import ExperimentConfig
from power_risk.metrics import (
    binary_metrics,
    confusion_counts,
    multiclass_confusion_rows,
    multiclass_metrics,
    per_class_metrics,
    risk_level_distribution_from_classes,
)
from power_risk.reporting import format_markdown_table, with_average_row, write_csv, write_markdown_table
from power_risk.visualization import (
    plot_embedding_pca_by_class,
    plot_multiclass_roc_curve,
    plot_multiclass_score_distribution,
)


class RiskWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        sources: np.ndarray,
        y_binary: np.ndarray,
        y_class: np.ndarray,
        labeled_mask: np.ndarray | None = None,
    ) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.sources = torch.tensor(sources, dtype=torch.long)
        self.y_binary = torch.tensor(y_binary, dtype=torch.long)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        if labeled_mask is None:
            labeled_mask = np.ones(len(y_class), dtype=bool)
        self.labeled_mask = torch.tensor(labeled_mask, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x[index],
            self.sources[index],
            self.y_binary[index],
            self.y_class[index],
            self.labeled_mask[index],
        )


class PrototypeRiskLibrary:
    """External memory for typical risk features."""

    def __init__(self, n_classes: int, embedding_dim: int, device: torch.device) -> None:
        self.prototypes = torch.zeros(n_classes, embedding_dim, device=device)
        self.thresholds = torch.full((n_classes,), float("inf"), device=device)
        self.available = torch.zeros(n_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if embeddings.numel() == 0:
            return
        embeddings = embeddings[mask]
        labels = labels[mask]
        if len(labels) == 0:
            return
        for class_id in torch.unique(labels):
            class_mask = labels == class_id
            class_embeddings = embeddings[class_mask]
            if len(class_embeddings) == 0:
                continue
            center = class_embeddings.mean(dim=0)
            distances = torch.linalg.norm(class_embeddings - center, dim=1)
            threshold = torch.quantile(distances, 0.95) + 0.02
            idx = int(class_id.item())
            self.prototypes[idx] = center
            self.thresholds[idx] = threshold
            self.available[idx] = True

    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor | None:
        valid = self.available[labels]
        if not torch.any(valid):
            return None
        embeddings = embeddings[valid]
        labels = labels[valid]
        prototypes = nn.functional.normalize(self.prototypes, dim=1)
        embeddings = nn.functional.normalize(embeddings, dim=1)
        logits = embeddings @ prototypes.T / max(temperature, 1e-6)
        return nn.functional.cross_entropy(logits, labels)

    @torch.no_grad()
    def unknown_flags(
        self,
        embeddings: torch.Tensor,
        confidence: torch.Tensor,
        confidence_threshold: float,
    ) -> torch.Tensor:
        low_confidence = confidence < confidence_threshold
        if not torch.any(self.available):
            return low_confidence
        distances = torch.cdist(embeddings, self.prototypes)
        unavailable = ~self.available
        distances[:, unavailable] = float("inf")
        nearest_distance, nearest_class = distances.min(dim=1)
        class_threshold = self.thresholds[nearest_class]
        far_from_library = nearest_distance > class_threshold
        return low_confidence | far_from_library

    def to_jsonable(self, class_names: list[str]) -> list[dict[str, object]]:
        rows = []
        for idx, name in enumerate(class_names):
            rows.append(
                {
                    "class_id": idx,
                    "class_name": name,
                    "available": bool(self.available[idx].detach().cpu().item()),
                    "distance_threshold": float(self.thresholds[idx].detach().cpu().item()),
                    "prototype": self.prototypes[idx].detach().cpu().tolist(),
                }
            )
        return rows


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/ton_iot_risk.json")
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_path = Path(sys.argv[2])
    config = load_experiment_config(config_path)
    set_seed(config.train.seed)
    device = choose_device(config)

    bundle = load_risk_dataset(config.data)
    print_dataset_summary(bundle)

    train_loader = make_loader(
        bundle.train_x,
        bundle.train_sources,
        bundle.train_y_binary,
        bundle.train_y_class,
        bundle.train_labeled_mask,
        config.train.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        bundle.val_x,
        bundle.val_sources,
        bundle.val_y_binary,
        bundle.val_y_class,
        None,
        config.train.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        bundle.test_x,
        bundle.test_sources,
        bundle.test_y_binary,
        bundle.test_y_class,
        None,
        config.train.batch_size,
        shuffle=False,
    )

    model = MultiSourceRiskNet(
        feature_dim=bundle.feature_dim,
        n_sources=bundle.n_sources,
        n_classes=bundle.n_classes,
        config=config.model,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    ce_loss = nn.CrossEntropyLoss()
    risk_library = PrototypeRiskLibrary(
        n_classes=bundle.n_classes,
        embedding_dim=config.model.model_dim,
        device=device,
    )

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_thresholds = default_risk_thresholds(bundle.n_classes)
    for epoch in range(1, config.train.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            ce_loss,
            risk_library,
            config,
            device,
        )
        update_risk_library(model, train_loader, risk_library, device)
        val_pred = predict(model, val_loader, risk_library, config, device)
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
        record = {
            "epoch": float(epoch),
            "loss": train_loss,
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "auc": val_metrics["auc"],
            "udr": val_metrics["udr"],
            "fpr": val_metrics["fpr"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d} | "
            f"Loss={train_loss:.4f} | "
            f"P={record['precision']:.3f} | "
            f"R={record['recall']:.3f} | "
            f"F1={record['f1']:.3f} | "
            f"AUC={record['auc']:.3f} | "
            f"UDR={record['udr']:.3f} | "
            f"FPR={record['fpr']:.3f}"
        )
        if record["f1"] > best_f1:
            best_f1 = record["f1"]
            best_thresholds = thresholds
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    update_risk_library(model, train_loader, risk_library, device)
    test_pred = predict(model, test_loader, risk_library, config, device)
    test_metrics = evaluate_multiclass_predictions(
        test_pred,
        best_thresholds,
        bundle.n_classes,
        bundle.test_y_unknown,
    )
    confusion = multiclass_confusion_rows(
        test_pred["y_class"],
        test_metrics["pred"],
        bundle.class_names,
    )
    level_counts = risk_level_distribution_from_classes(
        test_metrics["pred"],
        bundle.class_names,
    )

    output_paths = save_outputs(
        config,
        bundle,
        history,
        test_pred,
        test_metrics,
        confusion,
        level_counts,
        risk_library,
        best_thresholds,
    )
    print_final_report(history, test_metrics, confusion, level_counts, output_paths)


def train_one_epoch(
    model: MultiSourceRiskNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss: nn.Module,
    risk_library: PrototypeRiskLibrary,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for x, sources, _, y_class, labeled in loader:
        x = x.to(device)
        sources = sources.to(device)
        y_class = y_class.to(device)
        labeled = labeled.to(device)

        logits, embeddings, _ = model(x, sources)
        loss = torch.zeros((), device=device)
        used_loss = False
        if torch.any(labeled):
            supervised = ce_loss(logits[labeled], y_class[labeled])
            loss = loss + supervised
            used_loss = True

        probs = torch.softmax(logits.detach(), dim=1)
        confidence, pseudo_labels = probs.max(dim=1)
        unlabeled = ~labeled
        pseudo_mask = unlabeled & (confidence >= config.train.pseudo_label_threshold)
        if torch.any(pseudo_mask):
            pseudo = ce_loss(logits[pseudo_mask], pseudo_labels[pseudo_mask])
            loss = loss + config.train.semi_supervised_weight * pseudo
            used_loss = True

        contrastive_labels = y_class.clone()
        contrastive_mask = labeled.clone()
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

        if not used_loss:
            continue
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def update_risk_library(
    model: MultiSourceRiskNet,
    loader: DataLoader,
    risk_library: PrototypeRiskLibrary,
    device: torch.device,
) -> None:
    model.eval()
    embeddings_list, labels_list, mask_list = [], [], []
    for x, sources, _, y_class, labeled in loader:
        x = x.to(device)
        sources = sources.to(device)
        _, embeddings, _ = model(x, sources)
        embeddings_list.append(embeddings)
        labels_list.append(y_class.to(device))
        mask_list.append(labeled.to(device))
    risk_library.update(
        torch.cat(embeddings_list, dim=0),
        torch.cat(labels_list, dim=0),
        torch.cat(mask_list, dim=0),
    )


@torch.no_grad()
def predict(
    model: MultiSourceRiskNet,
    loader: DataLoader,
    risk_library: PrototypeRiskLibrary,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    y_binary_list, y_class_list, probs_list, embeddings_list, unknown_list = [], [], [], [], []
    for x, sources, y_binary, y_class, _ in loader:
        x = x.to(device)
        sources = sources.to(device)
        logits, embeddings, _ = model(x, sources)
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max(dim=1).values
        unknown = risk_library.unknown_flags(
            embeddings,
            confidence,
            config.train.unknown_confidence_threshold,
        )
        y_binary_list.append(y_binary.numpy())
        y_class_list.append(y_class.numpy())
        probs_list.append(probs.detach().cpu().numpy())
        embeddings_list.append(embeddings.detach().cpu().numpy())
        unknown_list.append(unknown.detach().cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    if probs.shape[1] > 1:
        severity = np.linspace(0.0, 1.0, probs.shape[1], dtype=np.float32)
        scores = probs @ severity
    else:
        scores = probs[:, 0]
    return {
        "y_binary": np.concatenate(y_binary_list, axis=0),
        "y_class": np.concatenate(y_class_list, axis=0),
        "probs": probs,
        "scores": scores,
        "embeddings": np.concatenate(embeddings_list, axis=0),
        "unknown": np.concatenate(unknown_list, axis=0).astype(bool),
    }


def evaluate_predictions(prediction: dict[str, np.ndarray], threshold: float) -> dict[str, object]:
    pred_binary = (prediction["scores"] >= threshold).astype(int)
    pred_binary = np.where(prediction["unknown"], 1, pred_binary)
    metrics = binary_metrics(prediction["y_binary"], prediction["scores"], pred_binary)
    metrics["pred"] = pred_binary
    metrics["unknown_rate"] = float(prediction["unknown"].mean())
    return metrics


def evaluate_multiclass_predictions(
    prediction: dict[str, np.ndarray],
    thresholds: np.ndarray,
    n_classes: int,
    unknown_attack_mask: np.ndarray | None = None,
) -> dict[str, object]:
    pred_class = risk_classes_from_scores(prediction["scores"], thresholds, n_classes)
    pred_class = np.where(prediction["unknown"], n_classes - 1, pred_class)
    metrics = multiclass_metrics(
        prediction["y_class"],
        prediction["probs"],
        pred_class,
        n_classes,
        metric_labels=risk_metric_labels(n_classes),
    )
    unknown_mask = (
        np.asarray(unknown_attack_mask, dtype=bool)
        if unknown_attack_mask is not None
        else np.zeros(len(pred_class), dtype=bool)
    )
    if len(unknown_mask) != len(pred_class):
        unknown_mask = np.zeros(len(pred_class), dtype=bool)
    if np.any(unknown_mask):
        unknown_attack_recall = float(np.mean(pred_class[unknown_mask] > 0))
        unknown_flag_recall = float(np.mean(prediction["unknown"][unknown_mask]))
    else:
        unknown_attack_recall = float("nan")
        unknown_flag_recall = float("nan")
    metrics["pred"] = pred_class
    metrics["unknown_rate"] = float(prediction["unknown"].mean())
    metrics["udr"] = unknown_attack_recall
    metrics["unknown_attack_recall"] = unknown_attack_recall
    metrics["unknown_flag_recall"] = unknown_flag_recall
    return metrics


def risk_metric_labels(n_classes: int) -> list[int]:
    if n_classes <= 1:
        return [0]
    return list(range(1, n_classes))


def default_risk_thresholds(n_classes: int) -> np.ndarray:
    if n_classes <= 1:
        return np.array([], dtype=np.float32)
    return np.linspace(0.0, 1.0, n_classes + 1, dtype=np.float32)[1:-1]


def risk_classes_from_scores(
    scores: np.ndarray,
    thresholds: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    if n_classes <= 1:
        return np.zeros(len(scores), dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    thresholds = np.sort(thresholds[: n_classes - 1])
    return np.digitize(scores, thresholds, right=False).clip(0, n_classes - 1).astype(np.int64)


def select_best_risk_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    if n_classes <= 1:
        return np.array([], dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return default_risk_thresholds(n_classes)

    base = default_risk_thresholds(n_classes)
    quantiles = np.linspace(0.05, 0.95, 19)
    candidates = np.unique(np.r_[base, np.quantile(scores, quantiles)])
    candidates = candidates[(candidates > 0.0) & (candidates < 1.0)]
    if len(candidates) < n_classes - 1:
        return base

    best_thresholds = base
    best_f1 = -1.0
    for values in combinations(candidates.tolist(), n_classes - 1):
        thresholds = np.array(values, dtype=np.float32)
        pred = risk_classes_from_scores(scores, thresholds, n_classes)
        score = macro_f1_for_labels(y_true, pred, risk_metric_labels(n_classes))
        if score > best_f1:
            best_f1 = float(score)
            best_thresholds = thresholds
    return best_thresholds


def macro_f1_for_labels(y_true: np.ndarray, pred: np.ndarray, labels: list[int]) -> float:
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (pred == label))
        fp = np.sum((y_true != label) & (pred == label))
        fn = np.sum((y_true == label) & (pred != label))
        denominator = 2 * tp + fp + fn
        scores.append(float(2 * tp / denominator) if denominator > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def select_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.unique(np.r_[scores, 0.5])
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1 or (np.isclose(score, best_f1) and threshold > best_threshold):
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def save_outputs(
    config: ExperimentConfig,
    bundle,
    history: list[dict[str, float]],
    test_pred: dict[str, np.ndarray],
    test_metrics: dict[str, object],
    confusion: list[dict[str, int | str]],
    level_counts: dict[str, int],
    risk_library: PrototypeRiskLibrary,
    best_thresholds: np.ndarray,
) -> dict[str, Path]:
    results_path = Path(config.output.results_path)
    stem = results_path.with_suffix("")
    epoch_columns = [
        "epoch",
        "precision",
        "recall",
        "f1",
        "auc",
        "udr",
        "fpr",
    ]
    epoch_rows = with_average_row(
        history,
        label_column="epoch",
        metric_columns=["precision", "recall", "f1", "auc", "udr", "fpr"],
    )
    write_csv(stem.with_name(f"{stem.name}_epoch_metrics.csv"), epoch_rows, epoch_columns)
    write_markdown_table(
        stem.with_name(f"{stem.name}_epoch_metrics.md"),
        epoch_rows,
        epoch_columns,
        headers=["Epoch", "P", "R", "F1", "AUC", "UDR", "FPR"],
    )

    final_rows = [
        {
            "split": "Test",
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "auc": test_metrics["auc"],
            "udr": test_metrics["udr"],
            "fpr": test_metrics["fpr"],
        }
    ]
    final_columns = [
        "split",
        "precision",
        "recall",
        "f1",
        "auc",
        "udr",
        "fpr",
    ]
    write_csv(stem.with_name(f"{stem.name}_final_metrics.csv"), final_rows, final_columns)
    write_markdown_table(
        stem.with_name(f"{stem.name}_final_metrics.md"),
        final_rows,
        final_columns,
        headers=["Split", "P", "R", "F1", "AUC", "UDR", "FPR"],
    )

    per_class_rows = per_class_metrics(
        test_pred["y_class"],
        test_metrics["pred"],
        bundle.class_names,
        probabilities=test_pred["probs"],
        include_normal=False,
    )
    per_class_columns = ["class_id", "risk_level", "recall", "f1", "auc"]
    write_csv(stem.with_name(f"{stem.name}_per_class_metrics.csv"), per_class_rows, per_class_columns)
    write_markdown_table(
        stem.with_name(f"{stem.name}_per_class_metrics.md"),
        per_class_rows,
        per_class_columns,
        headers=["Class", "Risk Level", "R", "F1", "AUC"],
    )

    confusion_columns = ["actual", *bundle.class_names]
    write_csv(stem.with_name(f"{stem.name}_confusion_matrix.csv"), confusion, confusion_columns)
    write_markdown_table(
        stem.with_name(f"{stem.name}_confusion_matrix.md"),
        confusion,
        confusion_columns,
        headers=["Actual", *bundle.class_names],
    )

    level_rows = [{"risk_level": level, "count": count} for level, count in level_counts.items()]
    write_csv(stem.with_name(f"{stem.name}_risk_levels.csv"), level_rows, ["risk_level", "count"])
    write_markdown_table(
        stem.with_name(f"{stem.name}_risk_levels.md"),
        level_rows,
        ["risk_level", "count"],
        headers=["Risk Level", "Count"],
    )

    score_path = stem.with_name(f"{stem.name}_score_distribution.png")
    roc_path = stem.with_name(f"{stem.name}_roc_curve.png")
    pca_path = stem.with_name(f"{stem.name}_embedding_pca.png")
    plot_multiclass_score_distribution(
        test_pred["y_class"],
        test_pred["scores"],
        bundle.class_names,
        score_path,
    )
    plot_multiclass_roc_curve(
        test_pred["y_class"],
        test_pred["probs"],
        bundle.class_names,
        roc_path,
    )
    plot_embedding_pca_by_class(
        test_pred["embeddings"],
        test_pred["y_class"],
        bundle.class_names,
        pca_path,
        unknown_mask=bundle.test_y_unknown,
    )

    payload = {
        "dataset": {
            "used_synthetic_data": bundle.used_synthetic_data,
            "n_sources": bundle.n_sources,
            "source_names": bundle.source_names,
            "n_classes": bundle.n_classes,
            "class_names": bundle.class_names,
            "unknown_attack_types": bundle.unknown_attack_types,
            "feature_dim": bundle.feature_dim,
            "train_windows": int(len(bundle.train_x)),
            "val_windows": int(len(bundle.val_x)),
            "test_windows": int(len(bundle.test_x)),
            "train_labeled_windows": int(bundle.train_labeled_mask.sum()),
            "unknown_test_windows": int(bundle.test_y_unknown.sum()),
        },
        "risk_score_thresholds": [float(value) for value in best_thresholds],
        "history": history,
        "test_metrics": {
            key: float(value)
            for key, value in test_metrics.items()
            if isinstance(value, float | int)
        },
        "confusion_matrix": confusion,
        "risk_levels": level_counts,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    library_path = stem.with_name(f"{stem.name}_risk_feature_library.json")
    library_path.write_text(
        json.dumps(risk_library.to_jsonable(bundle.class_names), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "json": results_path,
        "epoch_md": stem.with_name(f"{stem.name}_epoch_metrics.md"),
        "final_md": stem.with_name(f"{stem.name}_final_metrics.md"),
        "per_class_md": stem.with_name(f"{stem.name}_per_class_metrics.md"),
        "confusion_md": stem.with_name(f"{stem.name}_confusion_matrix.md"),
        "risk_levels_md": stem.with_name(f"{stem.name}_risk_levels.md"),
        "library": library_path,
        "score_plot": score_path,
        "roc_plot": roc_path,
        "pca_plot": pca_path,
    }


def make_loader(
    x: np.ndarray,
    sources: np.ndarray,
    y_binary: np.ndarray,
    y_class: np.ndarray,
    labeled_mask: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        RiskWindowDataset(x, sources, y_binary, y_class, labeled_mask),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def choose_device(config: ExperimentConfig) -> torch.device:
    if config.train.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.train.device)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_dataset_summary(bundle) -> None:
    print("Dataset summary")
    print(
        json.dumps(
            {
                "used_synthetic_data": bundle.used_synthetic_data,
                "n_sources": bundle.n_sources,
                "source_names": bundle.source_names,
                "n_classes": bundle.n_classes,
                "class_names": bundle.class_names,
                "unknown_attack_types": bundle.unknown_attack_types,
                "feature_dim": bundle.feature_dim,
                "train_windows": int(len(bundle.train_x)),
                "val_windows": int(len(bundle.val_x)),
                "test_windows": int(len(bundle.test_x)),
                "train_labeled_windows": int(bundle.train_labeled_mask.sum()),
                "test_unknown_windows": int(bundle.test_y_unknown.sum()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def print_final_report(
    history: list[dict[str, float]],
    test_metrics: dict[str, object],
    confusion: list[dict[str, int | str]],
    level_counts: dict[str, int],
    output_paths: dict[str, Path],
) -> None:
    epoch_rows = with_average_row(
        history,
        label_column="epoch",
        metric_columns=["precision", "recall", "f1", "auc", "udr", "fpr"],
    )
    print("\nEpoch metrics")
    print(
        format_markdown_table(
            epoch_rows,
            [
                "epoch",
                "precision",
                "recall",
                "f1",
                "auc",
                "udr",
                "fpr",
            ],
            headers=["Epoch", "P", "R", "F1", "AUC", "UDR", "FPR"],
        )
    )
    print("\nTest metrics")
    print(
        format_markdown_table(
            [
                {
                    "split": "Test",
                    "precision": test_metrics["precision"],
                    "recall": test_metrics["recall"],
                    "f1": test_metrics["f1"],
                    "auc": test_metrics["auc"],
                    "udr": test_metrics["udr"],
                    "fpr": test_metrics["fpr"],
                }
            ],
            [
                "split",
                "precision",
                "recall",
                "f1",
                "auc",
                "udr",
                "fpr",
            ],
            headers=["Split", "P", "R", "F1", "AUC", "UDR", "FPR"],
        )
    )
    print("\nConfusion matrix")
    print(
        format_markdown_table(
            confusion,
            list(confusion[0].keys()) if confusion else ["actual"],
        )
    )
    print("\nRisk levels")
    print(
        format_markdown_table(
            [{"risk_level": level, "count": count} for level, count in level_counts.items()],
            ["risk_level", "count"],
            headers=["Risk Level", "Count"],
        )
    )
    print("\nSaved outputs")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
