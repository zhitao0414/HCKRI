from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        zero_division=0,
    )
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        **false_rate_metrics(y_true, pred, [1]),
    }


def multiclass_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    pred: np.ndarray,
    n_classes: int,
    metric_labels: list[int] | None = None,
) -> dict[str, float]:
    labels = list(range(n_classes))
    metric_labels = metric_labels or labels
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=metric_labels,
        average="macro",
        zero_division=0,
    )
    auc = macro_ovr_auc(y_true, probabilities, metric_labels)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        **false_rate_metrics(y_true, pred, metric_labels),
    }


def false_rate_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    labels: list[int],
) -> dict[str, float]:
    fnrs = []
    fprs = []
    for label in labels:
        true_positive = np.sum((y_true == label) & (pred == label))
        false_negative = np.sum((y_true == label) & (pred != label))
        false_positive = np.sum((y_true != label) & (pred == label))
        true_negative = np.sum((y_true != label) & (pred != label))
        fn_denom = true_positive + false_negative
        fp_denom = false_positive + true_negative
        if fn_denom > 0:
            fnrs.append(float(false_negative / fn_denom))
        if fp_denom > 0:
            fprs.append(float(false_positive / fp_denom))
    return {
        "fnr": float(np.mean(fnrs)) if fnrs else float("nan"),
        "fpr": float(np.mean(fprs)) if fprs else float("nan"),
    }


def per_class_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    class_names: list[str],
    probabilities: np.ndarray | None = None,
    include_normal: bool = False,
) -> list[dict[str, float | str]]:
    labels = [
        idx
        for idx, name in enumerate(class_names)
        if include_normal or str(name) != "正常"
    ]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    rows = []
    for row_idx, class_id in enumerate(labels):
        name = class_names[class_id]
        auc = (
            one_vs_rest_auc(y_true, probabilities[:, class_id], class_id)
            if probabilities is not None
            else float("nan")
        )
        rows.append(
            {
                "class_id": class_id,
                "risk_level": name,
                "recall": float(recall[row_idx]),
                "f1": float(f1[row_idx]),
                "auc": auc,
                "support": int(support[row_idx]),
            }
        )
    return rows


def macro_ovr_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[int],
) -> float:
    aucs = [
        one_vs_rest_auc(y_true, probabilities[:, label], label)
        for label in labels
        if label < probabilities.shape[1]
    ]
    aucs = [value for value in aucs if value == value]
    return float(np.mean(aucs)) if aucs else float("nan")


def one_vs_rest_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: int,
) -> float:
    y_binary = (y_true == label).astype(int)
    if len(np.unique(y_binary)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_binary, scores))
    except ValueError:
        return float("nan")


def multiclass_confusion_rows(
    y_true: np.ndarray,
    pred: np.ndarray,
    class_names: list[str],
) -> list[dict[str, int | str]]:
    rows = []
    for true_idx, true_name in enumerate(class_names):
        row: dict[str, int | str] = {"actual": true_name}
        true_mask = y_true == true_idx
        for pred_idx, pred_name in enumerate(class_names):
            row[pred_name] = int(np.sum(true_mask & (pred == pred_idx)))
        rows.append(row)
    return rows


def confusion_counts(y_true: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tn": int(np.sum((y_true == 0) & (pred == 0))),
        "fp": int(np.sum((y_true == 0) & (pred == 1))),
        "fn": int(np.sum((y_true == 1) & (pred == 0))),
        "tp": int(np.sum((y_true == 1) & (pred == 1))),
    }


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "I"
    if score >= 0.50:
        return "II"
    if score >= 0.25:
        return "III"
    return "IV"


def risk_level_distribution(scores: np.ndarray) -> dict[str, int]:
    levels = [risk_level(float(score)) for score in scores]
    return {level: levels.count(level) for level in ["I", "II", "III", "IV"]}


def risk_level_distribution_from_classes(
    pred: np.ndarray,
    class_names: list[str],
) -> dict[str, int]:
    return {
        class_name: int(np.sum(pred == class_id))
        for class_id, class_name in enumerate(class_names)
        if class_name != "正常"
    }
