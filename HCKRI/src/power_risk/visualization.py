from __future__ import annotations

from pathlib import Path

import numpy as np


def _configure_chinese_fonts() -> None:
    import matplotlib
    from matplotlib import font_manager

    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_score_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_chinese_fonts()
    normal_scores = scores[y_true == 0]
    risk_scores = scores[y_true == 1]
    bins = np.linspace(float(scores.min()), float(scores.max()), 24)
    if np.allclose(bins[0], bins[-1]):
        bins = 10

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(normal_scores, bins=bins, density=True, alpha=0.65, label="Normal")
    ax.hist(risk_scores, bins=bins, density=True, alpha=0.55, label="Risk")
    ax.axvline(0.5, color="#111111", linestyle="--", linewidth=1.4, label="Risk threshold")
    ax.set_title("Normal/Risk Score Distribution")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Density")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    auc: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    _configure_chinese_fonts()
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, linewidth=2.0, label=f"ROC AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.2)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_multiclass_score_distribution(
    y_class: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_chinese_fonts()
    bins = np.linspace(float(scores.min()), float(scores.max()), 26)
    if np.allclose(bins[0], bins[-1]):
        bins = 10

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for class_id, class_name in enumerate(class_names):
        class_scores = scores[y_class == class_id]
        if len(class_scores) == 0:
            continue
        ax.hist(
            class_scores,
            bins=bins,
            density=True,
            alpha=0.46,
            label=class_name,
        )
    ax.set_title("Risk Score Distribution by Level")
    ax.set_xlabel("Risk score R")
    ax.set_ylabel("Density")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_multiclass_roc_curve(
    y_class: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    _configure_chinese_fonts()
    labels = list(range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    if len(np.unique(y_class)) > 1:
        y_bin = label_binarize(y_class, classes=labels)
        for class_id, class_name in enumerate(class_names):
            if y_bin[:, class_id].max() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, class_id], probabilities[:, class_id])
            ax.plot(fpr, tpr, linewidth=1.8, label=f"{class_name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.2)
    ax.set_title("One-vs-Rest ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(loc="lower right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_embedding_pca_by_class(
    embeddings: np.ndarray,
    y_class: np.ndarray,
    class_names: list[str],
    output_path: Path,
    unknown_mask: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    _configure_chinese_fonts()
    if embeddings.shape[1] < 2:
        z2 = np.column_stack([embeddings[:, 0], np.zeros(len(embeddings))])
    else:
        z2 = PCA(n_components=2, random_state=0).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for class_id, class_name in enumerate(class_names):
        mask = y_class == class_id
        if not np.any(mask):
            continue
        ax.scatter(z2[mask, 0], z2[mask, 1], s=28, alpha=0.68, label=class_name)
    if unknown_mask is not None and np.any(unknown_mask):
        ax.scatter(
            z2[unknown_mask, 0],
            z2[unknown_mask, 1],
            s=54,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.2,
            label="Held-out unknown attack",
        )
    ax.set_title("Transformer Embedding Space by Risk Level (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_embedding_pca(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    y_class: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    _configure_chinese_fonts()
    if embeddings.shape[1] < 2:
        z2 = np.column_stack([embeddings[:, 0], np.zeros(len(embeddings))])
    else:
        z2 = PCA(n_components=2, random_state=0).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    normal_mask = y_true == 0
    ax.scatter(
        z2[normal_mask, 0],
        z2[normal_mask, 1],
        s=28,
        alpha=0.70,
        label="Normal",
    )
    for class_id in sorted(set(y_class[y_true == 1].tolist())):
        mask = y_class == class_id
        label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        ax.scatter(z2[mask, 0], z2[mask, 1], s=36, marker="x", label=label)
    ax.set_title("Transformer Embedding Space (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
