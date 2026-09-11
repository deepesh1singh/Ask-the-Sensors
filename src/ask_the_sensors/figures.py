from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .config import Config, load_config

def _savefig(fig, path: Path, bottom_margin: Optional[float] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    if bottom_margin is not None:
        fig.subplots_adjust(bottom=bottom_margin)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def figure_accuracy_by_question_type(
    per_type_accuracy: Dict[str, float],
    overall_accuracy: float,
    correctness_rules: Dict[str, str],
    out_path: Path,
) -> Path:
    types = list(per_type_accuracy.keys())
    values = [per_type_accuracy[t] for t in types]
    types_with_overall = types + ["Overall"]
    values_with_overall = values + [overall_accuracy]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(types_with_overall)), 5))
    colors = ["#4C72B0"] * len(types) + ["#C44E52"]
    bars = ax.bar(types_with_overall, values_with_overall, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (per-type correctness rule)")
    ax.set_title("Figure 1: Accuracy by question type")
    ax.set_xticks(range(len(types_with_overall)))
    ax.set_xticklabels(types_with_overall, rotation=30, ha="right")

    for bar, val in zip(bars, values_with_overall):
        if val == val:  # not NaN
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=9)

    caption_lines = [f"{t}: {correctness_rules.get(t, 'n/a')}" for t in types]
    caption = "Correctness rule per group -> " + "; ".join(caption_lines)
    fig.text(0.01, 0.01, caption, wrap=True, fontsize=7, ha="left", va="bottom")

    _savefig(fig, out_path, bottom_margin=0.30)
    return out_path

def figure_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    per_class_metrics: Dict[str, Dict[str, float]],
    out_path: Path,
) -> Path:
    fig, (ax_cm, ax_table) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(labels)))
    ax_cm.set_yticks(range(len(labels)))
    ax_cm.set_xticklabels(labels, rotation=45, ha="right")
    ax_cm.set_yticklabels(labels)
    ax_cm.set_xlabel("Predicted activity")
    ax_cm.set_ylabel("True activity")
    ax_cm.set_title("Figure 2: Activity confusion matrix")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_table.axis("off")
    table_data = [["Class", "Precision", "Recall", "F1"]]
    for label in labels:
        m = per_class_metrics.get(label, {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")})
        table_data.append([label, f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}"])
    table = ax_table.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    _savefig(fig, out_path)
    return out_path

def figure_accuracy_vs_strictness(
    numeric_curve: Dict[float, float],
    iou_curve: Dict[float, float],
    out_path: Path,
) -> Path:
    fig, ax1 = plt.subplots(figsize=(8, 5))

    if numeric_curve:
        x_num = sorted(numeric_curve.keys())
        y_num = [numeric_curve[x] for x in x_num]
        ax1.plot(x_num, y_num, marker="o", label="Numeric answers (tolerance, seconds)", color="#4C72B0")
    ax1.set_xlabel("Numeric error tolerance (seconds)")
    ax1.set_ylabel("Fraction of answers accepted")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twiny()
    if iou_curve:
        x_iou = sorted(iou_curve.keys())
        y_iou = [iou_curve[x] for x in x_iou]
        ax2.plot(x_iou, y_iou, marker="s", label="Temporal/evidence answers (IoU threshold)", color="#C44E52")
    ax2.set_xlabel("IoU threshold")
    ax2.set_xlim(0, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax1.set_title("Figure 3: Accuracy versus strictness")
    _savefig(fig, out_path)
    return out_path

def figure_accuracy_vs_overhead(
    points: List[Dict[str, float]],
    cost_key: str,
    cost_label: str,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    xs = [p[cost_key] for p in points]
    ys = [p["accuracy"] for p in points]
    names = [p["name"] for p in points]

    ax.scatter(xs, ys, s=80, color="#4C72B0", zorder=3)
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    order = np.argsort(xs)
    frontier_x, frontier_y = [], []
    best_acc = -np.inf
    for idx in order:
        if ys[idx] >= best_acc:
            frontier_x.append(xs[idx])
            frontier_y.append(ys[idx])
            best_acc = ys[idx]
    if len(frontier_x) >= 2:
        ax.plot(frontier_x, frontier_y, linestyle="--", color="#C44E52", label="Pareto frontier", zorder=2)
        ax.legend()

    ax.set_xlabel(cost_label)
    ax.set_ylabel("Overall QA accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 4: Accuracy versus overhead")
    _savefig(fig, out_path)
    return out_path

def figure_robustness_curve(
    degradation_values: Sequence[float],
    accuracies: Sequence[float],
    degradation_label: str,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degradation_values, accuracies, marker="o", color="#55A868")
    ax.set_xlabel(degradation_label)
    ax.set_ylabel("Accuracy on fixed question set")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 5: Robustness curve")
    _savefig(fig, out_path)
    return out_path