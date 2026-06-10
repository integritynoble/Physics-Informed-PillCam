"""Make Precision-Recall curves for RGB / +PI 5-ch / +Distill, alongside the
existing Fig 3 ROC. Pooled across 6 seeds; per-class panels (4 of the 11
evaluable classes that have ≥20 positives in test).

Reads test_predictions.npz from effb0_paper_seed*_{rgb,pi}.
Writes figures/fig_pr_curves.{pdf,png}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path("/project/BME/Zaman_lab/s248103/GI_outputs/cross_backbone")
FIG_DIR = Path("/home2/s248103/abraham/GI/GI_Multi_Task/paper/Capsule-Endoscopy/medIA_submission_new1/figures")
SEEDS = [41, 42, 43, 44, 45, 47]

ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]


def _pool(arm: str):
    probs_all, labels_all = [], []
    for s in SEEDS:
        p = ROOT / f"effb0_paper_seed{s}_{arm}" / "test_predictions.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        probs_all.append(z["probs"])
        labels_all.append(z["labels"])
    return np.concatenate(probs_all, axis=0), np.concatenate(labels_all, axis=0)


def main():
    rgb_probs, rgb_labels = _pool("rgb")
    pi_probs,  pi_labels  = _pool("pi")
    assert (rgb_labels == pi_labels).all(), "label mismatch between arms"

    focal = ["Lymphangiectasia", "Blood - fresh", "Angiectasia", "Erosion"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, c in zip(axes.flat, focal):
        idx = ALL_CLASSES.index(c)
        y = (rgb_labels == idx).astype(int)
        if y.sum() < 5:
            ax.text(0.5, 0.5, f"{c}: insufficient positives", ha="center")
            ax.set_axis_off(); continue
        for arm, probs, color, lab in (
            ("RGB-only",      rgb_probs, "#999999", "RGB EfficientNet-B0"),
            ("+PI 5-channel", pi_probs,  "#1f77b4", "+PI 5-channel"),
        ):
            p, r, _ = precision_recall_curve(y, probs[:, idx])
            ap = average_precision_score(y, probs[:, idx])
            ax.plot(r, p, color=color, label=f"{lab}  AP={ap:.3f}", lw=1.8)
        baseline = float(y.mean())
        ax.axhline(baseline, ls="--", color="0.6", lw=0.8,
                    label=f"prevalence {baseline:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"{c}  (n_pos={int(y.sum()):,})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Precision-Recall curves on Kvasir-Capsule test set (frames pooled across 6 seeds)",
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_pr_curves.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_pr_curves.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_DIR/'fig_pr_curves.pdf'} and .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
