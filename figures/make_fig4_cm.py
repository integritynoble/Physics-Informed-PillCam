"""Figure 4: Side-by-side normalized confusion matrices for the 3 conditions.

Reads test_predictions.json files. Rows normalized to sum to 1 so per-class
recall/sensitivity is directly readable on the diagonal.

Usage:
    python make_fig4_cm.py \
      --rgb     D:/kvasir_capsule/outputs/stage2_rgb_effb0/test_predictions.json \
      --pi      D:/kvasir_capsule/outputs/stage2_pi_effb0/test_predictions.json \
      --distill D:/kvasir_capsule/outputs/stage2_distill_effb0/test_predictions.json \
      --out     fig4_confusion_matrices.png
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


CONDITION_NAMES = {
    "rgb": "RGB-only baseline",
    "pi": "+MC prior (5-channel)",
    "distill": "+Distillation (3-channel + aux head)",
}


def load(path: Optional[str]) -> Optional[Dict]:
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def short(name: str) -> str:
    """Compact class label for axis tick labels."""
    # Kvasir-Capsule metadata.csv uses Title Case for "Ampulla of Vater",
    # "Reduced Mucosal View", and "Foreign Body" (verified against the v1.0
    # release). REVISIONS.md item 1.2 incorrectly claimed sentence case here.
    return (name
            .replace("Ampulla of Vater", "Ampulla")
            .replace("Reduced Mucosal View", "ReducedMV")
            .replace("Lymphangiectasia", "Lymphang.")
            .replace("Normal clean mucosa", "Normal")
            .replace("Ileocecal valve", "Ileoc.valve")
            .replace("Foreign Body", "FB")
            .replace("Blood - fresh", "Blood-F")
            .replace("Blood - hematin", "Blood-H"))


def plot_one_cm(ax, d: Dict, title: str):
    classes = d["class_names"]
    labels = d["labels"]
    preds = d["preds"]
    idx = list(range(len(classes)))
    cm = confusion_matrix(labels, preds, labels=idx).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1.0))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(idx); ax.set_yticks(idx)
    ax.set_xticklabels([short(c) for c in classes], rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels([short(c) for c in classes], fontsize=7)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10)

    # annotate diagonal recall, and any off-diagonal cell > 0.10
    for i in idx:
        for j in idx:
            v = cm_norm[i, j]
            if i == j and row_sums[i, 0] > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
            elif v >= 0.10:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="#883")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", default=None)
    ap.add_argument("--pi", default=None)
    ap.add_argument("--distill", default=None)
    ap.add_argument("--out", default="fig4_confusion_matrices.png")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    conditions = [(k, load(v)) for k, v in
                  [("rgb", args.rgb), ("pi", args.pi), ("distill", args.distill)]
                  if v is not None]
    conditions = [(k, d) for k, d in conditions if d is not None]
    if not conditions:
        raise SystemExit("Pass at least one of --rgb / --pi / --distill")

    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.5), squeeze=False)
    last_im = None
    for i, (variant, d) in enumerate(conditions):
        last_im = plot_one_cm(axes[0, i], d, CONDITION_NAMES[variant])
    cbar = fig.colorbar(last_im, ax=axes[0, :].tolist(), location="right",
                         shrink=0.85, pad=0.02)
    cbar.set_label("Row-normalized frequency", fontsize=9)

    fig.suptitle("Figure 4 — Confusion matrices on held-out Kvasir-Capsule test split (row-normalized)",
                  fontsize=12, fontweight="bold", y=1.0)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    base, _ = os.path.splitext(args.out)
    plt.savefig(base + ".pdf", bbox_inches="tight")
    print(f"[fig4] saved {args.out} and {base}.pdf")


if __name__ == "__main__":
    main()
