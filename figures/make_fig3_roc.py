"""Figure 3: ROC overlay across the 3 conditions (RGB, +PI, +Distill).

Produces a 2x2 grid:
    [Macro-average ROC]   [Polyp ROC]
    [Blood-fresh ROC]     [Ulcer ROC]
plus a sidebar listing per-class AUC for all three conditions.

Reads the JSON files produced by eval_test_predictions.py.

Usage:
    python make_fig3_roc.py \
      --rgb     D:/kvasir_capsule/outputs/stage2_rgb_effb0/test_predictions.json \
      --pi      D:/kvasir_capsule/outputs/stage2_pi_effb0/test_predictions.json \
      --distill D:/kvasir_capsule/outputs/stage2_distill_effb0/test_predictions.json \
      --out     fig3_roc.png

Either --pi or --distill can be omitted; the script will draw what it has.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


CONDITION_STYLES = {
    "rgb":     {"label": "RGB-only baseline",   "color": "#0d4a8c", "ls": "-"},
    "pi":      {"label": "+MC prior (5-ch)",     "color": "#a02335", "ls": "-"},
    "distill": {"label": "+Distillation (3-ch)", "color": "#5a2876", "ls": "--"},
}

# Polyp, Ampulla of papilla, and Blood-hematin are single-video classes excluded
# from the headline test split (see setup_kvasir_capsule.split_videos and
# paper_outline.md §3.4). FOCUS_CLASSES picks three evaluable, clinically
# interesting classes for the ROC subplots.
FOCUS_CLASSES = ["Blood - fresh", "Angiectasia", "Ulcer"]


def load(path: Optional[str]) -> Optional[Dict]:
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def per_class_auc(d: Dict) -> Dict[str, float]:
    y_true = np.asarray(d["labels"])
    probs = np.asarray(d["probs"])
    classes = d["class_names"]
    aucs = {}
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            aucs[cls] = float("nan")
            continue
        try:
            aucs[cls] = roc_auc_score(y_bin, probs[:, i])
        except ValueError:
            aucs[cls] = float("nan")
    return aucs


def macro_auc(d: Dict) -> float:
    aucs = per_class_auc(d)
    vals = [v for v in aucs.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def plot_class_roc(ax, conditions: Dict[str, Dict], cls_name: str,
                    title: Optional[str] = None):
    """Overlay one ROC curve per condition for a single class."""
    plotted = False
    for variant, d in conditions.items():
        if d is None:
            continue
        if cls_name not in d["class_names"]:
            continue
        i = d["class_names"].index(cls_name)
        y_true = np.asarray(d["labels"])
        probs = np.asarray(d["probs"])
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
        auc = roc_auc_score(y_bin, probs[:, i])
        s = CONDITION_STYLES[variant]
        ax.plot(fpr, tpr, color=s["color"], linestyle=s["ls"], linewidth=1.6,
                label=f"{s['label']}  AUC={auc:.3f}")
        plotted = True
    ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=0.8)
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate", fontsize=9)
    ax.set_title(title or cls_name, fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=8)
    if plotted:
        ax.legend(loc="lower right", fontsize=7.5)
    else:
        ax.text(0.5, 0.5, f"{cls_name}: not in any test split",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#888")


def plot_macro_roc(ax, conditions: Dict[str, Dict]):
    """Macro-average ROC: mean per-class TPR over a fixed FPR grid."""
    fpr_grid = np.linspace(0, 1, 200)
    for variant, d in conditions.items():
        if d is None:
            continue
        y_true = np.asarray(d["labels"])
        probs = np.asarray(d["probs"])
        classes = d["class_names"]
        tprs = []
        for i, cls in enumerate(classes):
            y_bin = (y_true == i).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
            tprs.append(np.interp(fpr_grid, fpr, tpr))
        if not tprs:
            continue
        mean_tpr = np.mean(tprs, axis=0)
        macro = macro_auc(d)
        s = CONDITION_STYLES[variant]
        ax.plot(fpr_grid, mean_tpr, color=s["color"], linestyle=s["ls"], linewidth=1.8,
                label=f"{s['label']}  macro AUC={macro:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=0.8)
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("Mean true positive rate (macro)", fontsize=9)
    ax.set_title("Macro-average ROC across all evaluable classes", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower right", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", default=None)
    ap.add_argument("--pi", default=None)
    ap.add_argument("--distill", default=None)
    ap.add_argument("--out", default="fig3_roc.png")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    conditions = {
        "rgb":     load(args.rgb),
        "pi":      load(args.pi),
        "distill": load(args.distill),
    }
    if not any(conditions.values()):
        raise SystemExit("Pass at least one of --rgb / --pi / --distill")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_macro_roc(axes[0, 0], conditions)
    for ax, cls in zip([axes[0, 1], axes[1, 0], axes[1, 1]], FOCUS_CLASSES):
        plot_class_roc(ax, conditions, cls)

    fig.suptitle("Figure 3 — ROC curves on held-out Kvasir-Capsule test split",
                  fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    base, _ = os.path.splitext(args.out)
    plt.savefig(base + ".pdf", bbox_inches="tight")
    print(f"[fig3] saved {args.out} and {base}.pdf")

    # Also dump a CSV of per-class AUCs for Table 2
    rows = []
    classes = next((d["class_names"] for d in conditions.values() if d is not None), [])
    for cls in classes:
        row = {"class": cls}
        for variant, d in conditions.items():
            if d is None:
                row[variant] = ""
            else:
                aucs = per_class_auc(d)
                row[variant] = f"{aucs.get(cls, float('nan')):.3f}"
        rows.append(row)
    csv_path = os.path.splitext(args.out)[0] + "_aucs.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class,rgb,pi,distill\n")
        for r in rows:
            f.write(f"{r['class']},{r['rgb']},{r['pi']},{r['distill']}\n")
    print(f"[fig3] per-class AUCs -> {csv_path}")


if __name__ == "__main__":
    main()
