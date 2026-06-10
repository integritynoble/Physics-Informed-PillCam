"""Figure: ablation summary across the six paper arms.

Two panels in one figure:
    (a) Macro-AUC + accuracy + weighted-F1 per arm (grouped bars with 95% CIs
        on macro-AUC from per_class_auc.json, errorless on accuracy / weighted-F1
        which are point estimates from test_metrics.json).
    (b) Per-class AUC heatmap for the 11 evaluable classes × 6 arms,
        emphasizing the abnormal-finding rows (bleeding / Angiectasia /
        Erosion / Ulcer / Erythema / Lymphangiectasia).

Inputs:
    D:/kvasir_capsule/outputs/<arm>/test_metrics.json
    D:/kvasir_capsule/outputs/per_class_auc.json   (produced by per_class_auc_table.py)

Outputs:
    figures/fig_ablation.png + .pdf
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ARMS = [
    ("RGB-only",                "stage2_rgb_effb0",            "#888888"),
    ("+P_blood only",           "stage2_pi_blood_only_effb0",  "#d62728"),
    ("+H_AFI only",             "stage2_pi_hafi_only_effb0",   "#1f77b4"),
    ("Physics-only\n(no RGB)",  "stage2_pi_physics_only_effb0","#9467bd"),
    ("Full PI",                 "stage2_pi_effb0",             "#2ca02c"),
    ("Distill",                 "stage2_distill_effb0",        "#ff7f0e"),
]

HEADLINE_CLASSES = ["Blood - fresh", "Angiectasia", "Erosion", "Ulcer",
                    "Erythema", "Lymphangiectasia"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="D:/kvasir_capsule/outputs")
    p.add_argument("--out", default=None,
                   help="Output PNG (the matching .pdf is written next to it). "
                        "Default: figures/fig_ablation.png next to this script.")
    p.add_argument("--pca_path", default=None,
                   help="Path to per_class_auc.json (defaults to <out_dir>/per_class_auc.json). "
                        "Use this to render an extended figure that spans v1+v2 arms.")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "fig_ablation.png")

    # ---- load per-arm headline metrics ----
    rows = []
    for label, dirname, color in ARMS:
        path = os.path.join(args.out_dir, dirname, "test_metrics.json")
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "label": label, "color": color, "dirname": dirname,
            "accuracy": m["accuracy"], "weighted_f1": m["weighted_f1"],
            "macro_f1_evaluable": m["macro_f1_evaluable"],
        })

    # ---- load per-class AUC table ----
    pca_path = args.pca_path or os.path.join(args.out_dir, "per_class_auc.json")
    with open(pca_path, "r", encoding="utf-8") as f:
        pca = json.load(f)
    arm_labels = pca["arms"]
    classes = pca["classes"]
    macros = pca["macro_auc_evaluable"]

    # ---- figure layout ----
    fig = plt.figure(figsize=(13.5, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4], hspace=0.45)

    # Panel (a): grouped bars of accuracy / weighted-F1 / macro-AUC
    ax = fig.add_subplot(gs[0])
    metrics = [
        ("Accuracy",       [r["accuracy"] for r in rows]),
        ("Weighted-F1",    [r["weighted_f1"] for r in rows]),
        ("Macro-AUC (eval, 11 cls)",
                            [macros.get(_align(rows[i]["label"], arm_labels), float("nan"))
                             for i in range(len(rows))]),
    ]
    n_arms = len(rows)
    bw = 0.26
    x = np.arange(n_arms)
    for j, (m_label, m_vals) in enumerate(metrics):
        ax.bar(x + (j - 1) * bw, m_vals, bw, label=m_label,
               edgecolor="black", linewidth=0.4,
               alpha=0.85)
        for k, v in enumerate(m_vals):
            if np.isfinite(v):
                ax.text(x[k] + (j - 1) * bw, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(rows[0]["accuracy"], ls="--", lw=0.5, color="#888",
               label="RGB-only accuracy reference")
    ax.set_title("(a) Test-set headline metrics across ablation arms (n=6,423 frames)",
                 loc="left", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=9, frameon=True)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    # Panel (b): per-class AUC heatmap for evaluable classes
    ax2 = fig.add_subplot(gs[1])
    evaluable = [c for c in classes
                 if pca["per_class"][c].get(arm_labels[0], {}).get("auc") is not None]
    # Ensure deterministic order: headline classes first (in their listed order),
    # then the rest alphabetical
    head = [c for c in HEADLINE_CLASSES if c in evaluable]
    rest = sorted([c for c in evaluable if c not in head])
    cls_order = head + rest
    M = np.full((len(cls_order), len(arm_labels)), np.nan)
    for i, cls in enumerate(cls_order):
        for j, lbl in enumerate(arm_labels):
            cell = pca["per_class"][cls].get(lbl, {})
            v = cell.get("auc")
            if v is not None:
                M[i, j] = v
    im = ax2.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.01)
    cbar.set_label("AUC (one-vs-rest)", rotation=90, fontsize=9)
    ax2.set_xticks(range(len(arm_labels)))
    ax2.set_xticklabels(arm_labels, rotation=0, fontsize=8.5)
    ax2.set_yticks(range(len(cls_order)))
    yticklabels = [(c + " ★" if c in HEADLINE_CLASSES else c) for c in cls_order]
    ax2.set_yticklabels(yticklabels, fontsize=8.5)
    # Boldface row labels for headline classes
    for i, cls in enumerate(cls_order):
        if cls in HEADLINE_CLASSES:
            ax2.get_yticklabels()[i].set_fontweight("bold")
    # Annotate each cell
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                txt_color = "white" if (v < 0.5 or v > 0.85) else "black"
                ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                          fontsize=7, color=txt_color)
    ax2.set_title("(b) Per-class one-vs-rest AUC (★ = headline abnormal-finding classes)",
                  loc="left", fontsize=11)

    plt.tight_layout()
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    print(f"[fig_ablation] saved {out} and {os.path.splitext(out)[0]}.pdf")


def _align(panel_label: str, arm_labels):
    """Map the panel-A short label to the arm_labels used in per_class_auc.json."""
    panel_label_norm = panel_label.replace("\n", " ")
    for lbl in arm_labels:
        if panel_label_norm == lbl:
            return lbl
        if panel_label_norm in lbl or lbl in panel_label_norm:
            return lbl
    # Fallback: try heuristic substrings
    if "Full" in panel_label_norm: return next(l for l in arm_labels if "Full" in l)
    if "RGB-only" in panel_label_norm: return "RGB-only"
    if "P_blood" in panel_label_norm: return next(l for l in arm_labels if "P_blood" in l)
    if "H_AFI" in panel_label_norm: return next(l for l in arm_labels if "H_AFI" in l)
    if "Physics" in panel_label_norm: return next(l for l in arm_labels if "Physics" in l)
    if "Distill" in panel_label_norm: return next(l for l in arm_labels if "Distill" in l)
    raise KeyError(panel_label)


if __name__ == "__main__":
    main()
