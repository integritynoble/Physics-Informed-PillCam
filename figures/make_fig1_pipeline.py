"""Figure 1: Pipeline diagram for the physics-informed WCE classifier.

Three parallel branches share an ImageNet-pretrained backbone:
    A. RGB-only baseline           (3-channel input)
    B. +MC physics prior           (5-channel input = RGB + P_blood + H_AFI*Phi)
    C. +Distillation auxiliary     (3-channel input + auxiliary P_blood decoder)

Pure matplotlib, no torch.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


COLOR_INPUT = "#e9eef7"
COLOR_PRIOR = "#fde7e9"
COLOR_BACKBONE = "#e7f3ec"
COLOR_HEAD = "#fff3d6"
COLOR_AUX = "#f1e3f5"
EDGE = "#3b3b3b"
TXT = "#1a1a1a"


def box(ax, x, y, w, h, label, fc, sub=None, fs=16, sub_fs=13):
    # Default font sizes bumped 2026-05-04 (was fs=10, sub_fs=8) per
    # Dr. Zaman's review: at the 12.5-inch source figsize fitted into
    # a 6.5-inch text column, source fonts are scaled ~0.52x in print,
    # so old defaults rendered at ~5 pt and crowded adjacent boxes.
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.0, edgecolor=EDGE, facecolor=fc)
    ax.add_patch(p)
    if sub is None:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=TXT)
    else:
        ax.text(x + w / 2, y + h * 0.66, label, ha="center", va="center",
                fontsize=fs, color=TXT)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sub_fs, color="#404040", style="italic")


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                        linewidth=lw, color=color)
    ax.add_patch(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "fig1_pipeline.png"))
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    # 2026-05-04 redesign per Dr. Zaman's "too crowded / too many overlaps"
    # feedback. Stripped: the inline figure title/subtitle (info already in
    # the LaTeX caption below the figure), the bottom legend (colors are
    # explained by the box labels themselves), and the redundant "branch C
    # only" italic annotation. Boxes widened (2.6 -> 3.2 inputs, 2.5 -> 2.9
    # heads), and figsize raised to (12, 7) so the now-fewer elements have
    # genuine breathing room rather than fighting for the same pixels.
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0.5, 10)
    ax.set_aspect("auto")
    ax.axis("off")

    # === input column (left) ===
    box(ax, 0.3, 5.0, 2.0, 1.5, "Capsule\nframe", COLOR_INPUT,
        sub="RGB, 224x224", fs=17, sub_fs=13)

    # === three branches (vertically stacked) ===
    # Branch labels are centered ABOVE each branch's top box (not in the
    # arrow corridor between the capsule frame and the boxes), so the
    # "RGB-only / +MC prior / +Distill" text never collides with the
    # capsule->box arrows. The "5 ch" annotation that previously sat next
    # to a bracket is folded into branch B's label.

    # Branch labels are deliberately SHORT (one or two words past the
    # letter prefix) so the centered label at x=5.3 stays inside the
    # box-width band [3.7, 6.9] and never extends back into the arrow
    # corridor at x=2.3-3.7. Channel-count details are in the box
    # sub-labels and the LaTeX caption, so we don't repeat them here.

    # Branch A: RGB-only (top)
    yA = 7.8
    ax.text(5.3, yA + 1.25, "A:  RGB-only",
            ha="center", fontsize=17, fontweight="bold", color="#0d4a8c")
    box(ax, 3.7, yA, 3.2, 1.0, "RGB normalize", COLOR_INPUT,
        sub="3 channels", fs=16, sub_fs=13)

    # Branch B: +MC prior (middle) — two stacked input boxes
    yB = 4.6
    ax.text(5.3, yB + 2.45, "B:  +MC prior",
            ha="center", fontsize=17, fontweight="bold", color="#a02335")
    box(ax, 3.7, yB + 1.2, 3.2, 1.0, "RGB normalize", COLOR_INPUT,
        sub="3 channels", fs=16, sub_fs=13)
    box(ax, 3.7, yB - 0.05, 3.2, 1.0, "Physics prior", COLOR_PRIOR,
        sub="P_blood + H_AFI*Phi  (2 ch)", fs=16, sub_fs=13)

    # Branch C: +Distill (bottom)
    yC = 1.5
    ax.text(5.3, yC + 1.25, "C:  +Distill",
            ha="center", fontsize=17, fontweight="bold", color="#5a2876")
    box(ax, 3.7, yC, 3.2, 1.0, "RGB normalize", COLOR_INPUT,
        sub="3 channels", fs=16, sub_fs=13)

    # arrows from capsule -> each branch
    arrow(ax, 2.3, 5.7, 3.7, yA + 0.5)
    arrow(ax, 2.3, 5.7, 3.7, yB + 1.1)
    arrow(ax, 2.3, 5.7, 3.7, yC + 0.5)

    # === shared backbone ===
    bx, by, bw, bh = 8.4, 4.4, 2.8, 2.0
    box(ax, bx, by, bw, bh, "Pretrained\nbackbone", COLOR_BACKBONE,
        sub="EfficientNet-B0", fs=18, sub_fs=14)

    # branch -> backbone arrows (all start at right edge of input box(es), x=6.9)
    arrow(ax, 6.9, yA + 0.5, bx, by + 1.5)              # A
    arrow(ax, 6.9, yB + 1.07, bx, by + 1.0)             # B (centered between B's two stacked boxes)
    arrow(ax, 6.9, yC + 0.5, bx, by + 0.5)              # C

    # === heads (right column) ===
    # class head (always)
    box(ax, 12.1, 5.6, 2.7, 1.2, "Classifier head", COLOR_HEAD,
        sub="14-class softmax", fs=16, sub_fs=13)
    arrow(ax, bx + bw, 5.4, 12.1, 6.2)

    # auxiliary distillation head (training only — branch C)
    box(ax, 12.1, 2.5, 2.7, 1.4, "P_blood decoder", COLOR_AUX,
        sub="training only,\nbranch C", fs=16, sub_fs=13)
    # dashed arrow to indicate training-time only
    a_aux = FancyArrowPatch((bx + bw, 4.8), (12.1, 3.2), arrowstyle="-|>",
                            mutation_scale=12, linewidth=1.0,
                            color="#5a2876", linestyle=(0, (4, 2)))
    ax.add_patch(a_aux)

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    base, _ = os.path.splitext(args.out)
    plt.savefig(base + ".pdf", bbox_inches="tight")
    print(f"[fig1] saved {args.out} and {base}.pdf")


if __name__ == "__main__":
    main()
