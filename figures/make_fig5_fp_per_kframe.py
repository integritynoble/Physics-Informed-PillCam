"""Figure 5: False positives per 1000 frames vs per-lesion sensitivity.

The temporal-consistency story. For each test video (sequenced by video_id
inferred from the path), sweep a persistence threshold k in {1..k_max}: a
positive is "confirmed" only if k consecutive frames predict the same
abnormal class. Plot FP/1000-frames (x) vs sensitivity (y) for each k, one
line per condition.

Inputs are the same test_predictions.json files used by F3/F4. We treat any
non-Normal class as "abnormal" for the FP/sensitivity computation.

Usage:
    python make_fig5_fp_per_kframe.py \
      --rgb     D:/kvasir_capsule/outputs/stage2_rgb_effb0/test_predictions.json \
      --pi      D:/kvasir_capsule/outputs/stage2_pi_effb0/test_predictions.json \
      --distill D:/kvasir_capsule/outputs/stage2_distill_effb0/test_predictions.json \
      --out     fig5_fp_per_kframe.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


CONDITION_STYLES = {
    "rgb":     {"label": "RGB-only baseline",   "color": "#0d4a8c", "marker": "o"},
    "pi":      {"label": "+MC prior (5-ch)",     "color": "#a02335", "marker": "s"},
    "distill": {"label": "+Distillation (3-ch)", "color": "#5a2876", "marker": "^"},
}

NORMAL_CLASS = "Normal clean mucosa"


def load(path: Optional[str]) -> Optional[Dict]:
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def video_id_from_path(p: str) -> str:
    """Kvasir-Capsule frame names look like '{video_id}_{frame_n}.{ext}'."""
    base = os.path.basename(p)
    m = re.match(r"^([0-9a-f]+)_\d+\.(?:jpg|png|jpeg|bmp)$", base, re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: take everything before the last underscore
    stem = os.path.splitext(base)[0]
    return stem.rsplit("_", 1)[0]


def frame_index(p: str) -> int:
    base = os.path.splitext(os.path.basename(p))[0]
    m = re.search(r"_(\d+)$", base)
    return int(m.group(1)) if m else 0


def temporal_confirm(seq_pred_abnormal: np.ndarray, k: int) -> np.ndarray:
    """Mark a frame as 'confirmed abnormal' iff k consecutive frames are abnormal,
    centered on the frame (rolling window). Returns a bool array same length."""
    if k <= 1:
        return seq_pred_abnormal.astype(bool)
    n = len(seq_pred_abnormal)
    out = np.zeros(n, dtype=bool)
    pad = k // 2
    cs = np.concatenate([[0], np.cumsum(seq_pred_abnormal.astype(int))])
    for i in range(n):
        lo, hi = max(0, i - pad), min(n, i - pad + k)
        if cs[hi] - cs[lo] >= k:
            out[i] = True
    return out


def fp_sens_for_k(d: Dict, k: int) -> Tuple[float, float]:
    classes = d["class_names"]
    normal_idx = classes.index(NORMAL_CLASS) if NORMAL_CLASS in classes else -1
    paths = d["paths"]
    labels = np.asarray(d["labels"])
    preds = np.asarray(d["preds"])

    is_pred_abnormal = preds != normal_idx
    is_true_abnormal = labels != normal_idx

    # group by video_id, sort by frame index
    by_video: Dict[str, List[int]] = {}
    for i, p in enumerate(paths):
        vid = video_id_from_path(p)
        by_video.setdefault(vid, []).append(i)

    confirmed_abnormal = np.zeros_like(is_pred_abnormal)
    for vid, idxs in by_video.items():
        idxs_sorted = sorted(idxs, key=lambda i: frame_index(paths[i]))
        local = is_pred_abnormal[idxs_sorted]
        conf = temporal_confirm(local, k)
        for j, i in enumerate(idxs_sorted):
            confirmed_abnormal[i] = conf[j]

    tp = int(((confirmed_abnormal == 1) & (is_true_abnormal == 1)).sum())
    fn = int(((confirmed_abnormal == 0) & (is_true_abnormal == 1)).sum())
    fp = int(((confirmed_abnormal == 1) & (is_true_abnormal == 0)).sum())
    n_frames = len(labels)
    n_normal = int((is_true_abnormal == 0).sum())

    sensitivity = tp / max(tp + fn, 1)
    fp_per_1000 = (fp / max(n_normal, 1)) * 1000.0  # FP per 1000 normal frames
    return fp_per_1000, sensitivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", default=None)
    ap.add_argument("--pi", default=None)
    ap.add_argument("--distill", default=None)
    ap.add_argument("--out", default="fig5_fp_per_kframe.png")
    ap.add_argument("--k_max", type=int, default=9)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    conditions = {
        "rgb":     load(args.rgb),
        "pi":      load(args.pi),
        "distill": load(args.distill),
    }
    if not any(conditions.values()):
        raise SystemExit("Pass at least one of --rgb / --pi / --distill")

    ks = list(range(1, args.k_max + 1, 2))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for variant, d in conditions.items():
        if d is None:
            continue
        xs, ys = [], []
        for k in ks:
            x, y = fp_sens_for_k(d, k)
            xs.append(x); ys.append(y)
        s = CONDITION_STYLES[variant]
        ax.plot(xs, ys, color=s["color"], marker=s["marker"], markersize=8,
                linewidth=1.5, label=s["label"])
        for k, x, y in zip(ks, xs, ys):
            ax.annotate(f"k={k}", (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=7, color=s["color"])

    ax.set_xlabel("False positives per 1000 normal frames", fontsize=10)
    ax.set_ylabel("Per-frame sensitivity (any abnormal vs Normal)", fontsize=10)
    ax.set_title("Figure 5 — Temporal-consistency tradeoff on held-out Kvasir-Capsule videos",
                  fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    base, _ = os.path.splitext(args.out)
    plt.savefig(base + ".pdf", bbox_inches="tight")
    print(f"[fig5] saved {args.out} and {base}.pdf")


if __name__ == "__main__":
    main()
