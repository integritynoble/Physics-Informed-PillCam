"""Compare v1 (per-image-quantile) vs v2 (scale-fixed NDVI_red) P_blood priors.

Renders a 4xN grid where N is the number of example frames sampled from a
"Blood - fresh" folder and a "Normal clean mucosa" folder (so the per-image
mean of P_blood is interpretable: high in blood, low in mucosa).

Columns: RGB | P_blood (v1) | P_blood (v2) | mean P_blood per frame (bar)

The fourth column is a tiny annotation showing the area-mean of each prior
inside a central crop, which is the scalar a CNN's first-layer pool would
see. The whole point of v2 is that this scalar is *low* on normal mucosa and
*high* on blood, while v1 is roughly constant ≈0.5 by construction.

Outputs:
    figures/fig_prior_v1_v2.png + .pdf
"""
from __future__ import annotations

import argparse
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

EPS = 1e-6


# v1 (existing draft formulation) — copied here in numpy so the script
# does not depend on torch
def _hemoglobin_index_v1(rgb01):
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return r / (g + b + EPS)


def _fluence_map(h, w, lambda_eff=None):
    if lambda_eff is None:
        lambda_eff = 0.25 * float(np.sqrt(h * h + w * w))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return np.exp(-r / lambda_eff)


def _pblood_v1(rgb01, alpha=4.0, clip_pct=0.01):
    h_idx = _hemoglobin_index_v1(rgb01)
    h_lo = np.percentile(h_idx, clip_pct * 100.0)
    h_hi = np.percentile(h_idx, (1.0 - clip_pct) * 100.0)
    h_clipped = np.clip(h_idx, h_lo, h_hi)
    h_norm = (h_clipped - h_lo) / (h_hi - h_lo + EPS)
    sig = 1.0 / (1.0 + np.exp(-alpha * (h_norm - 0.5)))
    phi = _fluence_map(rgb01.shape[0], rgb01.shape[1])
    return sig * phi


def _pblood_v2(rgb01, alpha=6.0, pivot=0.30):
    r, g = rgb01[..., 0], rgb01[..., 1]
    h = (r - g) / (r + g + EPS)
    sig = 1.0 / (1.0 + np.exp(-alpha * (h - pivot)))
    phi = _fluence_map(rgb01.shape[0], rgb01.shape[1])
    return sig * phi


def _load(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--blood_dir",
                    default="D:/kvasir_capsule/stage2_data/train/Blood - fresh")
    ap.add_argument("--mucosa_dir",
                    default="D:/kvasir_capsule/stage2_data/train/Normal clean mucosa")
    ap.add_argument("--n_blood", type=int, default=3)
    ap.add_argument("--n_mucosa", type=int, default=3)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "fig_prior_v1_v2.png")

    def _sample(d, n):
        ps = sorted(sum([glob(os.path.join(d, e))
                         for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
        if not ps:
            raise SystemExit(f"no frames in {d}")
        # Stride sample to avoid sequential frames
        step = max(1, len(ps) // (n * 7))
        return ps[::step][:n]

    blood = _sample(args.blood_dir, args.n_blood)
    mucosa = _sample(args.mucosa_dir, args.n_mucosa)
    paths = [(p, "Blood-fresh") for p in blood] + [(p, "Normal mucosa") for p in mucosa]

    # central-crop area mean (50% box) — what the CNN's first-stage pool roughly sees
    def _center_mean(m, frac=0.5):
        H, W = m.shape
        ch, cw = int(H * frac), int(W * frac)
        y0 = (H - ch) // 2; x0 = (W - cw) // 2
        return float(m[y0:y0 + ch, x0:x0 + cw].mean())

    cmap = LinearSegmentedColormap.from_list("blood", ["#000000", "#ff1a1a", "#ffd400"])

    fig, axes = plt.subplots(len(paths), 4, figsize=(11.5, 2.5 * len(paths)),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0]})
    if len(paths) == 1:
        axes = np.array([axes])

    col_titles = [
        "RGB frame",
        r"v1 $P_\mathrm{blood}$ (per-image quantile-norm)",
        r"v2 $P_\mathrm{blood}$ (NDVI$_{\mathrm{red}}$, scale-fixed)",
        "Center-50% mean of $P_\\mathrm{blood}$",
    ]

    means_v1, means_v2, lbls = [], [], []
    for i, (p, lbl) in enumerate(paths):
        rgb = _load(p, args.size)
        pb1 = _pblood_v1(rgb)
        pb2 = _pblood_v2(rgb)
        m1 = _center_mean(pb1); m2 = _center_mean(pb2)
        means_v1.append(m1); means_v2.append(m2); lbls.append(lbl)

        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(pb1, cmap=cmap, vmin=0, vmax=1)
        axes[i, 2].imshow(pb2, cmap=cmap, vmin=0, vmax=1)
        # Bar plot of center-mean
        axes[i, 3].bar(["v1", "v2"], [m1, m2],
                       color=["#888888", "#2ca02c"],
                       edgecolor="black", linewidth=0.5)
        axes[i, 3].set_ylim(0, 1.0)
        for j, v in enumerate([m1, m2]):
            axes[i, 3].text(j, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        axes[i, 3].set_ylabel("mean $P_\\mathrm{blood}$")
        axes[i, 3].axhline(0.5, ls=":", lw=0.5, color="#555")

        for j in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(f"{lbl}\n{os.path.basename(p)}",
                              rotation=0, ha="right", va="center", fontsize=8)
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=10)

    # Annotation — separation between blood and mucosa
    blood_v1 = [means_v1[i] for i, l in enumerate(lbls) if l == "Blood-fresh"]
    blood_v2 = [means_v2[i] for i, l in enumerate(lbls) if l == "Blood-fresh"]
    muc_v1   = [means_v1[i] for i, l in enumerate(lbls) if l == "Normal mucosa"]
    muc_v2   = [means_v2[i] for i, l in enumerate(lbls) if l == "Normal mucosa"]
    sep_v1 = (np.mean(blood_v1) if blood_v1 else 0) - (np.mean(muc_v1) if muc_v1 else 0)
    sep_v2 = (np.mean(blood_v2) if blood_v2 else 0) - (np.mean(muc_v2) if muc_v2 else 0)
    fig.suptitle(
        f"P_blood prior comparison — center-50% mean separation "
        f"(blood − mucosa):  v1 = {sep_v1:+.2f},  v2 = {sep_v2:+.2f}",
        fontsize=11, y=1.005)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")
    print(f"[fig_prior_v1_v2] saved {out} and {os.path.splitext(out)[0]}.pdf")
    print(f"[fig_prior_v1_v2] center-50% mean P_blood:")
    for (p, lbl), m1, m2 in zip(paths, means_v1, means_v2):
        print(f"  {lbl:<14} {os.path.basename(p):<30}  v1={m1:.3f}  v2={m2:.3f}")
    print(f"\n[fig_prior_v1_v2] separation (blood − mucosa):  v1={sep_v1:+.3f}  v2={sep_v2:+.3f}")


if __name__ == "__main__":
    main()
