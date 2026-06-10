"""Population-scale separation analysis of v1 vs v2 P_blood priors.

For every frame in {train, val, test}/{Blood - fresh, Normal clean mucosa},
compute the center-50% mean of v1 and v2 P_blood and report:
    (a) means, std, and Cohen's d separation between blood and mucosa
    (b) a one-sided AUC for "blood frame vs normal" using the per-frame
        prior mean as the only feature

This is the diagnostic that justifies the v2 redesign: a paper-worthy prior
should be able to discriminate blood frames from normal frames at the
*frame level* using just its area-mean, with no learning. v1 fails this
test on Kvasir-Capsule (its area-mean is roughly constant ≈0.5 by
construction); v2 should pass.

Output:
    D:/kvasir_capsule/outputs/prior_separation.json
    D:/kvasir_capsule/outputs/prior_separation.csv
    figures/fig_prior_separation.png + .pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def _hindex_v1(rgb01):
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return r / (g + b + EPS)


def _fluence(h, w):
    lam = 0.25 * float(np.sqrt(h * h + w * w))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return np.exp(-r / lam)


def _pblood_v1(rgb01, alpha=4.0, clip_pct=0.01):
    h_idx = _hindex_v1(rgb01)
    h_lo = np.percentile(h_idx, clip_pct * 100.0)
    h_hi = np.percentile(h_idx, (1.0 - clip_pct) * 100.0)
    h_clipped = np.clip(h_idx, h_lo, h_hi)
    h_norm = (h_clipped - h_lo) / (h_hi - h_lo + EPS)
    sig = 1.0 / (1.0 + np.exp(-alpha * (h_norm - 0.5)))
    return sig * _fluence(rgb01.shape[0], rgb01.shape[1])


def _pblood_v2(rgb01, alpha=6.0, pivot=0.30):
    r, g = rgb01[..., 0], rgb01[..., 1]
    h = (r - g) / (r + g + EPS)
    sig = 1.0 / (1.0 + np.exp(-alpha * (h - pivot)))
    return sig * _fluence(rgb01.shape[0], rgb01.shape[1])


def _center_mean(m, frac=0.5):
    H, W = m.shape
    ch, cw = int(H * frac), int(W * frac)
    y0 = (H - ch) // 2; x0 = (W - cw) // 2
    return float(m[y0:y0 + ch, x0:x0 + cw].mean())


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="D:/kvasir_capsule/stage2_data")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--blood_class", default="Blood - fresh")
    ap.add_argument("--mucosa_class", default="Normal clean mucosa")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max_per_class", type=int, default=600,
                    help="Cap to keep runtime reasonable (mucosa class has ~25k frames).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="D:/kvasir_capsule/outputs")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    fig_out = os.path.join(here, "fig_prior_separation.png")
    csv_out = os.path.join(args.out_dir, "prior_separation.csv")
    json_out = os.path.join(args.out_dir, "prior_separation.json")

    rows = []
    for split in args.splits:
        for cls in [args.blood_class, args.mucosa_class]:
            d = os.path.join(args.data_dir, split, cls)
            if not os.path.isdir(d):
                continue
            paths = sorted(sum([glob(os.path.join(d, e))
                                for e in ("*.png", "*.jpg", "*.jpeg")], []))
            if len(paths) > args.max_per_class:
                idx = rng.choice(len(paths), args.max_per_class, replace=False)
                paths = [paths[i] for i in sorted(idx)]
            print(f"[sep] {split}/{cls:<24} n={len(paths)}", flush=True)
            for i, p in enumerate(paths):
                try:
                    img = Image.open(p).convert("RGB").resize(
                        (args.size, args.size), Image.BILINEAR)
                except Exception:
                    continue
                rgb = np.asarray(img, dtype=np.float32) / 255.0
                v1m = _center_mean(_pblood_v1(rgb))
                v2m = _center_mean(_pblood_v2(rgb))
                rows.append({"split": split, "class": cls,
                             "frame": os.path.basename(p),
                             "v1_mean": v1m, "v2_mean": v2m})
                if (i + 1) % 200 == 0:
                    print(f"   ..{i+1}/{len(paths)}", flush=True)

    if not rows:
        raise SystemExit("[sep] no rows — check the data dir / class names")

    # Write CSV
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "class", "frame",
                                           "v1_mean", "v2_mean"])
        w.writeheader()
        w.writerows(rows)
    print(f"[sep] wrote {csv_out} ({len(rows)} rows)")

    # Compute summary
    def _arrs(cls):
        a = np.array([r["v1_mean"] for r in rows if r["class"] == cls])
        b = np.array([r["v2_mean"] for r in rows if r["class"] == cls])
        return a, b

    blood_v1, blood_v2 = _arrs(args.blood_class)
    muc_v1, muc_v2 = _arrs(args.mucosa_class)

    def _cohen_d(a, b):
        n1, n2 = len(a), len(b)
        s = float(np.sqrt(((n1 - 1) * a.var(ddof=1) +
                           (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)))
        if s == 0:
            return float("nan")
        return float((a.mean() - b.mean()) / s)

    y_true = np.concatenate([np.ones_like(blood_v1), np.zeros_like(muc_v1)])
    auc_v1 = float(roc_auc_score(y_true, np.concatenate([blood_v1, muc_v1])))
    auc_v2 = float(roc_auc_score(y_true, np.concatenate([blood_v2, muc_v2])))

    summary = {
        "n_blood": int(len(blood_v1)),
        "n_mucosa": int(len(muc_v1)),
        "blood_v1_mean": float(blood_v1.mean()), "blood_v1_std": float(blood_v1.std()),
        "blood_v2_mean": float(blood_v2.mean()), "blood_v2_std": float(blood_v2.std()),
        "mucosa_v1_mean": float(muc_v1.mean()), "mucosa_v1_std": float(muc_v1.std()),
        "mucosa_v2_mean": float(muc_v2.mean()), "mucosa_v2_std": float(muc_v2.std()),
        "cohen_d_v1": _cohen_d(blood_v1, muc_v1),
        "cohen_d_v2": _cohen_d(blood_v2, muc_v2),
        "auc_v1_priormean_only": auc_v1,
        "auc_v2_priormean_only": auc_v2,
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[sep] wrote {json_out}")
    print(f"\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:30s} {v}")
    print(f"\nFrame-level AUC using only the prior's center-50% mean:")
    print(f"   v1 ({args.blood_class} vs {args.mucosa_class})  AUC = {auc_v1:.3f}")
    print(f"   v2 ({args.blood_class} vs {args.mucosa_class})  AUC = {auc_v2:.3f}")
    print(f"\nIf v1 AUC is near 0.5, v1 is *not* discriminative on its own —")
    print(f"that is exactly why the +P_blood ablation arm hurt blood detection.")

    # Histogram figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    bins = 40
    axes[0].hist(muc_v1, bins=bins, alpha=0.55, color="#888888", label=f"Normal (n={len(muc_v1)})")
    axes[0].hist(blood_v1, bins=bins, alpha=0.65, color="#d62728", label=f"Blood-fresh (n={len(blood_v1)})")
    axes[0].set_xlabel("center-50% mean of $P_\\mathrm{blood}$ (v1)")
    axes[0].set_ylabel("frame count")
    axes[0].set_title(f"v1 prior (per-image quantile-norm)\n"
                      f"AUC frame-level = {auc_v1:.3f},  Cohen d = {summary['cohen_d_v1']:+.2f}",
                      fontsize=10, loc="left")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, linestyle=":")

    axes[1].hist(muc_v2, bins=bins, alpha=0.55, color="#888888", label=f"Normal (n={len(muc_v2)})")
    axes[1].hist(blood_v2, bins=bins, alpha=0.65, color="#2ca02c", label=f"Blood-fresh (n={len(blood_v2)})")
    axes[1].set_xlabel("center-50% mean of $P_\\mathrm{blood}$ (v2)")
    axes[1].set_title(f"v2 prior (NDVI$_{{red}}$, scale-fixed)\n"
                      f"AUC frame-level = {auc_v2:.3f},  Cohen d = {summary['cohen_d_v2']:+.2f}",
                      fontsize=10, loc="left")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=200, bbox_inches="tight")
    plt.savefig(os.path.splitext(fig_out)[0] + ".pdf", bbox_inches="tight")
    print(f"[sep] saved {fig_out}")


if __name__ == "__main__":
    main()
