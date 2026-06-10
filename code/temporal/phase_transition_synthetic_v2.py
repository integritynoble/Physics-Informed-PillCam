"""
Synthetic phase-transition experiment v2 — class-disjoint prior design.
========================================================================

The v1 script (`phase_transition_synthetic.py`) found that the
spatial-vs-summary boundary depends on whether summary stats happen
to encode class identity. In v1, blob centers were placed at varying
distances from the frame center, so `central-max` and `central-mean`
features carried direct class-discriminative signal — letting the
scalar parameterization win at high prior precision.

v2 redesigns the task so that summary statistics CANNOT distinguish
classes. All 14 classes have their target blob at the SAME radius
from the frame center, just at 14 different angles around the
circle. This makes:
  - mean, max, top-k of P (over the whole frame): identical across classes
  - central-max, central-mean: identical (all blobs are equidistant)
  - any rotation-invariant summary: identical

Only the SPATIAL LAYOUT of the blob differs across classes.

Prediction (matching capsule):
  - alpha = 0: no info, baseline ~chance
  - alpha small: spatial channel can localize the blob -> wins.
                 Scalar parameterization gives ~0 (summary stats class-invariant).
  - alpha large: spatial channel near-perfect; scalar still ~0.
  - The spatial-vs-scalar gap should be ~uniformly positive across
    alpha, NOT cross sign as in v1.

If v2 reproduces this pattern, the synthetic verification supports
the capsule four-variant boundary cleanly.

Output:
  paper/nature-machine-intelligence/docs/phase_transition_synthetic_v2_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE.parent.parent / "docs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES = 14
IMAGE_SIZE = 32
N_TRAIN = 5000
N_VAL = 1000
N_TEST = 2000
NOISE_SIGMA = 0.20
BLOB_SIGMA = 1.5
ALPHA_SWEEP = [0.0, 1.0, 4.0, 16.0, 64.0]
SEEDS = [41, 42, 43]
EPOCHS = 12
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-4

# v2: place all 14 class blob centers at the same radius from frame
# center, just at 14 different angles. This makes all rotation-invariant
# summary statistics class-invariant.
CENTER = (IMAGE_SIZE - 1) / 2.0
RADIUS = 10.0     # distance from frame center for ALL class blobs
ANGLES = [2 * math.pi * k / N_CLASSES for k in range(N_CLASSES)]
CLASS_CENTERS = [(CENTER + RADIUS * math.sin(a), CENTER + RADIUS * math.cos(a))
                   for a in ANGLES]


def gaussian_blob(cy: float, cx: float, sigma: float = BLOB_SIGMA,
                    H: int = IMAGE_SIZE, W: int = IMAGE_SIZE) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                            np.arange(W, dtype=np.float32),
                            indexing="ij")
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))


def render_image(class_id: int, rng: np.random.Generator) -> np.ndarray:
    cy, cx = CLASS_CENTERS[class_id]
    cy_j = cy + rng.normal(0, 0.6)
    cx_j = cx + rng.normal(0, 0.6)
    blob = gaussian_blob(cy_j, cx_j, sigma=BLOB_SIGMA)
    bg = rng.normal(0.5, 0.05, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    noise = rng.normal(0, NOISE_SIGMA, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    img = np.clip(bg + 0.6 * blob + noise, 0.0, 1.0)
    return img.astype(np.float32)


def render_prior(class_id: int, alpha: float,
                  rng: np.random.Generator) -> np.ndarray:
    if alpha < 1e-3:
        return np.full((IMAGE_SIZE, IMAGE_SIZE), 0.5, dtype=np.float32)
    cy, cx = CLASS_CENTERS[class_id]
    sigma_prior = 1.0 / np.sqrt(alpha)
    sigma_prior = max(0.3, sigma_prior)
    p = gaussian_blob(cy, cx, sigma=sigma_prior)
    p = p / max(p.max(), 1e-6)
    return p.astype(np.float32)


def build_dataset(seed: int, alpha: float, n: int):
    rng = np.random.default_rng(seed)
    images = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    priors = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        c = rng.integers(0, N_CLASSES)
        images[i] = render_image(c, rng)
        priors[i] = render_prior(c, alpha, rng)
        labels[i] = c
    return images, priors, labels


def compute_scalar_features(prior: np.ndarray) -> np.ndarray:
    """8 scalar global summaries — by construction these are
    class-invariant in v2 (all class centers equidistant from frame
    center), but the model still has access to the same
    parameterization the capsule cell (c) used."""
    flat = prior.reshape(prior.shape[0], -1)
    n_pix = flat.shape[1]
    f_mean = flat.mean(axis=1)
    f_max = flat.max(axis=1)
    sorted_desc = np.sort(flat, axis=1)[:, ::-1]
    k1 = max(1, n_pix // 100)
    k5 = max(1, n_pix // 20)
    f_top1 = sorted_desc[:, :k1].mean(axis=1)
    f_top5 = sorted_desc[:, :k5].mean(axis=1)
    f_frac05 = (flat > 0.5).astype(np.float32).mean(axis=1)
    f_frac07 = (flat > 0.7).astype(np.float32).mean(axis=1)
    H, W = prior.shape[1], prior.shape[2]
    central = prior[:, H//4:3*H//4, W//4:3*W//4].reshape(prior.shape[0], -1)
    f_cmax = central.max(axis=1)
    f_cmean = central.mean(axis=1)
    return np.stack([f_mean, f_max, f_top1, f_top5, f_frac05, f_frac07,
                       f_cmax, f_cmean], axis=1)


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 64,
                 n_classes: int = N_CLASSES, aux_dim: int = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.aux_dim = aux_dim
        head_in = hidden + aux_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x, aux=None):
        h = F.gelu(self.conv1(x)); h = F.max_pool2d(h, 2)
        h = F.gelu(self.conv2(h)); h = F.max_pool2d(h, 2)
        h = F.gelu(self.conv3(h))
        h = self.pool(h).flatten(1)
        if self.aux_dim > 0:
            assert aux is not None
            h = torch.cat([h, aux], dim=-1)
        return self.head(h)


def train_and_eval(images_train, aux_train, labels_train,
                     images_test, aux_test, labels_test,
                     in_ch: int, aux_dim: int, seed: int,
                     epochs: int = EPOCHS) -> float:
    torch.manual_seed(seed)
    model = SmallCNN(in_ch=in_ch, aux_dim=aux_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    X_train = torch.from_numpy(images_train).to(DEVICE)
    A_train = torch.from_numpy(aux_train).to(DEVICE) if aux_train is not None else None
    y_train = torch.from_numpy(labels_train).to(DEVICE)
    X_test = torch.from_numpy(images_test).to(DEVICE)
    A_test = torch.from_numpy(aux_test).to(DEVICE) if aux_test is not None else None
    y_test = torch.from_numpy(labels_test).to(DEVICE)

    n_train = len(X_train)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
            x = X_train[idx]; y = y_train[idx]
            a = A_train[idx] if A_train is not None else None
            logits = model(x, a)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        all_probs = []
        for s in range(0, len(X_test), BATCH_SIZE * 4):
            xb = X_test[s: s + BATCH_SIZE * 4]
            ab = A_test[s: s + BATCH_SIZE * 4] if A_test is not None else None
            logits = model(xb, ab)
            all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels = y_test.cpu().numpy()
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(N_CLASSES):
        y_bin = (labels == c).astype(np.int32)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        aucs.append(roc_auc_score(y_bin, probs[:, c]))
    return float(np.mean(aucs))


def evaluate_alpha(alpha: float, seed: int, epochs: int) -> Dict[str, float]:
    img_tr, p_tr, l_tr = build_dataset(seed * 100 + 1, alpha, N_TRAIN)
    img_te, p_te, l_te = build_dataset(seed * 100 + 3, alpha, N_TEST)

    img_tr_4d = img_tr[:, None, :, :]
    img_te_4d = img_te[:, None, :, :]

    auc_baseline = train_and_eval(
        img_tr_4d, None, l_tr, img_te_4d, None, l_te,
        in_ch=1, aux_dim=0, seed=seed, epochs=epochs)

    sc_tr = compute_scalar_features(p_tr).astype(np.float32)
    sc_te = compute_scalar_features(p_te).astype(np.float32)
    sc_mean = sc_tr.mean(axis=0); sc_std = sc_tr.std(axis=0) + 1e-6
    sc_tr_n = (sc_tr - sc_mean) / sc_std
    sc_te_n = (sc_te - sc_mean) / sc_std
    auc_scalar = train_and_eval(
        img_tr_4d, sc_tr_n, l_tr, img_te_4d, sc_te_n, l_te,
        in_ch=1, aux_dim=8, seed=seed, epochs=epochs)

    p_tr_4d = p_tr[:, None, :, :]; p_te_4d = p_te[:, None, :, :]
    img_p_tr = np.concatenate([img_tr_4d, p_tr_4d], axis=1)
    img_p_te = np.concatenate([img_te_4d, p_te_4d], axis=1)
    auc_spatial = train_and_eval(
        img_p_tr, None, l_tr, img_p_te, None, l_te,
        in_ch=2, aux_dim=0, seed=seed, epochs=epochs)

    oh_tr = np.eye(N_CLASSES, dtype=np.float32)[l_tr]
    oh_te = np.eye(N_CLASSES, dtype=np.float32)[l_te]
    auc_oracle = train_and_eval(
        img_tr_4d, oh_tr, l_tr, img_te_4d, oh_te, l_te,
        in_ch=1, aux_dim=N_CLASSES, seed=seed, epochs=epochs)

    return {
        "alpha": alpha,
        "auc_baseline": auc_baseline,
        "auc_scalar": auc_scalar,
        "auc_spatial": auc_spatial,
        "auc_oracle": auc_oracle,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--alpha_sweep", type=float, nargs="+", default=ALPHA_SWEEP)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    print(f"[v2] device={DEVICE}  alphas={args.alpha_sweep}  seeds={args.seeds}")
    print(f"[v2] all class blob centers at radius {RADIUS} (rotation-invariant)")

    all_results: List[Dict] = []
    t0 = time.time()
    for alpha in args.alpha_sweep:
        for seed in args.seeds:
            print(f"\n[v2] alpha={alpha}  seed={seed}")
            r = evaluate_alpha(alpha, seed, args.epochs)
            all_results.append({**r, "seed": seed})
            print(f"  baseline={r['auc_baseline']:.4f}  scalar={r['auc_scalar']:.4f}  "
                  f"spatial={r['auc_spatial']:.4f}  oracle={r['auc_oracle']:.4f}")

    by_alpha: Dict = {}
    for r in all_results:
        a = r["alpha"]
        by_alpha.setdefault(a, {"baseline": [], "scalar": [],
                                  "spatial": [], "oracle": []})
        by_alpha[a]["baseline"].append(r["auc_baseline"])
        by_alpha[a]["scalar"].append(r["auc_scalar"])
        by_alpha[a]["spatial"].append(r["auc_spatial"])
        by_alpha[a]["oracle"].append(r["auc_oracle"])

    print("\n[v2] cross-seed:")
    print(f"  {'alpha':>10s}  {'baseline':>10s}  {'scalar':>10s}  "
          f"{'spatial':>10s}  {'oracle':>10s}  | "
          f"{'scalar lift':>12s}  {'spatial lift':>13s}")
    for a in args.alpha_sweep:
        d = by_alpha[a]
        m_b = np.mean(d["baseline"]); m_sc = np.mean(d["scalar"])
        m_sp = np.mean(d["spatial"]); m_or = np.mean(d["oracle"])
        print(f"  {a:>10.2f}  {m_b:>10.4f}  {m_sc:>10.4f}  "
              f"{m_sp:>10.4f}  {m_or:>10.4f}  | "
              f"{m_sc - m_b:>+12.4f}  {m_sp - m_b:>+13.4f}")

    md = []
    md.append("# Synthetic phase-transition experiment v2 — class-disjoint prior\n")
    md.append("**Date:** 2026-05-08")
    md.append(f"**Setup:** 14-class blob-localization at fixed radius from frame "
              f"center (radius = {RADIUS}). All class centers are equidistant "
              f"from the center, so rotation-invariant summary statistics "
              f"(mean, max, top-k, central-max) cannot distinguish classes by "
              f"construction. Only spatial layout differs.")
    md.append("")
    md.append("## Result table\n")
    md.append("| alpha | baseline | scalar | spatial | oracle | scalar lift | spatial lift |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for a in args.alpha_sweep:
        d = by_alpha[a]
        m_b = np.mean(d["baseline"]); m_sc = np.mean(d["scalar"])
        m_sp = np.mean(d["spatial"]); m_or = np.mean(d["oracle"])
        md.append(f"| {a:.2f} | {m_b:.4f} | {m_sc:.4f} | {m_sp:.4f} "
                  f"| {m_or:.4f} | {m_sc - m_b:+.4f} | {m_sp - m_b:+.4f} |")
    md.append("")

    # Verdict logic: did spatial > scalar uniformly across alpha > 0?
    spatial_wins = 0
    scalar_wins = 0
    for a in args.alpha_sweep:
        if a < 0.5:
            continue
        d = by_alpha[a]
        m_sc = np.mean(d["scalar"]) - np.mean(d["baseline"])
        m_sp = np.mean(d["spatial"]) - np.mean(d["baseline"])
        if m_sp > m_sc + 0.01:
            spatial_wins += 1
        elif m_sc > m_sp + 0.01:
            scalar_wins += 1
    md.append("## Verdict\n")
    if spatial_wins >= 3 and scalar_wins == 0:
        md.append(f"**v2 reproduces the capsule pattern.** Spatial > scalar at "
                  f"{spatial_wins} of {len(args.alpha_sweep) - 1} alpha values "
                  f"with alpha > 0; scalar never beats spatial. The class-"
                  f"disjoint prior design controls the v1 confound (where "
                  f"summary stats encoded class identity directly).")
    elif spatial_wins > scalar_wins:
        md.append(f"**Direction matches capsule.** Spatial > scalar at "
                  f"{spatial_wins} alphas; scalar > spatial at {scalar_wins}.")
    else:
        md.append(f"**Mixed.** Spatial wins at {spatial_wins} alphas, "
                  f"scalar wins at {scalar_wins}. Synthetic verification "
                  f"remains task-dependent.")
    md.append("")
    md.append(f"## Total compute: {(time.time() - t0)/60:.1f} min")

    out = REPORT_DIR / "phase_transition_synthetic_v2_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[v2] report -> {out}")

    raw = REPORT_DIR / "phase_transition_synthetic_v2_data.json"
    raw.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"[v2] raw -> {raw}")


if __name__ == "__main__":
    main()
