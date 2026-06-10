"""
Synthetic verification of the parameterization-mechanism phase transition.
============================================================================

Empirical hypothesis (from the four-variant C1 boundary on capsule):

  > For a fixed classifier f_theta trained on (X, Y), the test-time lift
  > of a deterministic prior P(X) depends on the *bandwidth* at which
  > the prior is delivered. Below a critical bandwidth, P contributes
  > zero (or negative) lift (the "summary-stat" regime, cells c/c'/c'').
  > Above the critical bandwidth, P contributes positive lift (the
  > "spatial-channel" regime, cell b+).

This script builds a synthetic 14-class classification problem where
the prior's bandwidth is a controllable scalar `alpha`, and runs the
4-variant ablation across a sweep `alpha in {0, 1, 2, 4, 8, 16, 32, 64}`
to visualize the transition.

Synthetic setup:
  - Image: 32x32 grayscale (small for fast iteration)
  - 14 classes, each defined by the location of a colored "lesion" in
    the image: a small Gaussian blob whose (x, y) center is class-specific
  - The "RGB" backbone is a small CNN that sees the image but cannot
    perfectly localize the blob (we cap its spatial receptive field)
  - The "analytic prior" P(x) for an image is the heatmap of where the
    blob is, blurred by a Gaussian with variance `1/alpha`. As alpha
    grows, the prior is more spatially precise (high-bandwidth, like
    pixel-level P_blood). As alpha shrinks toward 0, the prior is
    nearly uniform (low-bandwidth, like a 0-dim global summary).

Four C1 parameterizations (analogous to capsule cells):
  scalar    : feed mean(P), max(P), top-k(P) as auxiliary scalars to head
              (bandwidth-collapsed)
  topology  : feed connected-component features of P > 0.5 to head
              (lossy spatial summary)
  spatial   : feed P as a spatial channel concatenated with image at input
              (high-bandwidth pixel-level access)
  pixel     : the upper-bound — the model can directly see the blob
              location label as additional input

Expected pattern (the phase transition):
  - At alpha = 0 (uniform prior): all variants give 0 lift
  - At small alpha: scalar/topology may give small positive lift
    because the prior is still globally informative
  - At critical alpha (somewhere ~4-8): spatial variant lifts sharply,
    summary variants flatten or regress
  - At alpha = inf (delta function): spatial saturates at the
    pixel-label upper bound

If the transition is sharp at some alpha_c, this is direct evidence
for a phase transition in the parameterization-mechanism boundary.

Output:
  paper/nature-machine-intelligence/docs/phase_transition_synthetic_report.md
  paper/nature-machine-intelligence/docs/phase_transition_synthetic_curve.png
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

# Synthetic dataset hyperparameters
N_CLASSES = 14
IMAGE_SIZE = 32
N_TRAIN = 5000
N_VAL = 1000
N_TEST = 2000
NOISE_SIGMA = 0.20
BLOB_SIGMA = 1.5
ALPHA_SWEEP = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1e6]
SEEDS = [41, 42, 43]
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-4

# Hand-set class centers (14 distinct locations on a 32x32 grid)
CLASS_CENTERS = [
    (8, 8), (8, 16), (8, 24),
    (16, 4), (16, 12), (16, 20), (16, 28),
    (24, 8), (24, 16), (24, 24),
    (12, 12), (12, 20), (20, 12), (20, 20),
]


def gaussian_blob(cy: int, cx: int, sigma: float = BLOB_SIGMA,
                    H: int = IMAGE_SIZE, W: int = IMAGE_SIZE) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                            np.arange(W, dtype=np.float32),
                            indexing="ij")
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))


def render_image(class_id: int, rng: np.random.Generator) -> np.ndarray:
    """Render a 32x32 grayscale image: a Gaussian blob at the class
    location, plus noise, plus background. Receptive-field-limited
    classifiers should struggle to localize the center precisely."""
    cy, cx = CLASS_CENTERS[class_id]
    # Jitter the blob center within +/- 1 pixel to make the task harder
    cy_j = cy + rng.normal(0, 0.6)
    cx_j = cx + rng.normal(0, 0.6)
    blob = gaussian_blob(cy_j, cx_j, sigma=BLOB_SIGMA)
    bg = rng.normal(0.5, 0.05, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    noise = rng.normal(0, NOISE_SIGMA, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    img = np.clip(bg + 0.6 * blob + noise, 0.0, 1.0)
    return img.astype(np.float32)


def render_prior(class_id: int, alpha: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Render the analytic prior at given alpha. alpha controls
    spatial precision: alpha = 0 => uniform (zero info). alpha very
    large => delta function at the true class center."""
    if alpha < 1e-3:
        return np.full((IMAGE_SIZE, IMAGE_SIZE), 0.5, dtype=np.float32)
    cy, cx = CLASS_CENTERS[class_id]
    sigma_prior = 1.0 / np.sqrt(alpha)
    sigma_prior = max(0.3, sigma_prior)
    p = gaussian_blob(cy, cx, sigma=sigma_prior)
    p = p / max(p.max(), 1e-6)
    return p.astype(np.float32)


def build_dataset(seed: int, alpha: float, n: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    """8 scalar global summaries (analogous to capsule's cell c)."""
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
    """Compact CNN with deliberately limited spatial precision via a
    tight global-pool architecture. Mimics the capsule backbone's
    behavior: 'sees' the image but can't perfectly localize."""

    def __init__(self, in_ch: int = 1, hidden: int = 64,
                 n_classes: int = N_CLASSES,
                 aux_dim: int = 0):
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

    def forward(self, x: torch.Tensor,
                 aux: torch.Tensor = None) -> torch.Tensor:
        h = F.gelu(self.conv1(x))
        h = F.max_pool2d(h, 2)
        h = F.gelu(self.conv2(h))
        h = F.max_pool2d(h, 2)
        h = F.gelu(self.conv3(h))
        h = self.pool(h).flatten(1)
        if self.aux_dim > 0:
            assert aux is not None
            h = torch.cat([h, aux], dim=-1)
        return self.head(h)


def train_and_eval(images_train, aux_train, labels_train,
                     images_val, aux_val, labels_val,
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
            x = X_train[idx]
            y = y_train[idx]
            a = A_train[idx] if A_train is not None else None
            logits = model(x, a)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        # Eval test set in mini-batches to fit memory
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
    img_train, prior_train, lab_train = build_dataset(seed * 100 + 1, alpha, N_TRAIN)
    img_val, prior_val, lab_val = build_dataset(seed * 100 + 2, alpha, N_VAL)
    img_test, prior_test, lab_test = build_dataset(seed * 100 + 3, alpha, N_TEST)

    img_train_4d = img_train[:, None, :, :]
    img_val_4d = img_val[:, None, :, :]
    img_test_4d = img_test[:, None, :, :]

    prior_train_4d = prior_train[:, None, :, :]
    prior_val_4d = prior_val[:, None, :, :]
    prior_test_4d = prior_test[:, None, :, :]

    # variant 1: baseline (RGB-only equivalent — image alone)
    auc_baseline = train_and_eval(
        img_train_4d, None, lab_train,
        img_val_4d, None, lab_val,
        img_test_4d, None, lab_test,
        in_ch=1, aux_dim=0, seed=seed, epochs=epochs)

    # variant 2: scalar (8 global summaries of the prior)
    sc_train = compute_scalar_features(prior_train).astype(np.float32)
    sc_val = compute_scalar_features(prior_val).astype(np.float32)
    sc_test = compute_scalar_features(prior_test).astype(np.float32)
    sc_mean = sc_train.mean(axis=0); sc_std = sc_train.std(axis=0) + 1e-6
    sc_train_n = (sc_train - sc_mean) / sc_std
    sc_val_n = (sc_val - sc_mean) / sc_std
    sc_test_n = (sc_test - sc_mean) / sc_std
    auc_scalar = train_and_eval(
        img_train_4d, sc_train_n, lab_train,
        img_val_4d, sc_val_n, lab_val,
        img_test_4d, sc_test_n, lab_test,
        in_ch=1, aux_dim=8, seed=seed, epochs=epochs)

    # variant 3: spatial (image + prior as a 2-channel input)
    img_with_prior_train = np.concatenate([img_train_4d, prior_train_4d], axis=1)
    img_with_prior_val = np.concatenate([img_val_4d, prior_val_4d], axis=1)
    img_with_prior_test = np.concatenate([img_test_4d, prior_test_4d], axis=1)
    auc_spatial = train_and_eval(
        img_with_prior_train, None, lab_train,
        img_with_prior_val, None, lab_val,
        img_with_prior_test, None, lab_test,
        in_ch=2, aux_dim=0, seed=seed, epochs=epochs)

    # variant 4: oracle (the true class label fed as a one-hot to head)
    oracle_train = np.eye(N_CLASSES, dtype=np.float32)[lab_train]
    oracle_val = np.eye(N_CLASSES, dtype=np.float32)[lab_val]
    oracle_test = np.eye(N_CLASSES, dtype=np.float32)[lab_test]
    auc_oracle = train_and_eval(
        img_train_4d, oracle_train, lab_train,
        img_val_4d, oracle_val, lab_val,
        img_test_4d, oracle_test, lab_test,
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

    print(f"[main] device={DEVICE}  alphas={args.alpha_sweep}  seeds={args.seeds}")
    print(f"[main] epochs={args.epochs}, train={N_TRAIN}, test={N_TEST}")

    all_results: List[Dict] = []
    t0 = time.time()
    for alpha in args.alpha_sweep:
        for seed in args.seeds:
            print(f"\n[run] alpha={alpha}  seed={seed}")
            r = evaluate_alpha(alpha, seed, args.epochs)
            all_results.append({**r, "seed": seed})
            print(f"  baseline={r['auc_baseline']:.4f}  scalar={r['auc_scalar']:.4f}  "
                  f"spatial={r['auc_spatial']:.4f}  oracle={r['auc_oracle']:.4f}")

    # Aggregate
    by_alpha: Dict[float, Dict[str, List[float]]] = {}
    for r in all_results:
        a = r["alpha"]
        by_alpha.setdefault(a, {"baseline": [], "scalar": [],
                                  "spatial": [], "oracle": []})
        by_alpha[a]["baseline"].append(r["auc_baseline"])
        by_alpha[a]["scalar"].append(r["auc_scalar"])
        by_alpha[a]["spatial"].append(r["auc_spatial"])
        by_alpha[a]["oracle"].append(r["auc_oracle"])

    print("\n[main] cross-seed:")
    print(f"  {'alpha':>10s}  {'baseline':>10s}  {'scalar':>10s}  "
          f"{'spatial':>10s}  {'oracle':>10s}  | "
          f"{'scalar lift':>12s}  {'spatial lift':>13s}")
    for a in args.alpha_sweep:
        d = by_alpha[a]
        m_b = np.mean(d["baseline"])
        m_sc = np.mean(d["scalar"])
        m_sp = np.mean(d["spatial"])
        m_or = np.mean(d["oracle"])
        print(f"  {a:>10.2f}  {m_b:>10.4f}  {m_sc:>10.4f}  "
              f"{m_sp:>10.4f}  {m_or:>10.4f}  | "
              f"{m_sc - m_b:>+12.4f}  {m_sp - m_b:>+13.4f}")

    # Markdown report
    md = []
    md.append("# Synthetic phase-transition verification\n")
    md.append("**Date:** 2026-05-07")
    md.append(f"**Setup:** 14-class synthetic blob-localization. Image is 32x32 "
              f"grayscale; class identity = blob location. The analytic prior "
              f"is the blob heatmap blurred with sigma = 1/sqrt(alpha). "
              f"alpha = 0 -> uniform prior (zero info); alpha -> infinity -> "
              f"delta function (perfect localization). Three C1 "
              f"parameterizations + an oracle upper bound:")
    md.append("- baseline: image alone (no prior)")
    md.append("- scalar: image + 8 global summary scalars of P (cell c analogue)")
    md.append("- spatial: image + P as a 2-channel input (cell b+ analogue)")
    md.append("- oracle: image + true class label one-hot (upper bound)")
    md.append("")
    md.append(f"Per (alpha, seed): train CNN for {args.epochs} epochs, "
              f"evaluate macro-AUC on held-out test split. Aggregated "
              f"across seeds {args.seeds}.")
    md.append("")
    md.append("## Result table\n")
    md.append("| alpha | baseline | scalar | spatial | oracle | scalar lift | spatial lift |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for a in args.alpha_sweep:
        d = by_alpha[a]
        m_b = np.mean(d["baseline"])
        m_sc = np.mean(d["scalar"])
        m_sp = np.mean(d["spatial"])
        m_or = np.mean(d["oracle"])
        md.append(f"| {a:.2f} | {m_b:.4f} | {m_sc:.4f} | "
                  f"{m_sp:.4f} | {m_or:.4f} | {m_sc - m_b:+.4f} | "
                  f"{m_sp - m_b:+.4f} |")
    md.append("")
    md.append("## Interpretation\n")
    md.append("The phase transition predicts:")
    md.append("- At alpha = 0 (uniform prior): both scalar and spatial give ~0 lift.")
    md.append("- At small alpha: scalar may give small positive lift (bandwidth still useful as a global hint).")
    md.append("- At critical alpha_c: spatial lifts sharply; scalar saturates.")
    md.append("- At alpha -> infinity: spatial saturates near oracle.")
    md.append("")
    md.append("If the spatial-vs-scalar lift gap widens monotonically with alpha "
              "and crosses 1% somewhere in alpha in [4, 16], that is direct "
              "evidence for a sharp parameterization-mechanism boundary.")
    md.append("")
    md.append(f"## Total compute: {(time.time() - t0)/60:.1f} min")

    out = REPORT_DIR / "phase_transition_synthetic_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out}")

    # Save raw data for plotting later
    raw = REPORT_DIR / "phase_transition_synthetic_data.json"
    raw.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"[main] raw -> {raw}")


if __name__ == "__main__":
    main()
