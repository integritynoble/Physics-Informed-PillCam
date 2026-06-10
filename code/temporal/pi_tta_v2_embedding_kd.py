"""
PI-TTA v2, experiment 1 — embedding-level distillation.
========================================================

The cheapest concrete test of the "cell (b+) gap is recoverable from
RGB pixels via the right test-time mapping" hypothesis. Pure CPU,
~minutes to run, uses only the already-cached RGB and +PI embeddings.

What it does:
  1. Train a small MLP `g: R^1280 -> R^1280` on the train split,
     supervised by paired (RGB embedding, +PI embedding) for the same
     frame and seed.
  2. At test time, given an RGB embedding `e_rgb`, compute the
     predicted +PI embedding `e_pi_hat = g(e_rgb)`.
  3. Train a 14-class linear probe on `e_pi_hat` over the train split,
     evaluate on test.
  4. Compare three macro-AUCs across 6 seeds:
       - probe_rgb     : trained and tested on actual RGB embeddings
                         (this is cell-(a) RGB linear-probe baseline)
       - probe_pi_real : trained and tested on actual +PI embeddings
                         (this is cell-(a) +PI linear-probe upper bound)
       - probe_pi_hat  : trained on g(RGB) predictions, tested on g(RGB)
                         (the recoverability score)

If `probe_pi_hat` recovers >=50% of (probe_pi_real - probe_rgb), then
the +PI signal is present in the RGB pixels but not extracted by the
RGB classifier in normal training. This is the empirical foothold
PI-TTA v2 needs in order to be worth implementing as a full TTA loop
on the backbone.

If `probe_pi_hat` recovers <20% of the gap, the +PI lift is genuinely
information that's only present in the prior channels — PI-TTA in any
form is unlikely to work. We would then drop PI-TTA from the NMI plan
and lean on the cell (b+)/(e+) input-fusion result as the cell.

Outputs:
  paper/nature-machine-intelligence/docs/pi_tta_v2_embedding_kd_report.md
"""

from __future__ import annotations

import argparse
import json
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
EMB_RGB_DIR = Path("D:/kvasir_capsule/outputs/embeddings")
EMB_PI_DIR = Path("D:/kvasir_capsule/outputs/embeddings_pi")
REPORT_DIR = HERE.parent.parent / "docs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [41, 42, 43, 44, 45, 47]

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
N_CLASSES = len(CLASS_NAMES)
EMB_DIM = 1280

# MLP for the RGB -> +PI mapping
MLP_HIDDEN = 1024
MLP_LAYERS = 3
MLP_DROPOUT = 0.1
MLP_EPOCHS = 30
MLP_BATCH = 256
MLP_LR = 1e-3
MLP_WD = 1e-4


def macro_auc(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    from sklearn.metrics import roc_auc_score
    pc: Dict[str, float] = {}
    for j, c in enumerate(CLASS_NAMES):
        y = (labels == j).astype(np.int32)
        if y.sum() == 0 or y.sum() == len(y):
            pc[c] = float("nan")
            continue
        pc[c] = float(roc_auc_score(y, probs[:, j]))
    vals = [v for v in pc.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan"), pc


def load_seed(seed: int) -> Dict[str, np.ndarray]:
    """Load and align RGB and +PI embeddings for one seed."""
    rgb = np.load(EMB_RGB_DIR / f"seed{seed}_embeddings.npz")
    pi = np.load(EMB_PI_DIR / f"seed{seed}_embeddings.npz")

    fn_rgb = list(rgb["filenames"])
    fn_pi = list(pi["filenames"])
    if fn_rgb != fn_pi:
        # Reindex to match by filename
        fn_to_idx_pi = {fn: i for i, fn in enumerate(fn_pi)}
        idx_pi = [fn_to_idx_pi[fn] for fn in fn_rgb if fn in fn_to_idx_pi]
        keep = [i for i, fn in enumerate(fn_rgb) if fn in fn_to_idx_pi]
        e_rgb = np.array(rgb["embeddings"])[keep]
        labels = np.array(rgb["labels"])[keep]
        splits = np.array(rgb["splits"])[keep]
        e_pi = np.array(pi["embeddings"])[idx_pi]
    else:
        e_rgb = np.array(rgb["embeddings"])
        e_pi = np.array(pi["embeddings"])
        labels = np.array(rgb["labels"])
        splits = np.array(rgb["splits"])

    rgb.close(); pi.close()
    return {
        "e_rgb": e_rgb.astype(np.float32),
        "e_pi": e_pi.astype(np.float32),
        "labels": labels.astype(np.int64),
        "splits": splits.astype(np.int64),
    }


class EmbeddingMLP(nn.Module):
    """Maps RGB embedding -> +PI embedding."""

    def __init__(self, in_dim: int = EMB_DIM, hidden: int = MLP_HIDDEN,
                 out_dim: int = EMB_DIM, n_layers: int = MLP_LAYERS,
                 dropout: float = MLP_DROPOUT):
        super().__init__()
        layers: List[nn.Module] = []
        d_in = in_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d_in = hidden
        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mapper(e_rgb_train: np.ndarray, e_pi_train: np.ndarray,
                  e_rgb_val: np.ndarray, e_pi_val: np.ndarray,
                  seed: int) -> EmbeddingMLP:
    """Train the RGB -> +PI MLP on the train split. Use val for early
    stop on cosine similarity."""
    torch.manual_seed(seed)
    model = EmbeddingMLP().to(DEVICE)

    X_train = torch.from_numpy(e_rgb_train)
    Y_train = torch.from_numpy(e_pi_train)
    X_val = torch.from_numpy(e_rgb_val).to(DEVICE)
    Y_val = torch.from_numpy(e_pi_val).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                                batch_size=MLP_BATCH, shuffle=True,
                                num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)

    best_val_cos = -1.0
    best_state = None
    no_improve = 0
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()
        train_mse = running / max(1, seen)

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            val_mse = F.mse_loss(pred_val, Y_val).item()
            val_cos = F.cosine_similarity(pred_val, Y_val, dim=-1).mean().item()
        if val_cos > best_val_cos:
            best_val_cos = val_cos
            best_state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def linear_probe_macro_auc(e_train: np.ndarray, y_train: np.ndarray,
                              e_test: np.ndarray, y_test: np.ndarray,
                              seed: int) -> Tuple[float, Dict[str, float]]:
    """Closed-form sklearn LogisticRegression as the linear probe."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0,
                              class_weight="balanced",
                              random_state=seed,
                              solver="lbfgs")
    clf.fit(e_train, y_train)
    probs = clf.predict_proba(e_test)
    return macro_auc(probs, y_test)


def evaluate_seed(seed: int) -> Dict:
    print(f"\n{'='*60}\nSEED = {seed}\n{'='*60}")
    d = load_seed(seed)
    e_rgb = d["e_rgb"]; e_pi = d["e_pi"]
    y = d["labels"]; s = d["splits"]
    train = (s == 0); val = (s == 1); test = (s == 2)
    print(f"[seed {seed}] train={train.sum()} val={val.sum()} test={test.sum()}")

    # Step 1: train mapper g: RGB -> +PI on train+val
    print(f"[seed {seed}] training RGB -> +PI mapper...")
    t0 = time.time()
    mapper = train_mapper(e_rgb[train], e_pi[train],
                            e_rgb[val], e_pi[val], seed=seed)
    print(f"[seed {seed}] mapper trained in {time.time()-t0:.1f} sec")

    # Step 2: apply mapper to all splits
    mapper.eval()
    with torch.no_grad():
        x_all = torch.from_numpy(e_rgb).to(DEVICE)
        e_pi_hat = mapper(x_all).cpu().numpy()

    # Step 3: three linear probes
    print(f"[seed {seed}] training linear probes...")
    auc_rgb, pc_rgb = linear_probe_macro_auc(
        e_rgb[train], y[train], e_rgb[test], y[test], seed)
    auc_pi_real, pc_pi_real = linear_probe_macro_auc(
        e_pi[train], y[train], e_pi[test], y[test], seed)
    auc_pi_hat, pc_pi_hat = linear_probe_macro_auc(
        e_pi_hat[train], y[train], e_pi_hat[test], y[test], seed)

    gap = auc_pi_real - auc_rgb
    recovered = auc_pi_hat - auc_rgb
    frac = recovered / max(1e-6, gap)
    print(f"[seed {seed}] probe_rgb     = {auc_rgb:.4f}")
    print(f"[seed {seed}] probe_pi_real = {auc_pi_real:.4f}  (gap = {gap:+.4f})")
    print(f"[seed {seed}] probe_pi_hat  = {auc_pi_hat:.4f}  (recovered = "
          f"{recovered:+.4f}, frac = {frac*100:.1f}% of gap)")

    return {
        "seed": seed,
        "auc_rgb": auc_rgb,
        "auc_pi_real": auc_pi_real,
        "auc_pi_hat": auc_pi_hat,
        "gap": gap,
        "recovered": recovered,
        "frac_recovered": frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    seeds = args.only_seeds or SEEDS
    print(f"[main] device={DEVICE}  seeds={seeds}")
    print(f"[main] embedding KD experiment for PI-TTA v2 design")

    results: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        try:
            r = evaluate_seed(seed)
            results.append(r)
        except FileNotFoundError as exc:
            print(f"[seed {seed}] missing file: {exc}; skipping")
            continue

    # Cross-seed summary
    if not results:
        print("[main] no results")
        return
    arr_rgb = np.array([r["auc_rgb"] for r in results])
    arr_pi_real = np.array([r["auc_pi_real"] for r in results])
    arr_pi_hat = np.array([r["auc_pi_hat"] for r in results])
    arr_recovered = np.array([r["recovered"] for r in results])
    arr_frac = np.array([r["frac_recovered"] for r in results])

    print(f"\n[main] cross-seed:")
    print(f"  probe_rgb     = {arr_rgb.mean():.4f} +- {arr_rgb.std():.4f}")
    print(f"  probe_pi_real = {arr_pi_real.mean():.4f} +- {arr_pi_real.std():.4f}")
    print(f"  probe_pi_hat  = {arr_pi_hat.mean():.4f} +- {arr_pi_hat.std():.4f}")
    print(f"  recovered     = {arr_recovered.mean():+.4f} +- {arr_recovered.std():.4f}")
    print(f"  frac of gap   = {arr_frac.mean()*100:.1f}% +- {arr_frac.std()*100:.1f}%")

    # Decision: > 50% recovery -> PI-TTA v2 promising
    #          < 20% recovery -> drop PI-TTA from NMI plan
    cs_frac = arr_frac.mean() * 100
    if cs_frac >= 50:
        verdict = ("**PROMISING.** RGB embeddings carry >=50% of the +PI lift "
                   "via a learned mapping; PI-TTA v2 (test-time alignment) is "
                   "worth implementing on the backbone.")
    elif cs_frac >= 20:
        verdict = ("**MARGINAL.** RGB embeddings carry 20-50% of the +PI lift. "
                   "PI-TTA v2 may give partial recovery; the test-time backbone "
                   "experiment is borderline-worth-running. Consider a "
                   "feature-level distillation training-time variant instead.")
    else:
        verdict = ("**NULL.** RGB embeddings carry <20% of the +PI lift. The "
                   "+PI lift requires actual prior input access (cannot be "
                   "recovered from RGB pixels via post-hoc mapping). Drop "
                   "PI-TTA from the NMI plan; lean on the cell (b+)/(e+) "
                   "input-fusion result as the algorithmic contribution.")

    md = []
    md.append("# PI-TTA v2, experiment 1 — embedding-level distillation report\n")
    md.append("**Date:** 2026-05-07")
    md.append("**Question:** Does an MLP `g: e_rgb -> e_pi` recover the "
              "cell (b)/(b+) macro-AUC gap?")
    md.append("**Method:** Per seed, train MLP on (RGB, +PI) embedding pairs "
              "from train split, then evaluate three linear probes: probe on "
              "RGB embeddings, probe on real +PI embeddings, probe on g(RGB) "
              "predicted +PI embeddings.")
    md.append("")
    md.append("## Per-seed results\n")
    md.append("| Seed | probe_rgb | probe_pi_real | probe_pi_hat | gap | recovered | frac of gap |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        md.append(f"| {r['seed']} | {r['auc_rgb']:.4f} "
                  f"| {r['auc_pi_real']:.4f} | {r['auc_pi_hat']:.4f} "
                  f"| {r['gap']:+.4f} | {r['recovered']:+.4f} "
                  f"| {r['frac_recovered']*100:.1f}% |")
    md.append("")
    md.append(f"**Cross-seed mean:**")
    md.append(f"- probe_rgb     = {arr_rgb.mean():.4f} ± {arr_rgb.std():.4f}")
    md.append(f"- probe_pi_real = {arr_pi_real.mean():.4f} ± {arr_pi_real.std():.4f}")
    md.append(f"- probe_pi_hat  = {arr_pi_hat.mean():.4f} ± {arr_pi_hat.std():.4f}")
    md.append(f"- recovered     = {arr_recovered.mean():+.4f} ± {arr_recovered.std():.4f}")
    md.append(f"- frac of gap   = {arr_frac.mean()*100:.1f}% ± {arr_frac.std()*100:.1f}%")
    md.append("")
    md.append(f"## Verdict\n")
    md.append(verdict)
    md.append("")
    md.append(f"## Implications for the NMI plan")
    if cs_frac >= 50:
        md.append("Proceed to PI-TTA v2 implementation on the backbone "
                  "(`pi_tta_v2_backbone.py`, ~1 day to write + ~6 GPU-h to "
                  "sweep). Expected lift: a fraction of the cell (b+) gap, "
                  "with the +PI teacher providing the upper bound.")
    elif cs_frac >= 20:
        md.append("Reassess. Consider a training-time KD variant (RGB "
                  "student trained with +PI teacher's logits as soft target) "
                  "instead of test-time. KD is less novel but has higher "
                  "expected-value per GPU-day.")
    else:
        md.append("Drop PI-TTA from the NMI plan. The NMI bid stands on "
                  "Track B (capsule + cardiac MRI) and the four-variant "
                  "C1 boundary as the methodological contribution.")

    out = REPORT_DIR / "pi_tta_v2_embedding_kd_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out}")
    print(f"[main] elapsed: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
