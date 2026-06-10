"""
Track B Week 3-4: train cell (b) -- temporal-only sequence transformer
======================================================================

For each of the 6 seeds, train a 4-layer transformer over the cached
per-frame embeddings to predict the center-frame class. Inputs to the
transformer are the 1280-dim RGB-classifier embeddings (frozen) projected
to 256-dim, plus sinusoidal positional encoding over a 16-frame window.

Sequence construction:
  - For each labeled center frame in train/val/test, the window is the
    16 nearest labeled frames in the same video (sorted by frame_number),
    centered on the target frame (with boundary padding by repeating
    edge frames)
  - The window draws from the SAME split as the center frame -- no
    cross-split leakage. This is conservative; a future ablation could
    try cross-split context using only the unlabeled-context frames
    in metadata.csv (but Kvasir-Capsule's labeled set has no
    unlabeled-context frames -- every metadata row is in some split).

Pass criterion (Week 4 main gate):
  Cell (b) cross-seed test macro-AUC ≥ Cell (a) + 0.005
  (Cell (a) baseline measured at 0.7598 ± 0.0271)

Outputs:
  D:/kvasir_capsule/outputs/temporal_cell_b/seed{seed}/best_model.pt
  D:/kvasir_capsule/outputs/temporal_cell_b/seed{seed}/test_predictions.json
  paper/nature-machine-intelligence/docs/track_b_cell_b_report.md

Compute: ~30 min per seed on RTX 5090 (or ~1 hr/seed on GTX 1660 Ti).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
EMB_DIR = Path("D:/kvasir_capsule/outputs/embeddings")
OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs/temporal_cell_b")
INDEX_JSON = HERE / "video_index.json"
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
SPLIT_INT = {"train": 0, "val": 1, "test": 2}

# ---- model hyperparameters ----
WINDOW = 16
EMB_DIM_IN = 1280     # EffNet-B0 pooled feature
EMB_DIM_HIDDEN = 256
N_HEADS = 8
N_LAYERS = 4
FF_DIM = 512
DROPOUT = 0.1

# ---- training hyperparameters ----
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5


# ---------------------------------------------------------------------
# Dataset: yield (window_embeddings, center_label) from cached files
# ---------------------------------------------------------------------

class CachedSequenceDataset(Dataset):
    """Each item is a (16, 1280) tensor of cached embeddings for the
    16-frame window centered on a labeled frame, plus the integer class
    label of the center frame."""

    def __init__(self, cache: Dict[str, np.ndarray],
                 video_index: Dict, split: str, window: int = WINDOW):
        self.cache = cache
        self.window = window
        self.split = split
        target_split = SPLIT_INT[split]

        # filename -> cache index
        fname_to_idx = {fn: i for i, fn in enumerate(cache["filenames"])}

        # Build per-video sequences in the requested split
        self.samples: List[Dict] = []
        for video_id, frames in video_index["by_video"].items():
            # Filter to frames in this split, sorted by frame_number
            in_split = [f for f in frames
                         if SPLIT_INT.get(f["split"], -1) == target_split]
            if not in_split:
                continue
            in_split.sort(key=lambda f: f["frame_number"])
            cache_idx_list = []
            class_list = []
            for f in in_split:
                fn = f"{video_id}_{f['frame_number']}.jpg"
                if fn in fname_to_idx:
                    cache_idx_list.append(fname_to_idx[fn])
                    class_list.append(CLASS_NAMES.index(f["class"]))

            n = len(cache_idx_list)
            if n == 0:
                continue
            half = window // 2
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i - half + window)
                if hi - lo < window:
                    if lo == 0:
                        hi = min(n, window)
                    elif hi == n:
                        lo = max(0, n - window)
                window_idxs = cache_idx_list[lo:hi]
                # pad to window length by repeating last frame
                while len(window_idxs) < window:
                    window_idxs.append(window_idxs[-1])
                self.samples.append({
                    "window_cache_idxs": window_idxs,
                    "center_class": class_list[i],
                    "video_id": video_id,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[i]
        emb = self.cache["embeddings"][s["window_cache_idxs"]]  # (W, 1280)
        return torch.from_numpy(emb).float(), s["center_class"]


# ---------------------------------------------------------------------
# Model: temporal transformer over per-frame embeddings
# ---------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float()
                              * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalTransformer(nn.Module):
    """Cell (b) model: transformer over per-frame embeddings.
    Predicts the class of the center frame (index window // 2)."""

    def __init__(self, in_dim: int = EMB_DIM_IN, hidden: int = EMB_DIM_HIDDEN,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                 ff: int = FF_DIM, window: int = WINDOW,
                 n_classes: int = N_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.window = window
        self.center_idx = window // 2

        self.proj = nn.Linear(in_dim, hidden)
        self.pe = SinusoidalPE(window, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, in_dim)
        h = self.proj(x)            # (B, W, hidden)
        h = self.pe(h)              # add PE
        h = self.transformer(h)     # (B, W, hidden)
        h = self.norm(h)
        h_center = h[:, self.center_idx]  # (B, hidden)
        return self.classifier(h_center)  # (B, n_classes)


# ---------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    inv = 1.0 / counts
    inv = inv * (n_classes / inv.sum())
    return torch.from_numpy(inv).float()


def per_class_auc(logits: np.ndarray, labels: np.ndarray
                  ) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    out: Dict[str, float] = {}
    for j, cname in enumerate(CLASS_NAMES):
        y = (labels == j).astype(np.int32)
        if y.sum() == 0 or y.sum() == len(y):
            out[cname] = float("nan")
            continue
        out[cname] = float(roc_auc_score(y, probs[:, j]))
    return out


def macro_auc(per_class: Dict[str, float]) -> float:
    vals = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[Dict[str, float], float, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    pc = per_class_auc(logits, labels)
    return pc, macro_auc(pc), logits, labels


def train_one_seed(seed: int, video_index: Dict, epochs: int = EPOCHS
                    ) -> Dict:
    print(f"\n{'='*60}\nSEED = {seed}\n{'='*60}")
    cache_path = EMB_DIR / f"seed{seed}_embeddings.npz"
    if not cache_path.exists():
        raise SystemExit(f"missing cache: {cache_path}")
    npz = np.load(cache_path)
    # Materialize in RAM. npz is lazy — leaving it lazy means every
    # __getitem__ decompresses the whole array. Eagerly load once.
    cache = {
        "embeddings": np.array(npz["embeddings"]),  # (N, 1280) float32
        "labels": np.array(npz["labels"]),
        "splits": np.array(npz["splits"]),
        "filenames": np.array(npz["filenames"]),
    }
    npz.close()
    print(f"[seed {seed}] cache loaded; n_frames={cache['embeddings'].shape[0]}  "
          f"emb_size={cache['embeddings'].nbytes / 1e6:.0f} MB")

    train_ds = CachedSequenceDataset(cache, video_index, "train")
    val_ds = CachedSequenceDataset(cache, video_index, "val")
    test_ds = CachedSequenceDataset(cache, video_index, "test")
    print(f"[seed {seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # Loader
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0)

    # Class weights from train labels
    train_labels = np.array([s["center_class"] for s in train_ds.samples])
    cw = compute_class_weights(train_labels, N_CLASSES).to(DEVICE)
    print(f"[seed {seed}] class weights: "
          f"{[f'{c}={w:.2f}' for c, w in zip(CLASS_NAMES, cw.cpu().tolist())]}")

    # Model
    torch.manual_seed(seed)
    model = TemporalTransformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seed {seed}] trainable params: {n_params:,}")

    ce_loss = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out_dir = OUTPUT_ROOT / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_macro = -1.0
    best_epoch = 0
    no_improve = 0
    history: List[Dict] = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            loss = ce_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running / max(1, seen)

        val_pc, val_macro, _, _ = evaluate(model, val_loader)
        scheduler.step()
        elapsed = (time.time() - t0) / 60
        print(f"[seed {seed}] epoch {epoch:2d}/{epochs}  "
              f"train_ce={train_loss:.4f}  val_macro_auc={val_macro:.4f}  "
              f"elapsed={elapsed:.1f} min")
        history.append({"epoch": epoch, "train_ce": train_loss,
                          "val_macro_auc": val_macro})

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_macro_auc": val_macro,
                "val_per_class": val_pc,
                "history": history,
            }, out_dir / "best_model.pt")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"[seed {seed}] early stop at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    # Load best checkpoint, evaluate on test
    ckpt = torch.load(out_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_pc, test_macro, test_logits, test_labels = evaluate(model, test_loader)
    print(f"[seed {seed}] best val macro-AUC = {best_val_macro:.4f} "
          f"(epoch {best_epoch})")
    print(f"[seed {seed}] TEST macro-AUC = {test_macro:.4f}")
    for c, a in test_pc.items():
        if not np.isnan(a):
            print(f"           {c:25s}: {a:.4f}")

    # Save predictions for downstream cells
    np.savez_compressed(out_dir / "test_predictions.npz",
                        logits=test_logits, labels=test_labels)

    return {
        "seed": seed,
        "best_val_macro_auc": best_val_macro,
        "best_epoch": best_epoch,
        "test_macro_auc": test_macro,
        "test_per_class": test_pc,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    seeds = args.only_seeds or SEEDS
    print(f"[main] cell (b) training on {seeds}")
    print(f"[main] device={DEVICE}  window={WINDOW}  hidden={EMB_DIM_HIDDEN}  "
          f"n_layers={N_LAYERS}  n_heads={N_HEADS}")

    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))

    results: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        r = train_one_seed(seed, video_index, epochs=args.epochs)
        results.append(r)
        # Incremental save
        (REPORT_DIR / "track_b_cell_b_predictions.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    # Cross-seed report
    macros = np.array([r["test_macro_auc"] for r in results])
    cs_mean = float(np.mean(macros))
    cs_std = float(np.std(macros))
    print(f"\n[main] CROSS-SEED test macro-AUC = {cs_mean:.4f} +- {cs_std:.4f}")

    cell_a_ref = 0.7598
    delta = cs_mean - cell_a_ref
    print(f"[main] cell (a) reference = {cell_a_ref:.4f}")
    print(f"[main] delta = {delta:+.4f}  "
          f"({'PASS' if delta >= 0.005 else 'FAIL'} Week-4 main gate "
          f"≥+0.005 threshold)")

    # Per-class cross-seed
    pc_agg: Dict[str, Dict[str, float]] = {}
    for cname in CLASS_NAMES:
        vals = np.array([r["test_per_class"].get(cname, float("nan"))
                          for r in results])
        if not np.isnan(vals).all():
            pc_agg[cname] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
            }

    md = []
    md.append("# Track B Week 3-4: cell (b) temporal-only training\n")
    md.append(f"**Date:** 2026-05-07")
    md.append(f"**Seeds:** {[r['seed'] for r in results]}")
    md.append(f"**Window:** {WINDOW} frames; transformer 4 layers x 8 heads x "
              f"256 hidden; classifier head on center-frame token")
    md.append("")
    md.append("## Per-seed test macro-AUC\n")
    md.append("| Seed | Test macro-AUC | Best val macro-AUC | Best epoch |")
    md.append("|---:|---:|---:|---:|")
    for r in results:
        md.append(f"| {r['seed']} | {r['test_macro_auc']:.4f} "
                  f"| {r['best_val_macro_auc']:.4f} | {r['best_epoch']} |")
    md.append("")
    md.append(f"**Cross-seed test macro-AUC:** {cs_mean:.4f} +- {cs_std:.4f}")
    md.append(f"**Cell (a) reference:** {cell_a_ref:.4f}")
    md.append(f"**Delta vs cell (a):** {delta:+.4f}")
    md.append("")
    md.append("## Per-class cross-seed test AUC\n")
    md.append("| Class | Mean +- sigma |")
    md.append("|---|---:|")
    for c in CLASS_NAMES:
        if c in pc_agg:
            a = pc_agg[c]
            md.append(f"| {c} | {a['mean']:.3f} +- {a['std']:.3f} |")
    md.append("")
    md.append("## Verdict (Week-4 main gate)\n")
    if delta >= 0.005:
        md.append(f"**PASS.** Cell (b) lifts macro-AUC by {delta:+.4f} >= +0.005 "
                  "over per-frame baseline. C2 (temporal coherence) "
                  "contributes signal. Proceed to Month 2: add C1 (cell c) and "
                  "C3 (cell d) channels.")
    elif delta >= 0.0:
        md.append(f"**MARGINAL.** Cell (b) lift = {delta:+.4f} > 0 but below the "
                  "+0.005 threshold. Direction is correct but not enough to "
                  "justify the multi-channel framework on this evidence "
                  "alone. Decision: continue to cells (c)(d) as one week of "
                  "extra investment, but be ready to pivot to MedIA if cells "
                  "(c)(d) also fail to clear +0.005.")
    else:
        md.append(f"**FAIL.** Cell (b) macro-AUC delta = {delta:+.4f} < 0. "
                  "Temporal aggregation does NOT lift on capsule. The "
                  "framework's C2 channel is empirically dead on Kvasir-Capsule. "
                  "Halt Track B and rely on Track A (MedIA submission of v2 "
                  "evidence).")
    md.append(f"\nTotal compute: {(time.time() - t0)/60:.1f} min for "
              f"{len(seeds)} seeds")

    out_path = REPORT_DIR / "track_b_cell_b_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out_path}")
    print(f"[main] elapsed: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
