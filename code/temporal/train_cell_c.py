"""
Track B Month 2: cell (c) — temporal + C1 structural prior
============================================================

Same architecture as cell (b), but each frame's input to the transformer
becomes:

    h_s = concat[ e_s (1280d backbone embedding), s_s (8d C1 features) ]   # 1288d
        → Linear(1288 → 256) → +PE → TransformerEncoder → classifier

The 8d C1 features are precomputed analytic-prior structural scalars
(mean, max, top-1%, top-5%, frac>0.5, frac>0.7, central_max, central_top_1pct)
from `build_c1_features.py`. The C1 features are seed-independent
(deterministic function of RGB).

Pass criterion (cell c → cell b): per-class lift on vascular classes
(Lymph, Erosion, RMV) where C1 is expected to help; macro-AUC ≥ cell (b).

Outputs:
  D:/kvasir_capsule/outputs/temporal_cell_c/seed{seed}/best_model.pt
  paper/nature-machine-intelligence/docs/track_b_cell_c_report.md

Compute: ~13 min total for 6 seeds (similar to cell b).
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
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
EMB_DIR = Path("D:/kvasir_capsule/outputs/embeddings")
C1_PATH = Path("D:/kvasir_capsule/outputs/c1_features.npz")
OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs/temporal_cell_c")
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

WINDOW = 16
EMB_DIM_RGB = 1280
C1_DIM = 8
EMB_DIM_IN = EMB_DIM_RGB + C1_DIM   # 1288
EMB_DIM_HIDDEN = 256
N_HEADS = 8
N_LAYERS = 4
FF_DIM = 512
DROPOUT = 0.1

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5


class CachedSequenceDatasetC1(Dataset):
    """Like cell (b) but also pulls C1 features per frame and concatenates
    to the embedding to form 1288-dim per-frame input."""

    def __init__(self, cache_emb: Dict[str, np.ndarray],
                 cache_c1: Dict[str, np.ndarray],
                 video_index: Dict, split: str, window: int = WINDOW):
        self.cache_emb = cache_emb
        self.cache_c1 = cache_c1
        self.window = window
        target_split = SPLIT_INT[split]

        # Map filename → cache index in BOTH caches (must align)
        emb_fn_to_idx = {fn: i for i, fn in enumerate(cache_emb["filenames"])}
        c1_fn_to_idx = {fn: i for i, fn in enumerate(cache_c1["filenames"])}

        # Z-score normalization parameters for C1 features computed
        # over train split only (no test leakage)
        train_mask = cache_c1["splits"] == 0
        self.c1_mean = cache_c1["features"][train_mask].mean(axis=0)
        self.c1_std = cache_c1["features"][train_mask].std(axis=0) + 1e-6

        self.samples: List[Dict] = []
        for video_id, frames in video_index["by_video"].items():
            in_split = [f for f in frames
                         if SPLIT_INT.get(f["split"], -1) == target_split]
            if not in_split:
                continue
            in_split.sort(key=lambda f: f["frame_number"])
            emb_idxs = []
            c1_idxs = []
            class_list = []
            for f in in_split:
                fn = f"{video_id}_{f['frame_number']}.jpg"
                if fn in emb_fn_to_idx and fn in c1_fn_to_idx:
                    emb_idxs.append(emb_fn_to_idx[fn])
                    c1_idxs.append(c1_fn_to_idx[fn])
                    class_list.append(CLASS_NAMES.index(f["class"]))

            n = len(emb_idxs)
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
                w_emb = emb_idxs[lo:hi]
                w_c1 = c1_idxs[lo:hi]
                while len(w_emb) < window:
                    w_emb.append(w_emb[-1])
                    w_c1.append(w_c1[-1])
                self.samples.append({
                    "emb_idxs": w_emb,
                    "c1_idxs": w_c1,
                    "center_class": class_list[i],
                    "video_id": video_id,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[i]
        emb = self.cache_emb["embeddings"][s["emb_idxs"]]   # (W, 1280)
        c1 = self.cache_c1["features"][s["c1_idxs"]]        # (W, 8)
        c1 = (c1 - self.c1_mean) / self.c1_std              # z-score normalize
        h = np.concatenate([emb, c1], axis=1)               # (W, 1288)
        return torch.from_numpy(h).float(), s["center_class"]


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


class TemporalTransformerC1(nn.Module):
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
        h = self.proj(x)
        h = self.pe(h)
        h = self.transformer(h)
        h = self.norm(h)
        return self.classifier(h[:, self.center_idx])


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    inv = 1.0 / counts
    inv = inv * (n_classes / inv.sum())
    return torch.from_numpy(inv).float()


def per_class_auc(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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


def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
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


def train_one_seed(seed: int, video_index: Dict, c1_cache: Dict,
                    epochs: int) -> Dict:
    print(f"\n{'='*60}\nSEED = {seed}\n{'='*60}")
    cache_path = EMB_DIR / f"seed{seed}_embeddings.npz"
    npz = np.load(cache_path)
    emb_cache = {
        "embeddings": np.array(npz["embeddings"]),
        "labels": np.array(npz["labels"]),
        "splits": np.array(npz["splits"]),
        "filenames": np.array(npz["filenames"]),
    }
    npz.close()
    print(f"[seed {seed}] embedding cache: {emb_cache['embeddings'].shape}")

    train_ds = CachedSequenceDatasetC1(emb_cache, c1_cache, video_index, "train")
    val_ds = CachedSequenceDatasetC1(emb_cache, c1_cache, video_index, "val")
    test_ds = CachedSequenceDatasetC1(emb_cache, c1_cache, video_index, "test")
    print(f"[seed {seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_labels = np.array([s["center_class"] for s in train_ds.samples])
    cw = compute_class_weights(train_labels, N_CLASSES).to(DEVICE)

    torch.manual_seed(seed)
    model = TemporalTransformerC1().to(DEVICE)
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
            x = x.to(DEVICE); y = y.to(DEVICE)
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
                print(f"[seed {seed}] early stop at epoch {epoch}")
                break

    ckpt = torch.load(out_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_pc, test_macro, test_logits, test_labels = evaluate(model, test_loader)
    print(f"[seed {seed}] best val macro-AUC = {best_val_macro:.4f} "
          f"(epoch {best_epoch})")
    print(f"[seed {seed}] TEST macro-AUC = {test_macro:.4f}")
    for c, a in test_pc.items():
        if not np.isnan(a):
            print(f"           {c:25s}: {a:.4f}")

    np.savez_compressed(out_dir / "test_predictions.npz",
                        logits=test_logits, labels=test_labels)

    return {
        "seed": seed,
        "best_val_macro_auc": best_val_macro,
        "best_epoch": best_epoch,
        "test_macro_auc": test_macro,
        "test_per_class": test_pc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    seeds = args.only_seeds or SEEDS
    print(f"[main] cell (c) on {seeds}; in_dim={EMB_DIM_IN} (1280 emb + 8 C1)")

    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    c1_npz = np.load(C1_PATH)
    c1_cache = {
        "filenames": np.array(c1_npz["filenames"]),
        "features": np.array(c1_npz["features"]),
        "labels": np.array(c1_npz["labels"]),
        "splits": np.array(c1_npz["splits"]),
    }
    c1_npz.close()
    print(f"[main] c1 cache: {c1_cache['features'].shape}")

    results: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        r = train_one_seed(seed, video_index, c1_cache, epochs=args.epochs)
        results.append(r)
        (REPORT_DIR / "track_b_cell_c_predictions.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    macros = np.array([r["test_macro_auc"] for r in results])
    cs_mean = float(np.mean(macros))
    cs_std = float(np.std(macros))
    print(f"\n[main] CROSS-SEED test macro-AUC = {cs_mean:.4f} +- {cs_std:.4f}")
    cell_a_ref = 0.7598
    cell_b_ref = 0.7788
    delta_a = cs_mean - cell_a_ref
    delta_b = cs_mean - cell_b_ref
    print(f"[main] vs cell (a) baseline: {delta_a:+.4f}")
    print(f"[main] vs cell (b) temporal: {delta_b:+.4f}")

    pc_agg: Dict[str, Dict[str, float]] = {}
    for cname in CLASS_NAMES:
        vals = np.array([r["test_per_class"].get(cname, float("nan"))
                          for r in results])
        if not np.isnan(vals).all():
            pc_agg[cname] = {"mean": float(np.nanmean(vals)),
                              "std": float(np.nanstd(vals))}

    md = []
    md.append("# Track B Month 2: cell (c) — temporal + C1 structural prior\n")
    md.append("**Date:** 2026-05-07")
    md.append(f"**Seeds:** {[r['seed'] for r in results]}")
    md.append("**Model:** transformer (4×8×256, window=16) over "
              "concat[1280-d backbone + 8-d C1 structural prior z-scored]")
    md.append("")
    md.append("## Per-seed test macro-AUC\n")
    md.append("| Seed | Cell (b) | Cell (c) | Δ |")
    md.append("|---:|---:|---:|---:|")
    cell_b_per_seed = {41: 0.7658, 42: 0.7957, 43: 0.7618,
                        44: 0.7875, 45: 0.7810, 47: 0.7810}
    for r in results:
        b = cell_b_per_seed.get(r["seed"], float("nan"))
        d = r["test_macro_auc"] - b
        md.append(f"| {r['seed']} | {b:.4f} | {r['test_macro_auc']:.4f} "
                  f"| {d:+.4f} |")
    md.append("")
    md.append(f"**Cross-seed cell (c):** {cs_mean:.4f} ± {cs_std:.4f}")
    md.append(f"**Cell (a) reference:** 0.7598 (Δ {delta_a:+.4f})")
    md.append(f"**Cell (b) reference:** 0.7788 (Δ {delta_b:+.4f})")
    md.append("")
    md.append("## Per-class cross-seed test AUC vs cell (b)\n")
    cell_b_pc = {
        "Angiectasia": (0.813, 0.103), "Blood - fresh": (0.591, 0.162),
        "Erosion": (0.776, 0.107), "Erythema": (0.949, 0.033),
        "Foreign Body": (0.993, 0.005), "Ileocecal valve": (0.809, 0.031),
        "Lymphangiectasia": (0.314, 0.088), "Normal clean mucosa": (0.856, 0.025),
        "Pylorus": (0.900, 0.021), "Reduced Mucosal View": (0.583, 0.107),
        "Ulcer": (0.983, 0.010),
    }
    md.append("| Class | Cell (b) | Cell (c) | Δ |")
    md.append("|---|---:|---:|---:|")
    for c in CLASS_NAMES:
        if c not in pc_agg or c not in cell_b_pc:
            continue
        a = pc_agg[c]
        bm, bs = cell_b_pc[c]
        d = a["mean"] - bm
        md.append(f"| {c} | {bm:.3f} ± {bs:.3f} "
                  f"| {a['mean']:.3f} ± {a['std']:.3f} | {d:+.3f} |")
    md.append("")
    md.append("## Verdict\n")
    if delta_b >= 0.005:
        md.append(f"**STRONG PASS.** Cell (c) lifts {delta_b:+.4f} above "
                  f"cell (b). C1 adds genuine information beyond temporal.")
    elif delta_b >= 0.0:
        md.append(f"**MARGINAL POSITIVE.** Cell (c) lift {delta_b:+.4f} "
                  f"is positive but small. Per-class table shows whether "
                  f"the lift concentrates on the expected vascular classes "
                  f"(Lymph, Erosion, RMV).")
    else:
        md.append(f"**REGRESSION.** Cell (c) lost {-delta_b:.4f} relative to "
                  f"cell (b). C1 may be redundant with what the transformer "
                  f"already extracts; or the C1 features are too coarse to "
                  f"add value.")
    md.append(f"\nTotal compute: {(time.time() - t0)/60:.1f} min")

    out_path = REPORT_DIR / "track_b_cell_c_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out_path}")


if __name__ == "__main__":
    main()
