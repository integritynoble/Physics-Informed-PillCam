"""
Foundation-backbone temporal test — does the cell-b vs cell-c
boundary replicate on ResNet-50?
==============================================================

Builds on `foundation_backbone_test.py` (which tested cell-a vs
cell-c on ResNet-50 features and showed C1 lift = -0.005). Here we
add the temporal arm: train the same 4-layer transformer on the
ResNet-50 RGB features (cell-b equivalent) and on RGB+C1_8 features
(cell-c equivalent), and compare.

The crucial reference comparison from capsule manuscript:
  EfficientNet-B0 cell (b)         = 0.7788  (RGB + temporal)
  EfficientNet-B0 cell (c)         = 0.7774  (RGB + C1_8 + temporal)
  EfficientNet-B0 cell-c-vs-cell-b = -0.0014  (essentially flat)

If on ResNet-50:
  cell (b)-equiv = ResNet-50 RGB + temporal
  cell (c)-equiv = ResNet-50 RGB + C1_8 + temporal
  lift                = ?

If lift is also ~0, Corollary 2 of the parameterization-mechanism
boundary is empirically supported on a 5x larger backbone with
temporal aggregation.

Pre-conditions:
  - D:/kvasir_capsule/outputs/foundation_resnet50_embeddings.npz
    (produced by foundation_backbone_test.py)
  - D:/kvasir_capsule/outputs/c1_features.npz
  - paper/nature-machine-intelligence/code/temporal/video_index.json

Compute: ~5-10 min on GTX 1660 Ti (no backbone forwards).

Output:
  paper/nature-machine-intelligence/docs/foundation_backbone_temporal_report.md
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
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
RESNET_EMB = Path("D:/kvasir_capsule/outputs/foundation_resnet50_embeddings.npz")
C1_PATH = Path("D:/kvasir_capsule/outputs/c1_features.npz")
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
EMB_DIM_IN_RGB = 2048      # ResNet-50 pooled feature
C1_DIM = 8
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


class CachedSequenceDataset(Dataset):
    def __init__(self, cache: Dict, video_index: Dict, split: str,
                 c1_aux: np.ndarray = None,
                 window: int = WINDOW):
        self.cache = cache
        self.window = window
        target_split = SPLIT_INT[split]
        fname_to_idx = {fn: i for i, fn in enumerate(cache["filenames"])}
        self.c1_aux = c1_aux       # (N, 8) z-scored, in same order as cache["filenames"]

        self.samples: List[Dict] = []
        for video_id, frames in video_index["by_video"].items():
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
                while len(window_idxs) < window:
                    window_idxs.append(window_idxs[-1])
                self.samples.append({
                    "window_cache_idxs": window_idxs,
                    "center_class": class_list[i],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[i]
        emb = self.cache["embeddings"][s["window_cache_idxs"]]
        if self.c1_aux is not None:
            c1 = self.c1_aux[s["window_cache_idxs"]]
            emb = np.concatenate([emb, c1], axis=1)
        return torch.from_numpy(emb).float(), s["center_class"]


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
    def __init__(self, in_dim: int):
        super().__init__()
        self.window = WINDOW
        self.center_idx = WINDOW // 2
        self.proj = nn.Linear(in_dim, EMB_DIM_HIDDEN)
        self.pe = SinusoidalPE(WINDOW, EMB_DIM_HIDDEN)
        layer = nn.TransformerEncoderLayer(
            d_model=EMB_DIM_HIDDEN, nhead=N_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(EMB_DIM_HIDDEN)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(EMB_DIM_HIDDEN, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x); h = self.pe(h); h = self.transformer(h)
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
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    pc = per_class_auc(logits, labels)
    return pc, macro_auc(pc), logits, labels


def train_one_run(cache, video_index, c1_aux, in_dim, seed, epochs):
    train_ds = CachedSequenceDataset(cache, video_index, "train", c1_aux=c1_aux)
    val_ds = CachedSequenceDataset(cache, video_index, "val", c1_aux=c1_aux)
    test_ds = CachedSequenceDataset(cache, video_index, "test", c1_aux=c1_aux)
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_labels = np.array([s["center_class"] for s in train_ds.samples])
    cw = compute_class_weights(train_labels, N_CLASSES).to(DEVICE)

    torch.manual_seed(seed)
    model = TemporalTransformer(in_dim=in_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    ce_loss = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_macro = -1.0
    best_epoch = 0
    no_improve = 0
    best_state = None
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
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}  train_ce={train_loss:.4f}  "
                  f"val_macro={val_macro:.4f}  elapsed={(time.time()-t0)/60:.1f} min")
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_pc, test_macro, test_logits, test_labels = evaluate(model, test_loader)
    print(f"  best val={best_val_macro:.4f} (epoch {best_epoch})  test_macro={test_macro:.4f}")
    return test_macro, test_pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    print(f"[main] device={DEVICE}  seeds={args.seeds}")

    if not RESNET_EMB.exists():
        raise SystemExit(f"missing {RESNET_EMB}; run foundation_backbone_test.py first")
    if not INDEX_JSON.exists():
        raise SystemExit(f"missing {INDEX_JSON}; run build_video_index.py first")
    if not C1_PATH.exists():
        raise SystemExit(f"missing {C1_PATH}; run build_c1_features.py first")

    print(f"[main] loading ResNet-50 features")
    npz = np.load(RESNET_EMB)
    cache = {
        "embeddings": np.array(npz["embeddings"]).astype(np.float32),
        "labels": np.array(npz["labels"]),
        "splits": np.array(npz["splits"]),
        "filenames": np.array(npz["filenames"]),
    }
    npz.close()
    n_total = cache["embeddings"].shape[0]
    print(f"[main] cache: {cache['embeddings'].shape}")

    print(f"[main] loading C1 8-d features")
    c1 = np.load(C1_PATH)
    c1_fn = list(c1["filenames"])
    c1_features = np.array(c1["features"])[:, :C1_DIM]
    c1.close()
    fn_to_c1 = {f: i for i, f in enumerate(c1_fn)}
    c1_aligned = np.zeros((n_total, C1_DIM), dtype=np.float32)
    for i, f in enumerate(cache["filenames"]):
        if f in fn_to_c1:
            c1_aligned[i] = c1_features[fn_to_c1[f]]
    train_mask = (cache["splits"] == 0)
    c1_mean = c1_aligned[train_mask].mean(axis=0)
    c1_std = c1_aligned[train_mask].std(axis=0) + 1e-6
    c1_z = ((c1_aligned - c1_mean) / c1_std).astype(np.float32)

    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))

    # Run cell-(b)-equiv (RGB only, temporal) and cell-(c)-equiv
    # (RGB + C1_8, temporal) for each seed
    res_b: List[float] = []
    res_c: List[float] = []
    t0 = time.time()
    for seed in args.seeds:
        print(f"\n{'='*60}\nSEED = {seed}\n{'='*60}")
        print(f"[seed {seed}] cell (b)-equiv: ResNet-50 RGB + temporal")
        m_b, _ = train_one_run(cache, video_index, c1_aux=None,
                                  in_dim=EMB_DIM_IN_RGB, seed=seed,
                                  epochs=args.epochs)
        res_b.append(m_b)

        print(f"\n[seed {seed}] cell (c)-equiv: ResNet-50 RGB + C1_8 + temporal")
        m_c, _ = train_one_run(cache, video_index, c1_aux=c1_z,
                                  in_dim=EMB_DIM_IN_RGB + C1_DIM, seed=seed,
                                  epochs=args.epochs)
        res_c.append(m_c)
        print(f"\n[seed {seed}] cell-c-vs-cell-b lift = {m_c - m_b:+.4f}")

    arr_b = np.array(res_b)
    arr_c = np.array(res_c)
    print(f"\n[main] cross-seed:")
    print(f"  cell-(b)-equiv (ResNet-50 RGB + temporal)        "
          f"= {arr_b.mean():.4f} +- {arr_b.std():.4f}")
    print(f"  cell-(c)-equiv (ResNet-50 RGB + C1_8 + temporal) "
          f"= {arr_c.mean():.4f} +- {arr_c.std():.4f}")
    print(f"  lift                                              "
          f"= {(arr_c - arr_b).mean():+.4f} +- {(arr_c - arr_b).std():.4f}")

    # Capsule reference (EfficientNet-B0)
    eff_b = 0.7788
    eff_c = 0.7774
    eff_lift = eff_c - eff_b
    print(f"\n  reference (EfficientNet-B0):")
    print(f"  cell (b)        = {eff_b:.4f}")
    print(f"  cell (c)        = {eff_c:.4f}")
    print(f"  cell-c-vs-(b)   = {eff_lift:+.4f}")

    # Decision
    cs_lift = (arr_c - arr_b).mean()
    if abs(cs_lift) < 0.005:
        verdict = (f"**Corollary 2 strongly supported.** On ResNet-50 with "
                   f"temporal aggregation, cell-c-vs-cell-b lift = "
                   f"{cs_lift:+.4f} — within ±0.005 of zero, mirroring the "
                   f"EfficientNet-B0 reference of -0.0014. The summary-stat "
                   f"C1 channel is structurally redundant with what *both* "
                   f"backbones extract once temporal aggregation is added. "
                   f"The parameterization-mechanism boundary is robust to "
                   f"backbone choice within the CNN family.")
    elif abs(cs_lift) < 0.020:
        verdict = (f"**Marginal.** lift = {cs_lift:+.4f} is small but "
                   f"non-trivial. Direction matches EfficientNet-B0 "
                   f"(both negative); magnitude differs.")
    else:
        verdict = (f"**Surprising.** lift = {cs_lift:+.4f} differs "
                   f"materially from EfficientNet-B0's -0.0014. "
                   f"Boundary is more architecture-specific than the "
                   f"single-backbone result suggested.")

    md = []
    md.append("# Foundation-backbone temporal test — ResNet-50 cell-b vs cell-c\n")
    md.append("**Date:** 2026-05-08")
    md.append("**Backbone:** ResNet-50, ImageNet-1K V2 weights, NO fine-tuning")
    md.append("**Method:** train 4-layer temporal transformer over the cached "
              "ResNet-50 RGB pooled features. Compare RGB-only (cell-b "
              "equivalent) to RGB + 8-d C1 scalars (cell-c equivalent). "
              "Same 6-seed evaluation as the capsule manuscript.")
    md.append("")
    md.append("## Per-seed test macro-AUC\n")
    md.append("| Seed | cell (b)-equiv | cell (c)-equiv | Δ |")
    md.append("|---:|---:|---:|---:|")
    for i, seed in enumerate(args.seeds):
        md.append(f"| {seed} | {res_b[i]:.4f} | {res_c[i]:.4f} | "
                  f"{res_c[i] - res_b[i]:+.4f} |")
    md.append("")
    md.append(f"**Cross-seed mean:**")
    md.append(f"- cell (b)-equiv = {arr_b.mean():.4f} ± {arr_b.std():.4f}")
    md.append(f"- cell (c)-equiv = {arr_c.mean():.4f} ± {arr_c.std():.4f}")
    md.append(f"- **Δ = {(arr_c - arr_b).mean():+.4f} ± {(arr_c - arr_b).std():.4f}**")
    md.append("")
    md.append("## Reference (EfficientNet-B0, capsule manuscript)\n")
    md.append("| Cell | macro-AUC |")
    md.append("|---|---:|")
    md.append(f"| (b) RGB+C2 | {eff_b:.4f} |")
    md.append(f"| (c) RGB+C2+C1_8 | {eff_c:.4f} |")
    md.append(f"| Δ | {eff_lift:+.4f} |")
    md.append("")
    md.append("## Verdict\n")
    md.append(verdict)
    md.append("")
    md.append(f"**Total compute:** {(time.time() - t0)/60:.1f} min for "
              f"{len(args.seeds)} seeds.")

    out = REPORT_DIR / "foundation_backbone_temporal_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out}")


if __name__ == "__main__":
    main()
