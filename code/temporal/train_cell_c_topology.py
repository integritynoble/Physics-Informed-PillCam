"""
Cell (c″) — temporal + C1 with vessel-topology features (13d total)
=====================================================================

Same architecture / training recipe as `train_cell_c.py`, but feeds the
augmented 13-dim C1 vector (8 scalar global summaries + 5 connected-
component topology features). Train-split z-scoring (the per-patient
variant from cell c′ already showed normalization is not the dominant
failure; we test the structural hypothesis here).

Hypothesis: cell (c) failed because 8-scalar global summaries are
redundant with backbone features and lossy. Topology features capture
spatial structure (blob count, area, centrality, aspect ratio, size
entropy) that the 8 scalars miss. If cell (c″) ≥ cell (b), the
"richer C1" claim is empirically supported.

Outputs:
  D:/kvasir_capsule/outputs/temporal_cell_c_topology/seed{seed}/best_model.pt
  paper/nature-machine-intelligence/docs/track_b_cell_c_topology_report.md
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
C1_PATH = Path("D:/kvasir_capsule/outputs/c1_topology_features.npz")
OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs/temporal_cell_c_topology")
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
C1_DIM = 13   # 8 scalar + 5 topology
EMB_DIM_IN = EMB_DIM_RGB + C1_DIM
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


class CachedSequenceDatasetC1Topo(Dataset):
    def __init__(self, cache_emb: Dict, cache_c1: Dict,
                 video_index: Dict, split: str, window: int = WINDOW):
        self.cache_emb = cache_emb
        self.cache_c1 = cache_c1
        self.window = window
        target_split = SPLIT_INT[split]

        emb_fn_to_idx = {fn: i for i, fn in enumerate(cache_emb["filenames"])}
        c1_fn_to_idx = {fn: i for i, fn in enumerate(cache_c1["filenames"])}

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
            emb_idxs, c1_idxs, class_list = [], [], []
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
                    "emb_idxs": w_emb, "c1_idxs": w_c1,
                    "center_class": class_list[i],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[i]
        emb = self.cache_emb["embeddings"][s["emb_idxs"]]
        c1 = self.cache_c1["features"][s["c1_idxs"]]
        c1 = (c1 - self.c1_mean) / self.c1_std
        h = np.concatenate([emb, c1], axis=1)
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


class TemporalTransformerC1Topo(nn.Module):
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


def train_one_seed(seed: int, video_index: Dict, c1_cache: Dict,
                    epochs: int) -> Dict:
    print(f"\n{'='*60}\nSEED = {seed}\n{'='*60}")
    npz = np.load(EMB_DIR / f"seed{seed}_embeddings.npz")
    emb_cache = {
        "embeddings": np.array(npz["embeddings"]),
        "labels": np.array(npz["labels"]),
        "splits": np.array(npz["splits"]),
        "filenames": np.array(npz["filenames"]),
    }
    npz.close()

    train_ds = CachedSequenceDatasetC1Topo(emb_cache, c1_cache, video_index, "train")
    val_ds = CachedSequenceDatasetC1Topo(emb_cache, c1_cache, video_index, "val")
    test_ds = CachedSequenceDatasetC1Topo(emb_cache, c1_cache, video_index, "test")
    print(f"[seed {seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_labels = np.array([s["center_class"] for s in train_ds.samples])
    cw = compute_class_weights(train_labels, N_CLASSES).to(DEVICE)

    torch.manual_seed(seed)
    model = TemporalTransformerC1Topo().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seed {seed}] trainable params: {n_params:,}  in_dim={EMB_DIM_IN}")

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
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val_macro_auc": val_macro,
                        "val_per_class": val_pc, "history": history},
                        out_dir / "best_model.pt")
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
    np.savez_compressed(out_dir / "test_predictions.npz",
                        logits=test_logits, labels=test_labels)

    return {"seed": seed, "best_val_macro_auc": best_val_macro,
            "best_epoch": best_epoch, "test_macro_auc": test_macro,
            "test_per_class": test_pc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    seeds = args.only_seeds or SEEDS
    print(f"[main] cell (c″) topology C1 on {seeds}; in_dim={EMB_DIM_IN}")

    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    c1_npz = np.load(C1_PATH)
    c1_cache = {k: np.array(c1_npz[k]) for k in
                  ("filenames", "features", "labels", "splits")}
    c1_npz.close()
    print(f"[main] c1 topology cache: {c1_cache['features'].shape}")

    results: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        r = train_one_seed(seed, video_index, c1_cache, epochs=args.epochs)
        results.append(r)
        (REPORT_DIR / "track_b_cell_c_topology_predictions.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    macros = np.array([r["test_macro_auc"] for r in results])
    cs_mean = float(np.mean(macros))
    cs_std = float(np.std(macros))
    print(f"\n[main] CROSS-SEED test macro-AUC = {cs_mean:.4f} +- {cs_std:.4f}")
    cell_a, cell_b, cell_c, cell_cp, cell_e = 0.7598, 0.7788, 0.7774, 0.7787, 0.7828
    print(f"[main] vs cell (a) baseline 0.7598: {cs_mean - cell_a:+.4f}")
    print(f"[main] vs cell (b) C2-only 0.7788: {cs_mean - cell_b:+.4f}")
    print(f"[main] vs cell (c) C2+C1 train-norm 0.7774: {cs_mean - cell_c:+.4f}")
    print(f"[main] vs cell (c′) C2+C1 patient-norm 0.7787: {cs_mean - cell_cp:+.4f}")
    print(f"[main] vs cell (e) C2+C3 0.7828: {cs_mean - cell_e:+.4f}")

    md = []
    md.append("# Cell (c″) — temporal + C1 with vessel-topology features (13d)\n")
    md.append("**Date:** 2026-05-07")
    md.append(f"**Seeds:** {[r['seed'] for r in results]}")
    md.append("**Hypothesis:** the cells (c)/(c′) failure was driven by lossy")
    md.append("8-scalar global summarization. Adding 5 connected-component")
    md.append("topology features (blob count, area, centrality, aspect ratio,")
    md.append("size entropy) preserves spatial structure that 8 scalars discard.")
    md.append("If cell (c″) ≥ cell (b), the richer-C1 hypothesis is empirically")
    md.append("supported.\n")
    md.append("## Per-seed test macro-AUC\n")
    md.append("| Seed | Cell (b) | Cell (c) train-norm | Cell (c′) patient-norm | Cell (c″) topology | Δ vs (b) | Δ vs (c) |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    b_per = {41: 0.7658, 42: 0.7957, 43: 0.7618, 44: 0.7875, 45: 0.7810, 47: 0.7810}
    c_per = {41: 0.7750, 42: 0.7386, 43: 0.7736, 44: 0.7899, 45: 0.7899, 47: 0.7976}
    cp_per = {41: 0.7765, 42: 0.7421, 43: 0.7670, 44: 0.7967, 45: 0.7915, 47: 0.7982}
    for r in results:
        s = r["seed"]
        ct = r["test_macro_auc"]
        b = b_per.get(s, float("nan"))
        c = c_per.get(s, float("nan"))
        cp = cp_per.get(s, float("nan"))
        md.append(f"| {s} | {b:.4f} | {c:.4f} | {cp:.4f} | {ct:.4f} "
                  f"| {ct - b:+.4f} | {ct - c:+.4f} |")
    md.append("")
    md.append(f"**Cross-seed cell (c″):** {cs_mean:.4f} ± {cs_std:.4f}")
    md.append(f"vs cell (b): {cs_mean - cell_b:+.4f}")
    md.append(f"vs cell (c): {cs_mean - cell_c:+.4f}")
    md.append(f"vs cell (c′): {cs_mean - cell_cp:+.4f}")
    md.append(f"vs cell (e): {cs_mean - cell_e:+.4f}")
    md.append("")
    md.append("## Verdict\n")
    delta_b = cs_mean - cell_b
    if delta_b >= 0.005:
        md.append(f"**STRONG PASS.** Topology C1 lifts {delta_b:+.4f} vs cell (b). "
                  f"The richer-feature hypothesis is supported. Three-channel "
                  f"framework (C2+C1_topology+C3) is the new architecture candidate.")
    elif delta_b >= 0.0:
        md.append(f"**MARGINAL POSITIVE.** Δ vs cell (b) = {delta_b:+.4f}. Topology "
                  f"features no longer hurt; the lossiness hypothesis (3.3 in the "
                  f"failure-mode doc) is partially supported. Worth combining with "
                  f"C3 in a four-channel C2+C1_topo+C3 ablation.")
    else:
        md.append(f"**FAIL.** Δ vs (b) = {delta_b:+.4f}. Topology features did not "
                  f"recover the C1 channel. Combined with cells (c) and (c′), "
                  f"this strongly suggests the C1 failure is dominated by "
                  f"redundancy with backbone features (3.2), not by lossy "
                  f"summarization (3.3). The two-channel C2+C3 architecture "
                  f"(cell e) remains the final framework.")
    md.append(f"\nTotal compute: {(time.time() - t0)/60:.1f} min")

    out_path = REPORT_DIR / "track_b_cell_c_topology_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out_path}")


if __name__ == "__main__":
    main()
