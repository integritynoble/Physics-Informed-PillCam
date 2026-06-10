"""
Cell (e+) ACDC — temporal + C3 over +PI 5-channel embeddings.
==============================================================

ACDC analogue of `train_cell_e_pi.py` and the final architecture in
the cardiac-MRI replication. Per-frame input is the concatenation of:
    - 1280-d +PI 5-ch backbone embedding
    - 192-d C3 autoencoder-residual feature

If both C2 (temporal) AND C1-spatial (+PI 5-ch) AND C3 stack
positively on ACDC the same way they did on capsule, the framework
generalizes across modalities and the NMI submission is justified.

Pre-conditions:
    - build_embedding_cache_pi_acdc.py has produced embeddings_pi/seed{seed}_embeddings.npz
    - build_c3_features_acdc.py has produced c3_features.npz
    - video_index_acdc.json exists

Output:
    D:/acdc/outputs/temporal_cell_e_pi/seed{seed}/best_model.pt
    paper/nature-machine-intelligence/docs/track_b_acdc_cell_e_pi_report.md
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
EMB_DIR = Path("D:/acdc/outputs/embeddings_pi")
C3_PATH = Path("D:/acdc/outputs/c3_features.npz")
OUTPUT_ROOT = Path("D:/acdc/outputs/temporal_cell_e_pi")
INDEX_JSON = HERE / "video_index_acdc.json"
REPORT_DIR = HERE.parent.parent / "docs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [41, 42, 43, 44, 45, 47]

CLASS_NAMES = ["NOR", "MINF", "DCM", "HCM", "RV"]
N_CLASSES = len(CLASS_NAMES)
SPLIT_INT = {"train": 0, "val": 1, "test": 2}

WINDOW = 16
EMB_DIM_RGB = 1280
C3_DIM = 192
EMB_DIM_IN = EMB_DIM_RGB + C3_DIM
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


class CachedSequenceDatasetE(Dataset):
    def __init__(self, cache_emb: Dict, cache_c3: Dict,
                 video_index: Dict, split: str, window: int = WINDOW):
        self.cache_emb = cache_emb
        self.cache_c3 = cache_c3
        self.window = window
        target_split = SPLIT_INT[split]

        emb_fn_to_idx = {fn: i for i, fn in enumerate(cache_emb["filenames"])}
        c3_fn_to_idx = {fn: i for i, fn in enumerate(cache_c3["filenames"])}

        train_mask = cache_c3["splits"] == 0
        self.c3_mean = cache_c3["features"][train_mask].mean(axis=0)
        self.c3_std = cache_c3["features"][train_mask].std(axis=0) + 1e-6

        self.samples: List[Dict] = []
        for video_id, frames in video_index["by_video"].items():
            in_split = [f for f in frames
                         if SPLIT_INT.get(f["split"], -1) == target_split]
            if not in_split:
                continue
            in_split.sort(key=lambda f: f["frame_number"])
            emb_idxs, c3_idxs, class_list = [], [], []
            for f in in_split:
                fn = f["filename"]
                if fn in emb_fn_to_idx and fn in c3_fn_to_idx:
                    emb_idxs.append(emb_fn_to_idx[fn])
                    c3_idxs.append(c3_fn_to_idx[fn])
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
                w_c3 = c3_idxs[lo:hi]
                while len(w_emb) < window:
                    w_emb.append(w_emb[-1])
                    w_c3.append(w_c3[-1])
                self.samples.append({
                    "emb_idxs": w_emb, "c3_idxs": w_c3,
                    "center_class": class_list[i],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[i]
        emb = self.cache_emb["embeddings"][s["emb_idxs"]]
        c3 = self.cache_c3["features"][s["c3_idxs"]]
        c3 = (c3 - self.c3_mean) / self.c3_std
        h = np.concatenate([emb, c3], axis=1)
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


class TemporalTransformerE(nn.Module):
    def __init__(self):
        super().__init__()
        self.window = WINDOW
        self.center_idx = WINDOW // 2
        self.proj = nn.Linear(EMB_DIM_IN, EMB_DIM_HIDDEN)
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


def train_one_seed(seed: int, video_index: Dict, c3_cache: Dict,
                    epochs: int) -> Dict:
    print(f"\n{'='*60}\nSEED = {seed} (cell e+ ACDC)\n{'='*60}")
    npz = np.load(EMB_DIR / f"seed{seed}_embeddings.npz")
    emb_cache = {
        "embeddings": np.array(npz["embeddings"]),
        "labels": np.array(npz["labels"]),
        "splits": np.array(npz["splits"]),
        "filenames": np.array(npz["filenames"]),
    }
    npz.close()

    train_ds = CachedSequenceDatasetE(emb_cache, c3_cache, video_index, "train")
    val_ds = CachedSequenceDatasetE(emb_cache, c3_cache, video_index, "val")
    test_ds = CachedSequenceDatasetE(emb_cache, c3_cache, video_index, "test")
    print(f"[seed {seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_labels = np.array([s["center_class"] for s in train_ds.samples])
    cw = compute_class_weights(train_labels, N_CLASSES).to(DEVICE)

    torch.manual_seed(seed)
    model = TemporalTransformerE().to(DEVICE)
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
    history = []
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

    return {"seed": seed,
            "best_val_macro_auc": best_val_macro,
            "best_epoch": best_epoch,
            "test_macro_auc": test_macro,
            "test_per_class": test_pc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    seeds = args.only_seeds or SEEDS
    print(f"[main] cell (e+) ACDC on {seeds}; "
          f"in_dim={EMB_DIM_IN} (1280 +PI emb + 192 C3)")

    if not INDEX_JSON.exists():
        raise SystemExit(f"Missing {INDEX_JSON}; run build_video_index_acdc.py first.")
    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))

    if not C3_PATH.exists():
        raise SystemExit(f"Missing {C3_PATH}; run build_c3_features_acdc.py first.")
    c3_npz = np.load(C3_PATH)
    c3_cache = {k: np.array(c3_npz[k]) for k in
                  ("filenames", "features", "labels", "splits")}
    c3_npz.close()
    print(f"[main] c3: {c3_cache['features'].shape}")

    results: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        r = train_one_seed(seed, video_index, c3_cache, epochs=args.epochs)
        results.append(r)
        (REPORT_DIR / "track_b_acdc_cell_e_pi_predictions.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    macros = np.array([r["test_macro_auc"] for r in results])
    cs_mean = float(np.mean(macros))
    cs_std = float(np.std(macros))
    print(f"\n[main] CROSS-SEED test macro-AUC = {cs_mean:.4f} +- {cs_std:.4f}")

    pc_agg: Dict[str, Dict[str, float]] = {}
    for cname in CLASS_NAMES:
        vals = np.array([r["test_per_class"].get(cname, float("nan"))
                          for r in results])
        if not np.isnan(vals).all():
            pc_agg[cname] = {"mean": float(np.nanmean(vals)),
                              "std": float(np.nanstd(vals))}

    md = []
    md.append("# Track B (ACDC): cell (e+) — temporal + C3 over +PI 5-ch embeddings\n")
    md.append("**Date:** TBD")
    md.append(f"**Seeds:** {[r['seed'] for r in results]}")
    md.append("**Source:** D:/acdc/outputs/embeddings_pi/ + D:/acdc/outputs/c3_features.npz")
    md.append("")
    md.append("## Per-seed test macro-AUC\n")
    md.append("| Seed | Test macro-AUC | Best val macro-AUC |")
    md.append("|---:|---:|---:|")
    for r in results:
        md.append(f"| {r['seed']} | {r['test_macro_auc']:.4f} "
                  f"| {r['best_val_macro_auc']:.4f} |")
    md.append("")
    md.append(f"**Cross-seed mean:** {cs_mean:.4f} ± {cs_std:.4f}")
    md.append("")
    md.append("## Per-class cross-seed AUC\n")
    md.append("| Class | Mean ± σ |")
    md.append("|---|---:|")
    for c in CLASS_NAMES:
        if c in pc_agg:
            a = pc_agg[c]
            md.append(f"| {c} | {a['mean']:.3f} ± {a['std']:.3f} |")

    out_path = REPORT_DIR / "track_b_acdc_cell_e_pi_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out_path}")
    print(f"[main] elapsed: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
