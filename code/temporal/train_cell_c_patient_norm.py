"""
Cell (c') — temporal + C1 with PATIENT-LEVEL normalization
============================================================

Same architecture and training recipe as `train_cell_c.py`, but C1
features are z-scored PER PATIENT (using only that patient's frames)
rather than per train-split. Directly addresses the val→test
patient-shift failure mode documented for cell (c) on seed 42
(-0.057 collapse).

If patient-normalized C1 ≥ cell (b) (0.7788), the three-channel
framework story revives with empirical support. If still null,
the C1 failure is structural (redundancy + lossiness, not just
patient-shift), and the cell-(e) two-channel architecture stays.

Patient ID derivation: Kvasir-Capsule filenames are
`<video_id>_<frame_number>.jpg`; `video_id` IS the patient. Since
the dataset uses a video-level split, every patient is in exactly
one split. Per-patient z-score statistics are computed over ALL of
the patient's frames (no label use), so this is principled
unsupervised normalization, not test-set leakage.

Outputs:
  D:/kvasir_capsule/outputs/temporal_cell_c_patient_norm/seed{seed}/best_model.pt
  paper/nature-machine-intelligence/docs/track_b_cell_c_patient_norm_report.md
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
OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs/temporal_cell_c_patient_norm")
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


def patient_id_from_filename(fn: str) -> str:
    """Kvasir-Capsule filename: '<video_id>_<frame_number>.jpg'.
    video_id IS the patient. Returns the video_id string."""
    base = fn.rsplit(".", 1)[0]
    return base.rsplit("_", 1)[0]


def compute_patient_norm_stats(c1_cache: Dict
                                  ) -> Tuple[Dict[str, np.ndarray],
                                             Dict[str, np.ndarray]]:
    """For every unique patient in c1_cache, compute per-feature
    mean and std over ALL of that patient's frames.

    Returns (means, stds) where each is {patient_id: (8,) array}.
    No label use; purely unsupervised pixel-derived statistics.
    """
    fnames = c1_cache["filenames"]
    feats = c1_cache["features"]  # (N, 8)
    pids = np.array([patient_id_from_filename(fn) for fn in fnames])
    means: Dict[str, np.ndarray] = {}
    stds: Dict[str, np.ndarray] = {}
    for pid in np.unique(pids):
        mask = pids == pid
        means[pid] = feats[mask].mean(axis=0)
        stds[pid] = feats[mask].std(axis=0) + 1e-6
    return means, stds


class CachedSequenceDatasetC1Patient(Dataset):
    """Cell (c') — same as cell (c) but C1 features are z-scored
    per patient instead of per train split."""

    def __init__(self, cache_emb: Dict, cache_c1: Dict, video_index: Dict,
                 split: str, patient_means: Dict, patient_stds: Dict,
                 window: int = WINDOW):
        self.cache_emb = cache_emb
        self.cache_c1 = cache_c1
        self.window = window
        target_split = SPLIT_INT[split]

        emb_fn_to_idx = {fn: i for i, fn in enumerate(cache_emb["filenames"])}
        c1_fn_to_idx = {fn: i for i, fn in enumerate(cache_c1["filenames"])}

        # Pre-z-score the entire C1 feature array using each frame's
        # patient-level stats. Done once at construction.
        c1_feats = cache_c1["features"]
        c1_norm = np.zeros_like(c1_feats)
        for i, fn in enumerate(cache_c1["filenames"]):
            pid = patient_id_from_filename(fn)
            c1_norm[i] = (c1_feats[i] - patient_means[pid]) / patient_stds[pid]
        self.c1_norm = c1_norm.astype(np.float32)

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
                    "emb_idxs": w_emb, "c1_idxs": w_c1,
                    "center_class": class_list[i],
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[i]
        emb = self.cache_emb["embeddings"][s["emb_idxs"]]
        c1 = self.c1_norm[s["c1_idxs"]]   # already patient-normalized
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
                    patient_means: Dict, patient_stds: Dict,
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

    train_ds = CachedSequenceDatasetC1Patient(
        emb_cache, c1_cache, video_index, "train",
        patient_means, patient_stds)
    val_ds = CachedSequenceDatasetC1Patient(
        emb_cache, c1_cache, video_index, "val",
        patient_means, patient_stds)
    test_ds = CachedSequenceDatasetC1Patient(
        emb_cache, c1_cache, video_index, "test",
        patient_means, patient_stds)
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
    print(f"[main] cell (c') patient-normalized C1 on {seeds}")

    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    c1_npz = np.load(C1_PATH)
    c1_cache = {k: np.array(c1_npz[k]) for k in
                  ("filenames", "features", "labels", "splits")}
    c1_npz.close()
    print(f"[main] c1 cache: {c1_cache['features'].shape}")

    patient_means, patient_stds = compute_patient_norm_stats(c1_cache)
    print(f"[main] patient-norm stats computed for {len(patient_means)} patients")

    results: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        r = train_one_seed(seed, video_index, c1_cache,
                            patient_means, patient_stds, epochs=args.epochs)
        results.append(r)
        (REPORT_DIR / "track_b_cell_c_patient_norm_predictions.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    macros = np.array([r["test_macro_auc"] for r in results])
    cs_mean = float(np.mean(macros))
    cs_std = float(np.std(macros))
    print(f"\n[main] CROSS-SEED test macro-AUC = {cs_mean:.4f} +- {cs_std:.4f}")
    cell_a, cell_b, cell_c, cell_e = 0.7598, 0.7788, 0.7774, 0.7828
    print(f"[main] vs cell (a)  baseline 0.7598: {cs_mean - cell_a:+.4f}")
    print(f"[main] vs cell (b)  C2-only  0.7788: {cs_mean - cell_b:+.4f}")
    print(f"[main] vs cell (c)  C2+C1    0.7774: {cs_mean - cell_c:+.4f}  (the cell this fixes)")
    print(f"[main] vs cell (e)  C2+C3    0.7828: {cs_mean - cell_e:+.4f}")

    pc_agg: Dict[str, Dict[str, float]] = {}
    for cname in CLASS_NAMES:
        vals = np.array([r["test_per_class"].get(cname, float("nan"))
                          for r in results])
        if not np.isnan(vals).all():
            pc_agg[cname] = {"mean": float(np.nanmean(vals)),
                              "std": float(np.nanstd(vals))}

    md = []
    md.append("# Cell (c') — temporal + C1 with PATIENT-LEVEL z-score normalization\n")
    md.append("**Date:** 2026-05-07")
    md.append(f"**Seeds:** {[r['seed'] for r in results]}")
    md.append("**Hypothesis tested:** C1 fails (cell c, Δ macro flat, σ +60%) due to")
    md.append("val→test patient shift. Z-scoring C1 per patient (using only that")
    md.append("patient's pixel statistics, no label use) directly addresses the")
    md.append("documented failure mode.\n")
    md.append("## Per-seed test macro-AUC\n")
    md.append("| Seed | Cell (b) | Cell (c) train-norm | Cell (c') patient-norm | Δ vs (b) | Δ vs (c) |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    b_per_seed = {41: 0.7658, 42: 0.7957, 43: 0.7618, 44: 0.7875, 45: 0.7810, 47: 0.7810}
    c_per_seed = {41: 0.7750, 42: 0.7386, 43: 0.7736, 44: 0.7899, 45: 0.7899, 47: 0.7976}
    for r in results:
        s = r["seed"]
        cp = r["test_macro_auc"]
        b = b_per_seed.get(s, float("nan"))
        c = c_per_seed.get(s, float("nan"))
        md.append(f"| {s} | {b:.4f} | {c:.4f} | {cp:.4f} "
                  f"| {cp - b:+.4f} | {cp - c:+.4f} |")
    md.append("")
    md.append(f"**Cross-seed cell (c'):** {cs_mean:.4f} ± {cs_std:.4f}")
    md.append(f"vs cell (a): {cs_mean - cell_a:+.4f}")
    md.append(f"vs cell (b): {cs_mean - cell_b:+.4f}")
    md.append(f"vs cell (c) train-norm: {cs_mean - cell_c:+.4f}")
    md.append(f"vs cell (e) C2+C3: {cs_mean - cell_e:+.4f}")
    md.append("")
    md.append("## Verdict\n")
    delta_b = cs_mean - cell_b
    delta_c = cs_mean - cell_c
    if delta_b >= 0.005:
        md.append(f"**STRONG PASS.** Patient-norm C1 lifts {delta_b:+.4f} vs cell (b). "
                  f"The C1 failure was indeed val→test patient shift; per-patient "
                  f"normalization fixes it. Three-channel framework (C2+C1_patient+C3) "
                  f"is the new architecture candidate.")
    elif delta_b >= 0.0:
        md.append(f"**MARGINAL POSITIVE.** Δ vs cell (b) = {delta_b:+.4f}. C1 is no "
                  f"longer hurting, and per-patient normalization recovered some "
                  f"signal. Worth including in three-channel framework if cell (e) "
                  f"is also kept.")
    elif delta_c >= 0.005:
        md.append(f"**PARTIAL FIX.** C1 patient-norm beats train-split-norm "
                  f"({delta_c:+.4f}) but still loses to cell (b). Patient shift was a "
                  f"real factor but not the only one — redundancy with backbone "
                  f"(structural failure 3.2) and lossy summarization (3.3) also "
                  f"contribute. Drop C1 from architecture; report as ablation.")
    else:
        md.append(f"**FAIL.** Patient-norm did not recover. The C1 failure is "
                  f"structural, not just patient-shift. Keep cell (e) two-channel "
                  f"as the architecture; report C1 ablation as a clean negative "
                  f"result.")
    md.append(f"\nTotal compute: {(time.time() - t0)/60:.1f} min")

    out_path = REPORT_DIR / "track_b_cell_c_patient_norm_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out_path}")


if __name__ == "__main__":
    main()
