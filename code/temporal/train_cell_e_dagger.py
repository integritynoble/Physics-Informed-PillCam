"""
Cell (e†) — temporal + C3 with the distilled 3-channel RGB backbone
====================================================================

Headline deployment variant for the medIA submission. Defined in
section3_methods.tex (lines 32, 43-47): pairs the C2 (4-layer
transformer over 16-frame window) and C3 (Normal-class AE residual)
streams with C1 in its *distillation* form (branch C of fig 1):
a 3-channel RGB backbone trained with the joint loss
    L = L_CE + lambda * L_BCE(aux(g(x)), P_blood(x))
and the auxiliary decoder head dropped at inference. The deployed
network sees plain 3-channel RGB and carries hemoglobin-aware
features in its internal representations.

Per-frame input:
    concat[ e_s (1280-d distilled backbone), r_feat_s (192-d C3) ]
        = 1472-d

Reference numbers (from the eight-cell ablation, six seeds
{41, 42, 43, 44, 45, 47}, frozen scope per the project's seed plan):

    cell (a)  per-frame baseline                 - 0.7598
    cell (b)  C2 only                            - 0.7788  (+0.019)
    cell (c)  C2 + C1_8 (train-norm summary)     - 0.7774  (-0.001)
    cell (c') C2 + C1_8 (per-patient z)          - 0.7787  (+0.000)
    cell (c'')C2 + C1_13 (with topology)         - 0.7596  (-0.019)
    cell (d)  C2 + C1_8 + C3                     - 0.7722  (-0.007)
    cell (e)  C2 + C3 (NO C1)                    - 0.7828  (+0.023)
    cell (b+) C1_5ch + C2 (input fusion)         - 0.7901  (+0.030)
    cell (e+) C1_5ch + C2 + C3 (headline)        - 0.8038  (+0.044)
    cell (e†) C1_distill + C2 + C3 (this run)    - ?

Per-frame distillation alone (no C2, no C3) reaches 0.773 +- 0.028
(section4_results.tex tab:per-frame). Cell (e) without any C1 is
0.7828, already above that. The empirical question this script
answers: does adding distilled-C1 features to C2+C3 buy anything
above cell (e)'s no-C1 number, and how does the 3-channel deployment
variant compare with the 5-channel input-fusion cell (e+) at 0.8038?

Expected interpretive outcomes:
    cell (e†) >= cell (e)   distilled C1 contributes a real signal
                             at the spatial-feature level even
                             though the input is 3-channel RGB.
    cell (e†) ~  cell (e)   distillation does not help on top of
                             C2+C3; the per-frame distillation lift
                             is what the temporal transformer was
                             already extracting.
    cell (e†) ~  cell (e+)  3-channel deployment matches the
                             5-channel headline -- the deployment
                             story is fully recovered.

Frozen-backbone temporal protocol (matches cell (e)):
    - per-frame embeddings: extracted ONCE from each distillation
      checkpoint at stage2_distill_effb0[_seed{seed}]/best_model.pt
      with model.deploy_mode = True (auxiliary head dropped). Cached
      to D:/kvasir_capsule/outputs/embeddings_distill/seed{seed}_embeddings.npz
      (see PREREQUISITE below).
    - C3 features: reused from c3_features.npz; the Normal-class
      autoencoder is a per-image function and does not depend on the
      backbone variant.
    - sequence transformer + head: 4-layer 8-head, hidden 256, FFN
      512, sinusoidal PE, dropout 0.1, 16-frame window. Trained
      fresh per seed (frozen backbone, only transformer + head
      trainable). AdamW, lr 1e-4, wd 1e-4, cosine, early stop on
      val macro-AUC with patience 5.

PREREQUISITE:
    D:/kvasir_capsule/outputs/embeddings_distill/seed{41,42,43,44,45,47}
        _embeddings.npz
    is built by build_embedding_cache_distill.py (a one-line edit
    of build_embedding_cache.py that swaps output_dir_for() from
    stage2_rgb_effb0[_seed*] to stage2_distill_effb0[_seed*] and
    sets model.deploy_mode = True after load_state_dict). If the
    cache is missing this script fails fast with a clear message.

Compute: ~13 min on the GTX 1660 Ti for all 6 seeds (matches cell
(e); the only added cost is the one-time embedding extraction).

Usage:
    python train_cell_e_dagger.py                 # all 6 seeds
    python train_cell_e_dagger.py --only_seeds 42 # one seed
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
EMB_DIR = Path("D:/kvasir_capsule/outputs/embeddings_distill")
C3_PATH = Path("D:/kvasir_capsule/outputs/c3_features.npz")
OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs/temporal_cell_e_dagger")
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
EMB_DIM_DISTILL = 1280
C3_DIM = 192
EMB_DIM_IN = EMB_DIM_DISTILL + C3_DIM   # 1472
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


class CachedSequenceDatasetEDagger(Dataset):
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
            emb_idxs = []
            c3_idxs = []
            class_list = []
            for f in in_split:
                fn = f"{video_id}_{f['frame_number']}.jpg"
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
        emb = self.cache_emb["embeddings"][s["emb_idxs"]]    # (W, 1280)
        c3 = self.cache_c3["features"][s["c3_idxs"]]         # (W, 192)
        c3 = (c3 - self.c3_mean) / self.c3_std
        h = np.concatenate([emb, c3], axis=1)                # (W, 1472)
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


class TemporalTransformerEDagger(nn.Module):
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
        h = self.proj(x); h = self.pe(h)
        h = self.transformer(h); h = self.norm(h)
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
    print(f"\n{'='*60}\nSEED = {seed}\n{'='*60}")
    emb_path = EMB_DIR / f"seed{seed}_embeddings.npz"
    if not emb_path.exists():
        raise SystemExit(
            f"[fatal] distill embedding cache missing: {emb_path}\n"
            f"        Build it first via build_embedding_cache_distill.py "
            f"(clone of build_embedding_cache.py with output_dir_for() "
            f"swapped to stage2_distill_effb0[_seed*] and "
            f"model.deploy_mode = True after load_state_dict).")
    npz = np.load(emb_path)
    emb_cache = {
        "embeddings": np.array(npz["embeddings"]),
        "labels": np.array(npz["labels"]),
        "splits": np.array(npz["splits"]),
        "filenames": np.array(npz["filenames"]),
    }
    npz.close()

    train_ds = CachedSequenceDatasetEDagger(emb_cache, c3_cache, video_index, "train")
    val_ds = CachedSequenceDatasetEDagger(emb_cache, c3_cache, video_index, "val")
    test_ds = CachedSequenceDatasetEDagger(emb_cache, c3_cache, video_index, "test")
    print(f"[seed {seed}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_labels = np.array([s["center_class"] for s in train_ds.samples])
    cw = compute_class_weights(train_labels, N_CLASSES).to(DEVICE)

    torch.manual_seed(seed)
    model = TemporalTransformerEDagger().to(DEVICE)
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
    for c, a in test_pc.items():
        if not np.isnan(a):
            print(f"           {c:25s}: {a:.4f}")

    np.savez_compressed(out_dir / "test_predictions.npz",
                        logits=test_logits, labels=test_labels)

    return {"seed": seed,
            "best_val_macro_auc": best_val_macro,
            "best_epoch": best_epoch,
            "test_macro_auc": test_macro,
            "test_per_class": test_pc}


# Reference cross-seed numbers (from the eight-cell ablation table)
# used both for run-time logging and for the verdict in the report.
REF_CELL_A      = 0.7598   # per-frame baseline
REF_CELL_B      = 0.7788   # C2 only
REF_CELL_E      = 0.7828   # C2 + C3 (no C1) -- the 3-channel reference
REF_CELL_B_PLUS = 0.7901   # 5-ch + C2
REF_CELL_E_PLUS = 0.8038   # 5-ch + C2 + C3 -- the 5-channel headline

# Per-seed cell (b) and cell (e) test macro-AUC, for the report's
# per-seed delta columns (drawn from cell_e_log.txt and the cell-(b)
# entries in the framework summary).
B_PER_SEED = {41: 0.7658, 42: 0.7957, 43: 0.7618,
              44: 0.7875, 45: 0.7810, 47: 0.7810}
E_PER_SEED = {41: 0.7574, 42: 0.7551, 43: 0.7713,
              44: 0.7995, 45: 0.8093, 47: 0.8041}

# cell (b) per-class numbers reused for the per-class delta column.
CELL_B_PC = {
    "Angiectasia": 0.813, "Blood - fresh": 0.591, "Erosion": 0.776,
    "Erythema": 0.949, "Foreign Body": 0.993, "Ileocecal valve": 0.809,
    "Lymphangiectasia": 0.314, "Normal clean mucosa": 0.856,
    "Pylorus": 0.900, "Reduced Mucosal View": 0.583, "Ulcer": 0.983,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    seeds = args.only_seeds or SEEDS
    print(f"[main] cell (e†) on {seeds}; in_dim={EMB_DIM_IN} "
          f"(1280 distill emb + 192 C3)")
    print(f"[main] emb source: {EMB_DIR}")
    print(f"[main] output:     {OUTPUT_ROOT}")

    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
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
        (REPORT_DIR / "track_b_cell_e_dagger_predictions.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    macros = np.array([r["test_macro_auc"] for r in results])
    cs_mean = float(np.mean(macros))
    cs_std = float(np.std(macros))
    print(f"\n[main] CROSS-SEED test macro-AUC = {cs_mean:.4f} +- {cs_std:.4f}")
    print(f"[main] vs cell (a) baseline        {REF_CELL_A:.4f}: "
          f"{cs_mean - REF_CELL_A:+.4f}")
    print(f"[main] vs cell (b) C2-only         {REF_CELL_B:.4f}: "
          f"{cs_mean - REF_CELL_B:+.4f}")
    print(f"[main] vs cell (e) C2+C3 no-C1     {REF_CELL_E:.4f}: "
          f"{cs_mean - REF_CELL_E:+.4f}    <-- 3-ch reference")
    print(f"[main] vs cell (b+) 5-ch+C2        {REF_CELL_B_PLUS:.4f}: "
          f"{cs_mean - REF_CELL_B_PLUS:+.4f}")
    print(f"[main] vs cell (e+) 5-ch+C2+C3     {REF_CELL_E_PLUS:.4f}: "
          f"{cs_mean - REF_CELL_E_PLUS:+.4f}  <-- 5-ch headline")

    pc_agg: Dict[str, Dict[str, float]] = {}
    for cname in CLASS_NAMES:
        vals = np.array([r["test_per_class"].get(cname, float("nan"))
                          for r in results])
        if not np.isnan(vals).all():
            pc_agg[cname] = {"mean": float(np.nanmean(vals)),
                              "std": float(np.nanstd(vals))}

    md: List[str] = []
    md.append("# Track B: cell (e†) — temporal + C3 on the distilled "
              "3-channel RGB backbone\n")
    md.append(f"**Seeds:** {[r['seed'] for r in results]}")
    md.append("**Model:** transformer over concat[1280-d distill backbone "
              "emb + 192-d C3 z-scored]")
    md.append("**Backbone:** stage2_distill_effb0[_seed*] with "
              "deploy_mode=True (auxiliary P_blood head dropped at "
              "inference); 3-channel RGB input throughout.")
    md.append("")
    md.append("## Per-seed test macro-AUC vs the 3-channel reference cell (e)\n")
    md.append("| Seed | Cell (b) | Cell (e) | **Cell (e†)** | Δ vs (e) | Δ vs (b) |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["seed"]
        ed = r["test_macro_auc"]
        b = B_PER_SEED.get(s, float("nan"))
        e = E_PER_SEED.get(s, float("nan"))
        md.append(f"| {s} | {b:.4f} | {e:.4f} | **{ed:.4f}** | "
                  f"{ed - e:+.4f} | {ed - b:+.4f} |")
    md.append("")
    md.append(f"**Cross-seed cell (e†):** {cs_mean:.4f} ± {cs_std:.4f}")
    md.append(f"vs cell (a) per-frame baseline {REF_CELL_A}: "
              f"{cs_mean - REF_CELL_A:+.4f}")
    md.append(f"vs cell (b) C2-only            {REF_CELL_B}: "
              f"{cs_mean - REF_CELL_B:+.4f}")
    md.append(f"vs **cell (e) 3-ch reference** {REF_CELL_E}: "
              f"**{cs_mean - REF_CELL_E:+.4f}**")
    md.append(f"vs cell (b+) 5-ch input fusion {REF_CELL_B_PLUS}: "
              f"{cs_mean - REF_CELL_B_PLUS:+.4f}")
    md.append(f"vs **cell (e+) 5-ch headline** {REF_CELL_E_PLUS}: "
              f"**{cs_mean - REF_CELL_E_PLUS:+.4f}**")
    md.append("")
    md.append("## Per-class cross-seed test AUC vs cell (b)\n")
    md.append("| Class | Cell (b) | Cell (e†) | Δ vs (b) |")
    md.append("|---|---:|---:|---:|")
    for c in CLASS_NAMES:
        if c not in pc_agg or c not in CELL_B_PC:
            continue
        a = pc_agg[c]
        bb = CELL_B_PC[c]
        md.append(f"| {c} | {bb:.3f} | {a['mean']:.3f} ± {a['std']:.3f} "
                  f"| {a['mean'] - bb:+.3f} |")
    md.append("")
    md.append("## Verdict\n")
    delta_vs_e = cs_mean - REF_CELL_E
    delta_vs_eplus = cs_mean - REF_CELL_E_PLUS
    if delta_vs_e >= 0.003 and delta_vs_eplus >= -0.005:
        md.append(
            f"**STRONG PASS for deployment.** cell (e†) = {cs_mean:.4f} "
            f"beats the 3-channel reference cell (e) by {delta_vs_e:+.4f} "
            f"and recovers cell (e+)'s 5-channel headline within "
            f"{delta_vs_eplus:+.4f}. The distillation form carries the "
            f"physics signal into a deployable 3-channel RGB classifier. "
            f"Recommend cell (e†) as the headline deployment variant in "
            f"the medIA submission.")
    elif delta_vs_e >= 0.003:
        md.append(
            f"**PASS.** cell (e†) = {cs_mean:.4f} beats cell (e) by "
            f"{delta_vs_e:+.4f} -- distilled C1 contributes above no-C1 "
            f"-- but trails cell (e+) by {delta_vs_eplus:+.4f}. The "
            f"3-channel deployment story is intact; the 5-channel form "
            f"remains the absolute ceiling.")
    elif abs(delta_vs_e) < 0.003:
        md.append(
            f"**MARGINAL.** cell (e†) ≈ cell (e) ({delta_vs_e:+.4f}). "
            f"Distillation contributes nothing beyond C2+C3 on the "
            f"3-channel side; the per-frame distillation lift was what "
            f"the temporal transformer was already extracting. Cell (e) "
            f"itself is the right deployment cell to report.")
    else:
        md.append(
            f"**REGRESSION.** cell (e†) = {cs_mean:.4f} < cell (e) "
            f"{REF_CELL_E} ({delta_vs_e:+.4f}). The distillation "
            f"backbone interacts adversely with C3 when stacked through "
            f"the temporal head. Report cell (e†) honestly as a "
            f"deployment-direction probe that did not pay out; keep "
            f"cell (e) as the 3-channel reference.")
    md.append(f"\nTotal compute: {(time.time() - t0)/60:.1f} min")

    out_path = REPORT_DIR / "track_b_cell_e_dagger_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out_path}")


if __name__ == "__main__":
    main()
