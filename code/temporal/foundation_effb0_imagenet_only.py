"""
Cross-fine-tuning test: EfficientNet-B0 ImageNet-only (no capsule
fine-tuning), with temporal aggregation.
==================================================================

Companion to `foundation_backbone_temporal_test.py` (which tested
ResNet-50 ImageNet-only and got cell-c-vs-cell-b lift = +0.042).
Here we use the SAME architecture family (EfficientNet-B0) as the
capsule manuscript's reference, but WITHOUT capsule fine-tuning.

If the cell-c-vs-cell-b lift on ImageNet-only EfficientNet-B0 is
also positive (matching ResNet-50's +0.042), the parameterization-
mechanism boundary's "domain fine-tuning is the relevant axis"
finding is replicated within the EfficientNet family — controlling
for cross-architecture confounds in the ResNet-50 result.

Constraints respected (per user 2026-05-08):
  - Public dataset (Kvasir-Capsule)
  - Small compute (GTX 1660 Ti compatible; ~40-60 min total)

Pipeline:
  1. Forward all 47,238 capsule frames through ImageNet-pretrained
     EfficientNet-B0 (NO fine-tuning). Cache 1280-d pooled features.
  2. Train 4-layer temporal Transformer over cached features,
     comparing RGB-only (cell-b equivalent) and RGB + 8-d C1 scalar
     (cell-c equivalent) across 3 seeds.
  3. Compare cell-c-vs-cell-b lift to:
     - EfficientNet-B0 + capsule fine-tune (manuscript reference): -0.001
     - ResNet-50 ImageNet only:                                     +0.042

Output:
  paper/nature-machine-intelligence/docs/foundation_effb0_imagenet_only_report.md
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
DATA_ROOT = Path("D:/kvasir_capsule/stage2_data")
C1_PATH = Path("D:/kvasir_capsule/outputs/c1_features.npz")
EMB_OUT = Path("D:/kvasir_capsule/outputs/foundation_effb0_imagenet_only.npz")
INDEX_JSON = HERE / "video_index.json"
REPORT_DIR = HERE.parent.parent / "docs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [41, 42, 43]
BATCH_SIZE = 32
IMAGE_SIZE = 224

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
EMB_DIM_HIDDEN = 256
N_HEADS = 8
N_LAYERS = 4
FF_DIM = 512
DROPOUT = 0.1
EPOCHS = 20
TR_BATCH = 64
LR = 1e-4
WD = 1e-4
EARLY_STOP_PATIENCE = 5


class AllFramesDataset(Dataset):
    def __init__(self, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        self.samples: List[Tuple[Path, str, int, int]] = []
        for split in ("train", "val", "test"):
            sd = DATA_ROOT / split
            if not sd.is_dir():
                continue
            for cd in sorted(sd.iterdir()):
                if not cd.is_dir() or cd.name not in CLASS_NAMES:
                    continue
                cidx = CLASS_NAMES.index(cd.name)
                sidx = SPLIT_INT[split]
                for f in cd.iterdir():
                    if f.suffix.lower() == ".jpg":
                        self.samples.append((f, f.name, cidx, sidx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from PIL import Image
        from torchvision import transforms
        path, fname, label, split = self.samples[i]
        img = Image.open(path).convert("RGB")
        tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                                 interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        return tfm(img).float(), fname, label, split


def build_effb0_imagenet():
    import torchvision.models as M
    m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Identity()
    m.eval()
    return m


def extract_features_if_needed():
    if EMB_OUT.exists():
        print(f"[cache] reusing {EMB_OUT}")
        d = np.load(EMB_OUT)
        cache = {
            "embeddings": np.array(d["embeddings"]).astype(np.float32),
            "labels": np.array(d["labels"]),
            "splits": np.array(d["splits"]),
            "filenames": np.array(d["filenames"]),
        }
        d.close()
        return cache

    print(f"[cache] building EffNet-B0 ImageNet-only feature cache")
    model = build_effb0_imagenet().to(DEVICE)
    dataset = AllFramesDataset()
    n = len(dataset)
    if n == 0:
        raise SystemExit(f"no frames in {DATA_ROOT}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    feats = np.zeros((n, EMB_DIM_RGB), dtype=np.float32)
    fnames: List[str] = [""] * n
    labels = np.zeros((n,), dtype=np.int64)
    splits = np.zeros((n,), dtype=np.int64)

    pos = 0
    t0 = time.time()
    with torch.no_grad():
        for x, fn, y, s in loader:
            x = x.to(DEVICE, non_blocking=True)
            f = model(x)
            bs = x.size(0)
            feats[pos: pos + bs] = f.cpu().numpy()
            for k in range(bs):
                fnames[pos + k] = fn[k]
            labels[pos: pos + bs] = y.numpy()
            splits[pos: pos + bs] = s.numpy()
            pos += bs
            if pos % (BATCH_SIZE * 50) == 0:
                rate = pos / max(0.001, (time.time() - t0))
                eta = (n - pos) / max(0.001, rate) / 60
                print(f"[cache] {pos}/{n}  rate={rate:.0f} fps  eta={eta:.1f} min")
    print(f"[cache] done in {(time.time() - t0)/60:.1f} min")
    np.savez_compressed(EMB_OUT, filenames=np.array(fnames), labels=labels,
                          splits=splits, embeddings=feats)
    print(f"[cache] -> {EMB_OUT}  ({EMB_OUT.stat().st_size / 1e6:.1f} MB)")
    return {"embeddings": feats, "labels": labels, "splits": splits,
            "filenames": np.array(fnames)}


class CachedSequenceDataset(Dataset):
    def __init__(self, cache, video_index, split, c1_aux=None, window=WINDOW):
        self.cache = cache; self.window = window; self.c1_aux = c1_aux
        target_split = SPLIT_INT[split]
        fname_to_idx = {fn: i for i, fn in enumerate(cache["filenames"])}
        self.samples = []
        for video_id, frames in video_index["by_video"].items():
            in_split = [f for f in frames
                         if SPLIT_INT.get(f["split"], -1) == target_split]
            if not in_split: continue
            in_split.sort(key=lambda f: f["frame_number"])
            cache_idx_list, class_list = [], []
            for f in in_split:
                fn = f"{video_id}_{f['frame_number']}.jpg"
                if fn in fname_to_idx:
                    cache_idx_list.append(fname_to_idx[fn])
                    class_list.append(CLASS_NAMES.index(f["class"]))
            n = len(cache_idx_list)
            if n == 0: continue
            half = window // 2
            for i in range(n):
                lo = max(0, i - half); hi = min(n, i - half + window)
                if hi - lo < window:
                    if lo == 0: hi = min(n, window)
                    elif hi == n: lo = max(0, n - window)
                w = cache_idx_list[lo:hi]
                while len(w) < window: w.append(w[-1])
                self.samples.append({"window_cache_idxs": w,
                                       "center_class": class_list[i]})

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        emb = self.cache["embeddings"][s["window_cache_idxs"]]
        if self.c1_aux is not None:
            c1 = self.c1_aux[s["window_cache_idxs"]]
            emb = np.concatenate([emb, c1], axis=1)
        return torch.from_numpy(emb).float(), s["center_class"]


class SinusoidalPE(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float()
                          * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.window = WINDOW; self.center_idx = WINDOW // 2
        self.proj = nn.Linear(in_dim, EMB_DIM_HIDDEN)
        self.pe = SinusoidalPE(WINDOW, EMB_DIM_HIDDEN)
        layer = nn.TransformerEncoderLayer(d_model=EMB_DIM_HIDDEN, nhead=N_HEADS,
                                              dim_feedforward=FF_DIM, dropout=DROPOUT,
                                              batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(EMB_DIM_HIDDEN)
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT),
                                          nn.Linear(EMB_DIM_HIDDEN, N_CLASSES))
    def forward(self, x):
        h = self.proj(x); h = self.pe(h); h = self.transformer(h)
        h = self.norm(h)
        return self.classifier(h[:, self.center_idx])


def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    inv = 1.0 / counts; inv = inv * (n_classes / inv.sum())
    return torch.from_numpy(inv).float()


def per_class_auc(logits, labels):
    from sklearn.metrics import roc_auc_score
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    out = {}
    for j, c in enumerate(CLASS_NAMES):
        y = (labels == j).astype(np.int32)
        if y.sum() == 0 or y.sum() == len(y):
            out[c] = float("nan"); continue
        out[c] = float(roc_auc_score(y, probs[:, j]))
    return out


def macro_auc(pc):
    vals = [v for v in pc.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(model, loader):
    model.eval()
    all_l, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            all_l.append(model(x).cpu().numpy()); all_y.append(y.cpu().numpy())
    logits = np.concatenate(all_l, axis=0); labels = np.concatenate(all_y, axis=0)
    return per_class_auc(logits, labels), macro_auc(per_class_auc(logits, labels))


def train_run(cache, video_index, c1_aux, in_dim, seed, epochs):
    train_ds = CachedSequenceDataset(cache, video_index, "train", c1_aux=c1_aux)
    val_ds = CachedSequenceDataset(cache, video_index, "val", c1_aux=c1_aux)
    test_ds = CachedSequenceDataset(cache, video_index, "test", c1_aux=c1_aux)

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=TR_BATCH, shuffle=True,
                                num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=TR_BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=TR_BATCH, shuffle=False)

    train_labels = np.array([s["center_class"] for s in train_ds.samples])
    cw = class_weights(train_labels, N_CLASSES).to(DEVICE)

    torch.manual_seed(seed)
    model = TemporalTransformer(in_dim=in_dim).to(DEVICE)
    ce = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = -1.0; no_imp = 0; best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            loss = ce(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        _, val_macro = evaluate(model, val_loader)
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP_PATIENCE: break
    model.load_state_dict(best_state)
    _, test_macro = evaluate(model, test_loader)
    return test_macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    print(f"[main] device={DEVICE}  seeds={args.seeds}")
    cache = extract_features_if_needed()
    print(f"[main] cache: {cache['embeddings'].shape}")

    c1 = np.load(C1_PATH)
    c1_fn = list(c1["filenames"])
    c1_features = np.array(c1["features"])[:, :C1_DIM]
    c1.close()
    fn_to_c1 = {f: i for i, f in enumerate(c1_fn)}
    n_total = cache["embeddings"].shape[0]
    c1_aligned = np.zeros((n_total, C1_DIM), dtype=np.float32)
    for i, f in enumerate(cache["filenames"]):
        if f in fn_to_c1:
            c1_aligned[i] = c1_features[fn_to_c1[f]]
    train_mask = (cache["splits"] == 0)
    c1_z = ((c1_aligned - c1_aligned[train_mask].mean(axis=0))
            / (c1_aligned[train_mask].std(axis=0) + 1e-6)).astype(np.float32)

    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))

    res_b, res_c = [], []
    t0 = time.time()
    for seed in args.seeds:
        print(f"\n[seed {seed}] cell-(b)-equiv: ImageNet EffNet-B0 + temporal")
        m_b = train_run(cache, video_index, None, EMB_DIM_RGB, seed, args.epochs)
        res_b.append(m_b)
        print(f"  test macro = {m_b:.4f}")

        print(f"[seed {seed}] cell-(c)-equiv: ImageNet EffNet-B0 + C1_8 + temporal")
        m_c = train_run(cache, video_index, c1_z, EMB_DIM_RGB + C1_DIM, seed, args.epochs)
        res_c.append(m_c)
        print(f"  test macro = {m_c:.4f}")
        print(f"[seed {seed}] cell-c-vs-cell-b lift = {m_c - m_b:+.4f}")

    arr_b = np.array(res_b); arr_c = np.array(res_c)
    cs_lift = (arr_c - arr_b).mean()
    print(f"\n[main] cross-seed:")
    print(f"  cell (b)-equiv = {arr_b.mean():.4f} +- {arr_b.std():.4f}")
    print(f"  cell (c)-equiv = {arr_c.mean():.4f} +- {arr_c.std():.4f}")
    print(f"  lift           = {cs_lift:+.4f} +- {(arr_c - arr_b).std():.4f}")

    # Reference comparisons
    eff_finetune_lift = -0.0014    # EffNet-B0 capsule fine-tune
    resnet_imagenet_lift = +0.0424  # ResNet-50 ImageNet only

    print(f"\n  reference comparisons:")
    print(f"  EffNet-B0 + capsule fine-tune (manuscript)     : {eff_finetune_lift:+.4f}")
    print(f"  ResNet-50  + ImageNet only                     : {resnet_imagenet_lift:+.4f}")
    print(f"  EffNet-B0 + ImageNet only (this run)           : {cs_lift:+.4f}")

    md = []
    md.append("# EffNet-B0 ImageNet-only test\n")
    md.append("**Date:** 2026-05-08")
    md.append("**Backbone:** EfficientNet-B0, ImageNet-1K-V1 weights, NO capsule fine-tuning")
    md.append("**Goal:** isolate the 'domain fine-tuning is the relevant axis' finding "
              "from cross-architecture confounds (the ResNet-50 result was on a different "
              "architecture). Same architecture as capsule manuscript, no fine-tuning, "
              "with temporal aggregation.")
    md.append("")
    md.append("## Per-seed results\n")
    md.append("| Seed | cell (b) | cell (c) | Δ |")
    md.append("|---:|---:|---:|---:|")
    for i, seed in enumerate(args.seeds):
        md.append(f"| {seed} | {res_b[i]:.4f} | {res_c[i]:.4f} | "
                  f"{res_c[i] - res_b[i]:+.4f} |")
    md.append("")
    md.append(f"**Cross-seed mean:**")
    md.append(f"- cell (b)-equiv = {arr_b.mean():.4f} ± {arr_b.std():.4f}")
    md.append(f"- cell (c)-equiv = {arr_c.mean():.4f} ± {arr_c.std():.4f}")
    md.append(f"- lift = {cs_lift:+.4f} ± {(arr_c - arr_b).std():.4f}")
    md.append("")
    md.append("## Reference comparisons\n")
    md.append("| Backbone | Domain fine-tune? | cell-c-vs-cell-b lift |")
    md.append("|---|---|---:|")
    md.append(f"| EfficientNet-B0 | Yes (capsule) | {eff_finetune_lift:+.4f} |")
    md.append(f"| ResNet-50 | No | {resnet_imagenet_lift:+.4f} |")
    md.append(f"| **EfficientNet-B0** | **No** | **{cs_lift:+.4f}** |")
    md.append("")
    md.append("## Verdict\n")
    if cs_lift > 0.010:
        md.append(f"**Domain fine-tuning is the relevant axis, not architecture.** "
                  f"EffNet-B0 ImageNet-only gives lift = {cs_lift:+.4f}, matching "
                  f"ResNet-50 ImageNet-only ({resnet_imagenet_lift:+.4f}) in "
                  f"direction and roughly in magnitude. Both ImageNet-only "
                  f"backbones benefit from summary-stat C1; both fine-tuned "
                  f"backbones do not. The parameterization-mechanism boundary's "
                  f"location is determined by what the backbone has learned, not "
                  f"by raw architecture.")
    elif cs_lift > -0.005:
        md.append(f"**Mixed evidence.** EffNet-B0 ImageNet-only gives "
                  f"lift = {cs_lift:+.4f}, between the fine-tuned EffNet-B0 "
                  f"({eff_finetune_lift:+.4f}) and ImageNet ResNet-50 "
                  f"({resnet_imagenet_lift:+.4f}). May indicate the boundary "
                  f"depends on both architecture and domain fine-tuning.")
    else:
        md.append(f"**Surprising.** lift = {cs_lift:+.4f} matches the fine-tuned "
                  f"EffNet-B0 reference, contradicting the ResNet-50 ImageNet-only "
                  f"result. Domain fine-tuning may not be the right axis.")
    md.append(f"\n**Compute:** {(time.time() - t0)/60:.1f} min for {len(args.seeds)} seeds.")

    out = REPORT_DIR / "foundation_effb0_imagenet_only_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out}")


if __name__ == "__main__":
    main()
