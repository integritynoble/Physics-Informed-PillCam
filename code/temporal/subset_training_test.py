"""
Subset-training test: does the parameterization-mechanism boundary
move continuously with the amount of backbone training data?
====================================================================

The conjecture in Section 5 says the boundary's location is
controlled by how much P-relevant information the backbone has
already extracted from training data. We test this directly:
artificially restrict the training data to 25%, 50%, and 100% of
the capsule train split, fine-tune EfficientNet-B0 from
ImageNet pretraining at each level, then run cells (b) and (c)
temporal aggregation on each.

Predictions:
  - 100%: cell-c-vs-cell-b lift ~ -0.001  (manuscript reference)
  - 50%:  lift somewhere between -0.001 and +0.021 (linear-ish interp)
  - 25%:  lift somewhere between +0.021 and the larger ImageNet-only result

If the lift moves smoothly with subset size, the conjecture's
mechanism is empirically supported as a continuous function of
backbone training data, not just a discrete fine-tuned/not split.

Constraints respected: public Kvasir-Capsule, GTX 1660 Ti.

Compute estimate: ~30-45 min per backbone fine-tune × 1 seed × 2
subsets (25%, 50%) = ~1-1.5 GPU-hours. The 100% reference is
already in the manuscript.

For minimal compute we run seed 42 only (the canonical seed) at
25% and 50% subsets. Three datapoints (25, 50, 100) are sufficient
to test the directional prediction.

Pipeline:
  1. For each subset_frac in [0.25, 0.50]:
     a. Sample stratified subset of capsule train frames
     b. Fine-tune EfficientNet-B0 from ImageNet pretrain on subset
        (3-channel input, no analytic prior)
     c. Cache 1280-d pooled features for ALL 47K frames
     d. Train cell (b) and cell (c) temporal heads on the cached features
     e. Compare macro-AUC

Output:
  paper/nature-machine-intelligence/docs/subset_training_test_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
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
INDEX_JSON = HERE / "video_index.json"
REPORT_DIR = HERE.parent.parent / "docs"
OUT_DIR_BASE = Path("D:/kvasir_capsule/outputs/subset_training")
OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SUBSETS = [0.25, 0.50]
BACKBONE_EPOCHS = 10
BACKBONE_LR = 1e-3
BACKBONE_WD = 1e-4
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
TEMPORAL_EPOCHS = 20
TEMPORAL_BATCH = 64
TEMPORAL_LR = 1e-4
TEMPORAL_WD = 1e-4
EARLY_STOP_PATIENCE = 5


# ----------------------------------------------------------------------
# Backbone training on a stratified subset
# ----------------------------------------------------------------------

class SubsetCapsuleDataset(Dataset):
    """Walks D:/kvasir_capsule/stage2_data/<split>/<class>/*.jpg and
    returns a stratified subset of the train split if `subset_frac`
    < 1.0. Val and test are always full."""

    def __init__(self, split: str, subset_frac: float = 1.0,
                 seed: int = SEED, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        sd = DATA_ROOT / split
        all_samples: List[Tuple[Path, int]] = []
        for cd in sorted(sd.iterdir()):
            if not cd.is_dir() or cd.name not in CLASS_NAMES:
                continue
            cidx = CLASS_NAMES.index(cd.name)
            for f in cd.iterdir():
                if f.suffix.lower() == ".jpg":
                    all_samples.append((f, cidx))
        if split == "train" and subset_frac < 1.0:
            by_class: Dict[int, List[int]] = defaultdict(list)
            for i, (_, lbl) in enumerate(all_samples):
                by_class[lbl].append(i)
            rng = np.random.default_rng(seed * 1000 + int(subset_frac * 100))
            kept = []
            for c, idxs in by_class.items():
                idxs_sorted = sorted(idxs)
                rng.shuffle(idxs_sorted)
                k = max(1, int(round(len(idxs_sorted) * subset_frac)))
                kept.extend(idxs_sorted[:k])
            all_samples = [all_samples[i] for i in kept]
        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from PIL import Image
        from torchvision import transforms
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                                 interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        return tfm(img).float(), label


def build_backbone():
    import torchvision.models as M
    m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
    feat_dim = m.classifier[1].in_features
    m.classifier = nn.Identity()
    return m, feat_dim


class BackboneClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, feat_dim = build_backbone()
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, N_CLASSES),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def fine_tune_backbone(subset_frac: float, seed: int) -> Path:
    """Fine-tune EfficientNet-B0 from ImageNet pretrain on the
    subset of capsule train. Returns the path to the saved
    backbone-classifier checkpoint."""
    out_dir = OUT_DIR_BASE / f"seed{seed}_subset{int(subset_frac * 100)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "best_model.pt"
    if ckpt.exists():
        print(f"[subset {int(subset_frac*100)}] reusing {ckpt}")
        return ckpt

    print(f"\n[subset {int(subset_frac*100)}] fine-tuning EfficientNet-B0 "
          f"on {int(subset_frac*100)}% of capsule train")
    train_ds = SubsetCapsuleDataset("train", subset_frac=subset_frac, seed=seed)
    val_ds = SubsetCapsuleDataset("val", subset_frac=1.0, seed=seed)
    print(f"[subset] train={len(train_ds)} val={len(val_ds)}")

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=2, pin_memory=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    train_labels = np.array([lbl for _, lbl in train_ds.samples])
    counts = np.bincount(train_labels, minlength=N_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    inv = 1.0 / counts; inv = inv * (N_CLASSES / inv.sum())
    cw = torch.from_numpy(inv).float().to(DEVICE)

    torch.manual_seed(seed)
    model = BackboneClassifier().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=BACKBONE_LR,
                              weight_decay=BACKBONE_WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BACKBONE_EPOCHS)
    ce = nn.CrossEntropyLoss(weight=cw)

    best_val_macro = -1.0
    best_state = None
    t0 = time.time()
    for epoch in range(1, BACKBONE_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            loss = ce(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        # Compute val accuracy as a quick proxy
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                pred = model(x).argmax(dim=-1)
                correct += (pred == y).sum().item(); total += y.size(0)
        val_acc = correct / max(1, total)
        if val_acc > best_val_macro:
            best_val_macro = val_acc
            best_state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
        elapsed = (time.time() - t0) / 60
        print(f"[subset] epoch {epoch:2d}  val_acc={val_acc:.4f}  "
              f"elapsed={elapsed:.1f} min")

    torch.save({"model_state": best_state,
                  "args": {"model_name": "efficientnet_b0",
                           "extra_channels": 0,
                           "subset_frac": subset_frac},
                  "class_names": CLASS_NAMES,
                  "best_val_acc": best_val_macro},
                  ckpt)
    print(f"[subset {int(subset_frac*100)}] saved -> {ckpt}")
    return ckpt


# ----------------------------------------------------------------------
# Embedding cache extraction
# ----------------------------------------------------------------------

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


def cache_features(ckpt_path: Path, subset_frac: float, seed: int) -> Dict:
    cache_path = OUT_DIR_BASE / f"seed{seed}_subset{int(subset_frac*100)}_features.npz"
    if cache_path.exists():
        print(f"[cache] reusing {cache_path}")
        d = np.load(cache_path)
        out = {k: np.array(d[k]) for k in ("filenames", "labels", "splits", "embeddings")}
        d.close()
        return out

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = BackboneClassifier().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = AllFramesDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    n = len(dataset)
    feats = np.zeros((n, EMB_DIM_RGB), dtype=np.float32)
    fnames: List[str] = [""] * n
    labels = np.zeros((n,), dtype=np.int64)
    splits = np.zeros((n,), dtype=np.int64)

    pos = 0
    t0 = time.time()
    with torch.no_grad():
        for x, fn, y, s in loader:
            x = x.to(DEVICE, non_blocking=True)
            f = model.backbone(x)
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
    np.savez_compressed(cache_path, filenames=np.array(fnames), labels=labels,
                          splits=splits, embeddings=feats)
    return {"embeddings": feats, "labels": labels, "splits": splits,
            "filenames": np.array(fnames)}


# ----------------------------------------------------------------------
# Temporal head
# ----------------------------------------------------------------------

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
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
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


def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    inv = 1.0 / counts; inv = inv * (n_classes / inv.sum())
    return torch.from_numpy(inv).float()


def train_temporal(cache, video_index, c1_aux, in_dim, seed, epochs):
    train_ds = CachedSequenceDataset(cache, video_index, "train", c1_aux=c1_aux)
    val_ds = CachedSequenceDataset(cache, video_index, "val", c1_aux=c1_aux)
    test_ds = CachedSequenceDataset(cache, video_index, "test", c1_aux=c1_aux)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=TEMPORAL_BATCH, shuffle=True,
                                num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=TEMPORAL_BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=TEMPORAL_BATCH, shuffle=False)

    train_labels = np.array([s["center_class"] for s in train_ds.samples])
    cw = class_weights(train_labels, N_CLASSES).to(DEVICE)

    torch.manual_seed(seed)
    model = TemporalTransformer(in_dim=in_dim).to(DEVICE)
    ce = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=TEMPORAL_LR, weight_decay=TEMPORAL_WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = -1.0; best_state = None; no_imp = 0
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
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--subsets", type=float, nargs="+", default=SUBSETS)
    args = ap.parse_args()

    print(f"[main] device={DEVICE}  seed={args.seed}  subsets={args.subsets}")

    # Load c1
    c1 = np.load(C1_PATH)
    c1_fn = list(c1["filenames"])
    c1_features = np.array(c1["features"])[:, :C1_DIM]
    c1.close()
    fn_to_c1 = {f: i for i, f in enumerate(c1_fn)}

    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))

    results = []
    t_start = time.time()
    for frac in args.subsets:
        print(f"\n{'='*60}\nSUBSET = {int(frac*100)}%\n{'='*60}")
        ckpt = fine_tune_backbone(frac, args.seed)
        cache = cache_features(ckpt, frac, args.seed)
        n_total = cache["embeddings"].shape[0]

        # Align C1 features
        c1_aligned = np.zeros((n_total, C1_DIM), dtype=np.float32)
        for i, f in enumerate(cache["filenames"]):
            if f in fn_to_c1:
                c1_aligned[i] = c1_features[fn_to_c1[f]]
        train_mask = (cache["splits"] == 0)
        c1_z = ((c1_aligned - c1_aligned[train_mask].mean(axis=0))
                / (c1_aligned[train_mask].std(axis=0) + 1e-6)).astype(np.float32)

        print(f"[temporal] training cell (b)-equiv (RGB)")
        m_b = train_temporal(cache, video_index, None, EMB_DIM_RGB,
                              args.seed, TEMPORAL_EPOCHS)
        print(f"  test macro = {m_b:.4f}")

        print(f"[temporal] training cell (c)-equiv (RGB + C1_8)")
        m_c = train_temporal(cache, video_index, c1_z, EMB_DIM_RGB + C1_DIM,
                              args.seed, TEMPORAL_EPOCHS)
        print(f"  test macro = {m_c:.4f}")
        print(f"  cell-c-vs-cell-b lift = {m_c - m_b:+.4f}")

        results.append({
            "subset_frac": frac,
            "auc_cell_b": m_b,
            "auc_cell_c": m_c,
            "lift": m_c - m_b,
        })

    print("\n[main] cross-subset summary:")
    print(f"  {'subset':>8s}  {'cell-b':>10s}  {'cell-c':>10s}  {'lift':>10s}")
    for r in results:
        print(f"  {int(r['subset_frac']*100):>6d}%  {r['auc_cell_b']:>10.4f}  "
              f"{r['auc_cell_c']:>10.4f}  {r['lift']:>+10.4f}")

    # Reference points (from manuscript and prior runs):
    print(f"\n  reference (seed 42):")
    print(f"  100%     0.7957  0.7386  -0.0571")
    print(f"  ImageNet only  0.6752  0.6734  -0.0017  (this is wrong direction;")
    print(f"  actually +0.021 cross-seed mean. seed 42 is the noisy one.)")

    md = []
    md.append("# Subset-training test on capsule\n")
    md.append("**Date:** 2026-05-08")
    md.append(f"**Seed:** {args.seed}")
    md.append(f"**Subsets tested:** {[f'{int(f*100)}%' for f in args.subsets]} of capsule train data")
    md.append("**Goal:** test whether the parameterization-mechanism boundary's "
              "location moves continuously with the amount of backbone training "
              "data, supporting the conjecture that the boundary is determined "
              "by what the backbone has extracted from training.")
    md.append("")
    md.append("## Results\n")
    md.append("| Subset | cell (b) | cell (c) | cell-c-vs-cell-b lift |")
    md.append("|---:|---:|---:|---:|")
    for r in results:
        md.append(f"| {int(r['subset_frac']*100)}% | {r['auc_cell_b']:.4f} "
                  f"| {r['auc_cell_c']:.4f} | {r['lift']:+.4f} |")
    md.append(f"| 100% (manuscript) | 0.7957 | 0.7386 | $-0.0571$ |")
    md.append(f"| ImageNet only (single seed=42) | 0.6752 | 0.6734 | $-0.0017$ |")
    md.append("")
    md.append("Note: seed 42 is the most variable seed across the eight-cell "
              "ablation; for cleaner subset-training evidence multiple seeds "
              "would be needed. We report the single-seed datapoint here as "
              "an initial probe.")
    md.append(f"\n**Compute:** {(time.time() - t_start)/60:.1f} min")

    out = REPORT_DIR / "subset_training_test_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out}")


if __name__ == "__main__":
    main()
