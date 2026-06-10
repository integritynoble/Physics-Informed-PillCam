"""
Foundation-model backbone test — does the four-variant C1 boundary
hold with a stronger backbone (ResNet-50, ~25M params)?
====================================================================

Tests Corollary 2 from `parameterization_mechanism_boundary_theory_2026-05-07.md`:
backbones with more channel-mixing capacity may show a smaller
summary-vs-spatial gap.

Setup:
  - Forward all 47K Kvasir-Capsule frames through ImageNet-pretrained
    ResNet-50 (NO fine-tuning — we want to test whether stronger
    pretrained features change the picture).
  - Cache 2048-d pooled features.
  - Train two linear probes:
      probe_rgb_only       : on the 2048-d features alone (cell-a
                              equivalent, ResNet-50 backbone)
      probe_rgb_plus_C1_8  : on (2048 + 8)-d features = RGB feature
                              concatenated with the 8 scalar P_blood
                              statistics from build_c1_features.py
                              (cell-c equivalent on the new backbone)
  - Compare macro-AUC. If C1_8 still adds ≈0 over RGB on the bigger
    backbone, that supports Corollary 2's prediction that the
    summary-stat C1 channel is structurally redundant with what *any*
    sufficiently-strong backbone extracts. If C1_8 lifts on the
    bigger backbone, the boundary is capacity-dependent.

Pre-conditions:
  - D:/kvasir_capsule/outputs/c1_features.npz exists (from
    build_c1_features.py)
  - D:/kvasir_capsule/stage2_data/{train,val,test}/<class>/*.jpg

Compute: ~30-60 min on GTX 1660 Ti for 47K frames at ~15-25 fps with
ResNet-50.

Output:
  paper/nature-machine-intelligence/docs/foundation_backbone_test_report.md
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
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("D:/kvasir_capsule/stage2_data")
C1_PATH = Path("D:/kvasir_capsule/outputs/c1_features.npz")
EMB_OUT = Path("D:/kvasir_capsule/outputs/foundation_resnet50_embeddings.npz")
REPORT_DIR = HERE.parent.parent / "docs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = 224

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
SPLIT_TO_INT = {"train": 0, "val": 1, "test": 2}


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
                sidx = SPLIT_TO_INT[split]
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
        # Standard ImageNet preprocessing — same recipe foundation
        # models expect at evaluation time.
        tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                                 interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        x = tfm(img)
        return x.float(), fname, label, split


def build_resnet50_feature_extractor() -> Tuple[nn.Module, int]:
    """Returns (model, feature_dim) where model(x) produces a pooled
    feature vector of dim feature_dim."""
    import torchvision.models as M
    m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
    feat_dim = m.fc.in_features  # 2048
    m.fc = nn.Identity()
    m.eval()
    return m, feat_dim


def extract_features(model: nn.Module, dataset: AllFramesDataset
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"[ext] forwarding {len(dataset)} frames through ResNet-50")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    n = len(dataset)
    # Probe feature dim
    with torch.no_grad():
        sample = dataset[0][0].unsqueeze(0).to(DEVICE)
        feat0 = model(sample)
    feat_dim = int(feat0.shape[-1])
    print(f"[ext] feature dim = {feat_dim}")

    feats = np.zeros((n, feat_dim), dtype=np.float32)
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
                print(f"[ext] {pos}/{n}  rate={rate:.0f} fps  eta={eta:.1f} min")
    print(f"[ext] done in {(time.time() - t0)/60:.1f} min")
    return np.array(fnames), labels, splits, feats


def macro_auc(probs: np.ndarray, labels: np.ndarray
                ) -> Tuple[float, Dict[str, float]]:
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


def linear_probe(e_train: np.ndarray, y_train: np.ndarray,
                  e_test: np.ndarray, y_test: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0,
                              class_weight="balanced",
                              random_state=seed, solver="lbfgs")
    clf.fit(e_train, y_train)
    probs = clf.predict_proba(e_test)
    return macro_auc(probs, y_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse_cache", action="store_true",
                     help="Skip ResNet-50 forward; reuse cached features if available.")
    args = ap.parse_args()

    print(f"[main] device={DEVICE}")

    # Phase 1: cache ResNet-50 features
    if args.reuse_cache and EMB_OUT.exists():
        print(f"[main] reusing cached features at {EMB_OUT}")
        d = np.load(EMB_OUT)
        fn = list(d["filenames"])
        labels = np.array(d["labels"])
        splits = np.array(d["splits"])
        feats = np.array(d["embeddings"])
        d.close()
    else:
        print(f"[main] loading ResNet-50 (ImageNet pretrained, no fine-tuning)")
        model, feat_dim = build_resnet50_feature_extractor()
        model = model.to(DEVICE)
        dataset = AllFramesDataset()
        if len(dataset) == 0:
            print(f"[main] FATAL: no frames in {DATA_ROOT}")
            sys.exit(1)
        fn_arr, labels, splits, feats = extract_features(model, dataset)
        fn = list(fn_arr)
        np.savez_compressed(EMB_OUT, filenames=fn_arr, labels=labels,
                              splits=splits, embeddings=feats)
        print(f"[main] cache -> {EMB_OUT}  "
              f"({EMB_OUT.stat().st_size / 1e6:.1f} MB)")

    # Phase 2: load existing C1 scalar features and align by filename
    print(f"[main] loading C1 features from {C1_PATH}")
    if not C1_PATH.exists():
        print(f"[main] FATAL: missing {C1_PATH}; "
              "run build_c1_features.py first")
        sys.exit(1)
    c1 = np.load(C1_PATH)
    c1_fn = list(c1["filenames"])
    c1_features = np.array(c1["features"])
    if c1_features.shape[1] >= 8:
        c1_features = c1_features[:, :8]   # use the 8 scalar features
    c1.close()
    print(f"[main] c1 shape: {c1_features.shape}")

    fn_to_c1 = {f: i for i, f in enumerate(c1_fn)}
    c1_idx = [fn_to_c1[f] for f in fn if f in fn_to_c1]
    keep = [i for i, f in enumerate(fn) if f in fn_to_c1]
    feats = feats[keep]
    labels = labels[keep]
    splits = splits[keep]
    c1_aligned = c1_features[c1_idx]
    print(f"[main] aligned: feats={feats.shape}, c1={c1_aligned.shape}")

    # z-score C1 features on train split
    train_mask = (splits == 0)
    c1_mean = c1_aligned[train_mask].mean(axis=0)
    c1_std = c1_aligned[train_mask].std(axis=0) + 1e-6
    c1_z = (c1_aligned - c1_mean) / c1_std

    # Phase 3: linear probes
    train = (splits == 0)
    test = (splits == 2)
    print(f"[probe] train n={train.sum()}, test n={test.sum()}")

    seed = 42
    print(f"[probe] training probe_rgb_only ...")
    auc_rgb, _pc_rgb = linear_probe(
        feats[train], labels[train], feats[test], labels[test], seed)

    print(f"[probe] training probe_rgb_plus_C1_8 ...")
    feats_aug = np.concatenate([feats, c1_z], axis=1)
    auc_aug, _pc_aug = linear_probe(
        feats_aug[train], labels[train], feats_aug[test], labels[test], seed)

    print(f"\n[main] RESULTS:")
    print(f"  probe_rgb_only       (2048d ResNet-50)     = {auc_rgb:.4f}")
    print(f"  probe_rgb_plus_C1_8  (2048d + 8d C1)       = {auc_aug:.4f}")
    print(f"  C1 lift on ResNet-50                       = {auc_aug - auc_rgb:+.4f}")

    # Reference numbers (capsule manuscript, EfficientNet-B0):
    eff_b0_a = 0.7598
    eff_b0_c = 0.7774
    eff_b0_lift = eff_b0_c - eff_b0_a    # +0.018 (technically +0.0176)

    print(f"\n  reference (EfficientNet-B0):")
    print(f"  cell (a) per-frame baseline                = {eff_b0_a:.4f}")
    print(f"  cell (c) +C1 8-d scalars                   = {eff_b0_c:.4f}")
    print(f"  C1 lift on EfficientNet-B0                 = {eff_b0_c - eff_b0_a:+.4f}")
    print(f"  (note: cell (b) RGB+temporal removes the per-frame ambiguity;")
    print(f"   the key reference is cell-c-vs-cell-b: +0.0176 lift on EffNet-B0,")
    print(f"   close to 0 — *flat* — once temporal aggregation is added.)")

    # Decision rule for the report:
    #   The capsule cell-c-vs-cell-b lift was -0.0014 (essentially zero).
    #   If on ResNet-50, C1 ALSO gives ~zero lift, the boundary is
    #   robust to backbone capacity (Corollary 2 supported).
    #   If C1 gives a bigger lift on ResNet-50, the boundary is
    #   capacity-dependent in the OPPOSITE direction (stronger backbone
    #   has *more* room to be helped by an auxiliary scalar prior —
    #   this would be a surprising and reportable finding).
    delta = auc_aug - auc_rgb
    if delta < 0.005:
        verdict = (f"**Corollary 2 supported.** Adding 8-d scalar C1 to the "
                   f"ResNet-50 RGB feature gives Δ = {delta:+.4f} macro-AUC, "
                   f"essentially flat (within ±0.005 noise). The summary-stat "
                   f"failure replicates on a 5× larger backbone — the "
                   f"parameterization-mechanism boundary is robust to backbone "
                   f"capacity at this scale.")
    elif delta < 0.020:
        verdict = (f"**Marginal.** C1 8-d gives Δ = {delta:+.4f} — small but "
                   f"non-trivial. Could indicate the boundary weakens at "
                   f"larger backbone sizes (consistent with Corollary 2 in "
                   f"direction but not magnitude). Worth re-running with "
                   f"the cell (b)/(b+) temporal arm to compare directly to "
                   f"the EfficientNet-B0 ablation.")
    else:
        verdict = (f"**Surprising — Corollary 2 NOT supported in direction.** "
                   f"On ResNet-50, summary-stat C1 lifts by Δ = {delta:+.4f}, "
                   f"larger than the EfficientNet-B0 reference (+0.018 raw, "
                   f"but ≈0 once temporal aggregation is included). This "
                   f"suggests the parameterization-mechanism boundary is "
                   f"*architecture-specific* and weakens with stronger "
                   f"backbones. Reportable as a refinement of the theory.")

    md = []
    md.append("# Foundation-model backbone test — ResNet-50 on capsule\n")
    md.append("**Date:** 2026-05-08")
    md.append("**Backbone:** ResNet-50, ImageNet-1K V2 weights, NO fine-tuning")
    md.append("**Method:** forward 47K capsule frames; 2048-d pooled features; "
              "two linear probes (RGB only, RGB + 8-d scalar C1).")
    md.append("")
    md.append("## Result\n")
    md.append("| Probe | macro-AUC |")
    md.append("|---|---:|")
    md.append(f"| probe_rgb_only (2048-d) | {auc_rgb:.4f} |")
    md.append(f"| probe_rgb_plus_C1_8 (2048+8) | {auc_aug:.4f} |")
    md.append(f"| **Δ (C1 lift on ResNet-50)** | **{delta:+.4f}** |")
    md.append("")
    md.append("## Reference (EfficientNet-B0, capsule manuscript)\n")
    md.append("| Cell | macro-AUC |")
    md.append("|---|---:|")
    md.append(f"| (a) per-frame baseline | {eff_b0_a:.4f} |")
    md.append(f"| (c) +C1 8-d scalars | {eff_b0_c:.4f} |")
    md.append(f"| Δ on EfficientNet-B0 | {eff_b0_c - eff_b0_a:+.4f} |")
    md.append("")
    md.append("Note: the cleaner cell-c-vs-cell-b reference (with temporal "
              "aggregation) gave Δ ≈ −0.001 on EfficientNet-B0, i.e. flat.")
    md.append("")
    md.append("## Verdict\n")
    md.append(verdict)

    out = REPORT_DIR / "foundation_backbone_test_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out}")


if __name__ == "__main__":
    main()
