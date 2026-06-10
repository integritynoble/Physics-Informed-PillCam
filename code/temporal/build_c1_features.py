"""
Build per-frame C1 structural-prior features for cell (c).
==========================================================

For each of the 47,238 Kvasir-Capsule frames, compute:
  P_blood(x)   ← analytic prior (Capsule-Endoscopy/physics_prior.py)
and extract 8 scalar structural features (matching Pilot 7's set):

  f_mean         frame mean of P_blood
  f_max          frame max  of P_blood
  f_top_1pct     mean of top 1% pixels
  f_top_5pct     mean of top 5% pixels
  f_frac_05      fraction of pixels above 0.5
  f_frac_07      fraction of pixels above 0.7
  f_central_max  max P_blood in central 50% of frame
  f_central_top_1pct  top 1% mean in central 50%

Output (one file, seed-independent):
  D:/kvasir_capsule/outputs/c1_features.npz
    filenames:  (47238,) array of frame filenames
    features:   (47238, 8) float32 — Pilot 7 scalar features

Compute: ~5 min on GTX 1660 Ti. Single forward pass over all frames;
no model state. The prior is a deterministic function of RGB, so this
file is reused across all 6 seeds in cell (c) training.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent.parent
CAPSULE = REPO_ROOT / "paper" / "Capsule-Endoscopy"
GASTRO = Path("D:/onedrive/UT_southwestern/GIproject/Dr. Zaman/"
              "gastroscopy_code_package (2)/gastroscopy_code_package")
sys.path.insert(0, str(CAPSULE))
sys.path.insert(0, str(GASTRO))

DATA_ROOT = Path("D:/kvasir_capsule/stage2_data")
OUTPUT_PATH = Path("D:/kvasir_capsule/outputs/c1_features.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 32

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
SPLIT_INT = {"train": 0, "val": 1, "test": 2}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

FEATURE_NAMES = [
    "f_mean", "f_max", "f_top_1pct", "f_top_5pct",
    "f_frac_05", "f_frac_07", "f_central_max", "f_central_top_1pct",
]


class CapsuleAllFramesDataset(Dataset):
    """Walks D:/kvasir_capsule/stage2_data/{train,val,test}/<class>/<file>.jpg.
    Uses gastroscopy_code build_transforms(224, train=False) for an
    identical pipeline to build_embedding_cache.py."""

    def __init__(self, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        self.samples: List[Tuple[Path, str, int, int]] = []
        for split in ("train", "val", "test"):
            split_dir = DATA_ROOT / split
            if not split_dir.is_dir():
                continue
            for class_dir in sorted(split_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                cname = class_dir.name
                if cname not in CLASS_NAMES:
                    continue
                cidx = CLASS_NAMES.index(cname)
                sidx = SPLIT_INT[split]
                for f in class_dir.iterdir():
                    if f.suffix.lower() == ".jpg":
                        self.samples.append((f, f.name, cidx, sidx))
        from datasets import build_transforms
        self.transform = build_transforms(image_size, train=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from PIL import Image
        path, fname, label, split = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x.float(), fname, label, split


def unnormalize(x_norm: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x_norm.device, dtype=x_norm.dtype)
    std = IMAGENET_STD.to(x_norm.device, dtype=x_norm.dtype)
    return (x_norm * std + mean).clamp(0.0, 1.0)


def compute_structural_features(p_blood: torch.Tensor) -> torch.Tensor:
    """p_blood: (B, H, W) in [0,1]; returns (B, 8) features."""
    B, H, W = p_blood.shape
    flat = p_blood.view(B, -1)
    n = flat.shape[1]

    f_mean = flat.mean(dim=1)
    f_max = flat.amax(dim=1)
    sorted_desc, _ = flat.sort(dim=1, descending=True)
    k_1 = max(1, n // 100)
    k_5 = max(1, n // 20)
    f_top_1pct = sorted_desc[:, :k_1].mean(dim=1)
    f_top_5pct = sorted_desc[:, :k_5].mean(dim=1)
    f_frac_05 = (flat > 0.5).float().mean(dim=1)
    f_frac_07 = (flat > 0.7).float().mean(dim=1)

    h_lo, h_hi = H // 4, (3 * H) // 4
    w_lo, w_hi = W // 4, (3 * W) // 4
    central = p_blood[:, h_lo:h_hi, w_lo:w_hi].contiguous().view(B, -1)
    f_central_max = central.amax(dim=1)
    sorted_c, _ = central.sort(dim=1, descending=True)
    kc_1 = max(1, central.shape[1] // 100)
    f_central_top_1pct = sorted_c[:, :kc_1].mean(dim=1)

    return torch.stack([f_mean, f_max, f_top_1pct, f_top_5pct,
                          f_frac_05, f_frac_07, f_central_max, f_central_top_1pct],
                         dim=1)


def main():
    from physics_prior import blood_probability

    print(f"[c1] device={DEVICE}")
    ds = CapsuleAllFramesDataset()
    n = len(ds)
    print(f"[c1] {n} frames")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    filenames: List[str] = [""] * n
    labels = np.zeros((n,), dtype=np.int64)
    splits = np.zeros((n,), dtype=np.int64)

    pos = 0
    t0 = time.time()
    with torch.no_grad():
        for x, fnames, ys, ss in loader:
            x = x.to(DEVICE, non_blocking=True)
            x_rgb = unnormalize(x.float())
            p_blood = blood_probability(x_rgb)
            feat = compute_structural_features(p_blood).cpu().numpy()
            bs = x.size(0)
            features[pos:pos + bs] = feat
            for k in range(bs):
                filenames[pos + k] = fnames[k]
            labels[pos:pos + bs] = ys.numpy()
            splits[pos:pos + bs] = ss.numpy()
            pos += bs
            if pos % (BATCH_SIZE * 50) == 0:
                rate = pos / max(0.001, (time.time() - t0))
                eta_min = (n - pos) / max(0.001, rate) / 60
                print(f"[c1] {pos}/{n}  rate={rate:.0f} fps  eta={eta_min:.1f} min")

    elapsed = (time.time() - t0) / 60
    print(f"[c1] done in {elapsed:.1f} min")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH,
                          filenames=np.array(filenames),
                          labels=labels, splits=splits,
                          features=features,
                          feature_names=np.array(FEATURE_NAMES))
    print(f"[c1] saved to {OUTPUT_PATH} "
          f"(size = {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")

    # Quick sanity stats
    print("\n[c1] feature distributions (mean +- std across all 47K frames):")
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        print(f"  {name:25s}: {col.mean():.3f} +- {col.std():.3f}  "
              f"min={col.min():.3f}  max={col.max():.3f}")


if __name__ == "__main__":
    main()
