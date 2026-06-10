"""
Extend C1 with vessel-topology features.
==========================================

Adds 5 connected-component topology features to the existing 8 scalar
C1 features, giving 13 total per-frame features. Tests the hypothesis
that the cell (c) regression was caused by the *summary statistics*
losing spatial structure -- topology proxies preserve some of that.

New features per frame (computed from `P_blood > 0.5` mask via
scipy.ndimage.label):

    n_blobs                  # number of connected components
    largest_blob_area        # area of largest blob (pixels), normalized to [0,1]
    largest_blob_centrality  # centroid distance from frame center, normalized
    largest_blob_aspect_ratio  # bounding-box w/h, log-scaled
    blob_size_entropy        # Shannon entropy of blob-size distribution

These distinguish punctate (Angiectasia: many small blobs) from diffuse
(Blood-fresh: one large blob), which the 8 scalars conflate.

Output: D:/kvasir_capsule/outputs/c1_topology_features.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

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
OUTPUT_PATH = Path("D:/kvasir_capsule/outputs/c1_topology_features.npz")
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

# 8 scalar features (existing) + 5 topology features (new) = 13 total
FEATURE_NAMES = [
    "f_mean", "f_max", "f_top_1pct", "f_top_5pct",
    "f_frac_05", "f_frac_07", "f_central_max", "f_central_top_1pct",
    "n_blobs", "largest_blob_area", "largest_blob_centrality",
    "largest_blob_aspect_ratio", "blob_size_entropy",
]


class CapsuleAllFramesDataset(Dataset):
    def __init__(self, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        self.samples: List[Tuple[Path, str, int, int]] = []
        for split in ("train", "val", "test"):
            sd = DATA_ROOT / split
            for cd in sorted(sd.iterdir()):
                if not cd.is_dir():
                    continue
                cname = cd.name
                if cname not in CLASS_NAMES:
                    continue
                cidx = CLASS_NAMES.index(cname)
                sidx = SPLIT_INT[split]
                for f in cd.iterdir():
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


def compute_scalar_features(p_blood: torch.Tensor) -> torch.Tensor:
    """8 scalar global summaries (matches existing build_c1_features.py)."""
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
                          f_frac_05, f_frac_07, f_central_max,
                          f_central_top_1pct], dim=1)


def compute_topology_features(p_blood: torch.Tensor) -> np.ndarray:
    """5 connected-component features per frame.
    Threshold P_blood at 0.5 to get binary blob mask, then label
    connected components and extract shape descriptors."""
    from scipy.ndimage import label as cc_label
    B, H, W = p_blood.shape
    p_np = p_blood.detach().cpu().numpy()
    feats = np.zeros((B, 5), dtype=np.float32)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    diag = np.sqrt(cx ** 2 + cy ** 2)

    for b in range(B):
        mask = p_np[b] > 0.5
        if mask.sum() == 0:
            # Empty blob set: zero out (will be z-scored later, so 0 is fine)
            continue
        labels, n = cc_label(mask)
        if n == 0:
            continue

        sizes = np.zeros(n, dtype=np.int64)
        for k in range(1, n + 1):
            sizes[k - 1] = (labels == k).sum()
        idx_sorted = np.argsort(sizes)[::-1]
        largest_label = idx_sorted[0] + 1
        largest_size = sizes[idx_sorted[0]]

        # f0: n_blobs (raw count, will be z-scored)
        feats[b, 0] = float(n)

        # f1: largest_blob_area (normalized)
        feats[b, 1] = largest_size / (H * W)

        # f2: largest_blob_centrality (centroid distance from frame center,
        # normalized by half-diagonal). 0 = at center, 1 = at corner.
        ys, xs = np.where(labels == largest_label)
        if len(ys) > 0:
            cy_blob = ys.mean()
            cx_blob = xs.mean()
            feats[b, 2] = np.sqrt((cy_blob - cy) ** 2
                                    + (cx_blob - cx) ** 2) / diag
            # f3: aspect ratio (log of bbox width / height)
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            h = (y_max - y_min) + 1
            w = (x_max - x_min) + 1
            feats[b, 3] = np.log((w + 1.0) / (h + 1.0))

        # f4: blob_size_entropy (Shannon entropy of blob-size distribution)
        if n > 1:
            p = sizes.astype(np.float64)
            p = p / p.sum()
            p = p[p > 0]
            feats[b, 4] = -float((p * np.log(p)).sum())
        else:
            feats[b, 4] = 0.0

    return feats


def main():
    from physics_prior import blood_probability

    print(f"[c1-topo] device={DEVICE}")
    ds = CapsuleAllFramesDataset()
    n = len(ds)
    print(f"[c1-topo] {n} frames")
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
            scalar_f = compute_scalar_features(p_blood).cpu().numpy()
            topo_f = compute_topology_features(p_blood)
            feat = np.concatenate([scalar_f, topo_f], axis=1)
            bs = x.size(0)
            features[pos:pos + bs] = feat
            for k in range(bs):
                filenames[pos + k] = fnames[k]
            labels[pos:pos + bs] = ys.numpy()
            splits[pos:pos + bs] = ss.numpy()
            pos += bs
            if pos % (BATCH_SIZE * 50) == 0:
                rate = pos / max(0.001, (time.time() - t0))
                eta = (n - pos) / max(0.001, rate) / 60
                print(f"[c1-topo] {pos}/{n}  rate={rate:.0f} fps  eta={eta:.1f} min")

    elapsed = (time.time() - t0) / 60
    print(f"[c1-topo] done in {elapsed:.1f} min")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH,
                          filenames=np.array(filenames),
                          labels=labels, splits=splits,
                          features=features,
                          feature_names=np.array(FEATURE_NAMES))
    print(f"[c1-topo] saved to {OUTPUT_PATH} "
          f"(size = {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")

    print("\n[c1-topo] feature distributions (mean +- std across all 47K frames):")
    for i, name in enumerate(FEATURE_NAMES):
        col = features[:, i]
        print(f"  {name:30s}: {col.mean():.3f} +- {col.std():.3f}  "
              f"min={col.min():.3f}  max={col.max():.3f}")


if __name__ == "__main__":
    main()
