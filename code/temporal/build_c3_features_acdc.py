"""
Build per-frame C3 autoencoder-residual features for ACDC.
============================================================

ACDC analogue of `build_c3_features.py`. Loads the ACDC-trained
NOR-class autoencoder (from `train_ae_acdc.py`) and computes per-frame
residual features `r_t = x_t - AE(x_t)`, summarized via 8x8 spatial
average pool to a 192-d feature vector.

Output:
    D:/acdc/outputs/c3_features.npz
        filenames:   (N,)
        features:    (N, 192) float32
        labels:      (N,)         # 0..4 (NOR/MINF/DCM/HCM/RV)
        splits:      (N,)         # 0=train, 1=val, 2=test

Compute: ~1-3 min on RTX 5090 (ACDC is small).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

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
REPO_ROOT = HERE.parent.parent.parent.parent
GASTRO = Path("D:/onedrive/UT_southwestern/GIproject/Dr. Zaman/"
              "gastroscopy_code_package (2)/gastroscopy_code_package")
sys.path.insert(0, str(GASTRO))

DATA_ROOT = Path("D:/acdc/stage2_data")
AE_CKPT = Path("D:/acdc/outputs/ae/best_ae.pt")
OUTPUT_PATH = Path("D:/acdc/outputs/c3_features.npz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 32

CLASS_NAMES = ["NOR", "MINF", "DCM", "HCM", "RV"]
SPLIT_INT = {"train": 0, "val": 1, "test": 2}


def conv_block(in_c: int, out_c: int, downsample: bool = True) -> nn.Module:
    layers = [
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if downsample:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def upconv_block(in_c: int, out_c: int) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class SmallAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = conv_block(3, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.enc4 = conv_block(64, 128)
        self.enc5 = conv_block(128, 64, downsample=True)
        self.dec1 = upconv_block(64, 128)
        self.dec2 = upconv_block(128, 64)
        self.dec3 = upconv_block(64, 32)
        self.dec4 = upconv_block(32, 16)
        self.dec5 = upconv_block(16, 8)
        self.out = nn.Conv2d(8, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc1(x); z = self.enc2(z); z = self.enc3(z)
        z = self.enc4(z); z = self.enc5(z)
        h = self.dec1(z); h = self.dec2(h); h = self.dec3(h)
        h = self.dec4(h); h = self.dec5(h)
        return self.out(h)


class AllFramesDataset(Dataset):
    def __init__(self, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        self.samples: List[Tuple[Path, str, int, int]] = []
        for split in ("train", "val", "test"):
            sd = DATA_ROOT / split
            if not sd.is_dir():
                continue
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


def main():
    print(f"[c3-acdc] device={DEVICE}")
    if not AE_CKPT.exists():
        print(f"[c3-acdc] FATAL: AE checkpoint not found at {AE_CKPT}. "
              f"Run train_ae_acdc.py first.")
        sys.exit(1)
    print(f"[c3-acdc] loading AE from {AE_CKPT}")
    ckpt = torch.load(AE_CKPT, map_location=DEVICE, weights_only=False)
    ae = SmallAutoencoder().to(DEVICE)
    ae.load_state_dict(ckpt["model_state"])
    ae.eval()
    print(f"[c3-acdc] AE epoch {ckpt['epoch']}  val_l1={ckpt['val_l1']:.4f}")

    ds = AllFramesDataset()
    n = len(ds)
    print(f"[c3-acdc] {n} frames")
    if n == 0:
        print(f"[c3-acdc] FATAL: no frames in {DATA_ROOT}. "
              f"Run preprocess_acdc.py first.")
        sys.exit(1)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    features = np.zeros((n, 192), dtype=np.float32)
    filenames: List[str] = [""] * n
    labels = np.zeros((n,), dtype=np.int64)
    splits = np.zeros((n,), dtype=np.int64)

    pos = 0
    t0 = time.time()
    with torch.no_grad():
        for x, fnames, ys, ss in loader:
            x = x.to(DEVICE, non_blocking=True)
            recon = ae(x)
            r = (x - recon).abs()
            pooled = F.adaptive_avg_pool2d(r, (8, 8))
            feat = pooled.reshape(pooled.size(0), -1).cpu().numpy()
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
                print(f"[c3-acdc] {pos}/{n}  rate={rate:.0f} fps  eta={eta_min:.1f} min")

    elapsed = (time.time() - t0) / 60
    print(f"[c3-acdc] done in {elapsed:.1f} min")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH,
                          filenames=np.array(filenames),
                          labels=labels, splits=splits,
                          features=features)
    print(f"[c3-acdc] saved to {OUTPUT_PATH} "
          f"(size = {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")

    for cidx, cname in enumerate(CLASS_NAMES):
        mask = labels == cidx
        if mask.sum() == 0:
            continue
        m = features[mask].mean()
        print(f"  {cname:6s}: mean(|r|) = {m:.4f}  n={mask.sum()}")


if __name__ == "__main__":
    main()
