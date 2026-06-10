"""
Train Normal-class autoencoder on ACDC NOR frames.
====================================================

ACDC analogue of `pilot9_autoencoder_c3b_verification.py` (training
phase only). Architecture identical: small encoder-decoder (UNet
without skip connections), bottleneck 7x7x64. Trained on NOR
(healthy myocardium) frames in the train split, with NOR val frames
for early stopping.

Compute: ~1-3 hr on RTX 5090 (ACDC has ~3-5x fewer Normal frames
than Kvasir-Capsule's "Normal clean mucosa" class).

Output:
    D:/acdc/outputs/ae/best_ae.pt
        {"model_state": ..., "epoch": int, "val_l1": float, "history": [...]}
"""

from __future__ import annotations

import argparse
import json
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

DATA_ROOT = Path("D:/acdc/stage2_data")
AE_OUTPUT = Path("D:/acdc/outputs/ae")
AE_OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = 224

CLASS_NAMES = ["NOR", "MINF", "DCM", "HCM", "RV"]
NORMAL_CLASS = "NOR"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ACDCFrameDataset(Dataset):
    """Walks D:/acdc/stage2_data/<split>/<class>/<file>.jpg.
    If `class_filter` is set, returns only frames in those classes."""

    def __init__(self, split: str, class_filter: List[str] | None = None,
                 image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        split_dir = DATA_ROOT / split
        self.samples: List[Tuple[Path, int]] = []
        if not split_dir.is_dir():
            return
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            cname = class_dir.name
            if class_filter is not None and cname not in class_filter:
                continue
            if cname not in CLASS_NAMES:
                continue
            cidx = CLASS_NAMES.index(cname)
            for f in class_dir.iterdir():
                if f.suffix.lower() == ".jpg":
                    self.samples.append((f, cidx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from PIL import Image
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x.float(), label


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
    """Encoder-decoder, NO skip connections. Bottleneck 7x7x64.
    Identical to pilot 9's capsule AE so that C3 features are
    architecturally comparable across modalities."""

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


def parameter_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"[ae-acdc] training small AE on NOR (healthy myocardium) train frames")
    print(f"[ae-acdc] data_root: {DATA_ROOT}")
    train_ds = ACDCFrameDataset(split="train", class_filter=[NORMAL_CLASS])
    val_ds = ACDCFrameDataset(split="val", class_filter=[NORMAL_CLASS])
    print(f"[ae-acdc] train={len(train_ds)} normal frames, val={len(val_ds)}")
    if len(train_ds) == 0:
        print(f"[ae-acdc] FATAL: no NOR frames in {DATA_ROOT}/train. "
              f"Run preprocess_acdc.py first.")
        sys.exit(1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = SmallAutoencoder().to(DEVICE)
    print(f"[ae-acdc] params: {parameter_count(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, _y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            recon = model(x)
            loss = F.l1_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_l1 = running / max(1, seen)

        model.eval()
        running_v, seen_v = 0.0, 0
        with torch.no_grad():
            for x, _y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                recon = model(x)
                loss = F.l1_loss(recon, x)
                running_v += loss.item() * x.size(0)
                seen_v += x.size(0)
        val_l1 = running_v / max(1, seen_v)
        scheduler.step()

        elapsed = (time.time() - t0) / 60
        print(f"[ae-acdc] epoch {epoch:2d}/{args.epochs}  train_L1={train_l1:.4f}  "
              f"val_L1={val_l1:.4f}  elapsed={elapsed:.1f} min")
        history.append({"epoch": epoch, "train_l1": train_l1, "val_l1": val_l1})
        if val_l1 < best_val:
            best_val = val_l1
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val_l1": val_l1,
                        "history": history},
                        AE_OUTPUT / "best_ae.pt")

    print(f"\n[ae-acdc] best val_L1 = {best_val:.4f}")
    print(f"[ae-acdc] checkpoint -> {AE_OUTPUT / 'best_ae.pt'}")


if __name__ == "__main__":
    main()
