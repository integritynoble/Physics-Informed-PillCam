"""
Build per-frame embedding cache from ACDC +PI input-fusion backbones.
======================================================================

ACDC analogue of `build_embedding_cache_pi.py`. Forward each of the 6
ACDC-fine-tuned +PI checkpoints (5-channel input: RGB + 2 prior
channels) over every cardiac-MRI slice and cache the 1280-d pooled
feature per frame.

For cardiac MRI the analytic prior is modality-specific. Two
options, controlled by `--prior_mode`:

  - `radial`: use only the radial-illumination map (`Φ`), zero out
    the blood probability. This is a simple isotropic prior tied to
    short-axis-slice geometry; defensible as "the same prior
    architecturally, modality-appropriate semantically."
  - `motion`: prior channels are inter-frame motion magnitude. Future
    work; not implemented in this builder.
  - `none`: zero out both prior channels and use only the +PI
    backbone weights as a (suboptimal) RGB-equivalent. Useful as a
    sanity-check that the +PI weights aren't trivially worse than RGB.

Default: `radial`.

Pre-conditions:
  - `preprocess_acdc.py` populated D:/acdc/stage2_data/
  - 6 +PI backbones at D:/acdc/outputs/stage2_pi_effb0[_seed{seed}]/best_model.pt
    (these are produced by capsule's train_stage2_pi.py with
    extra_channels=2 and ACDC paths; ~1 GPU-day per seed)

Output:
    D:/acdc/outputs/embeddings_pi/seed{seed}_embeddings.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
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

DATA_ROOT = Path("D:/acdc/stage2_data")
OUTPUT_ROOT = Path("D:/acdc/outputs")
EMB_DIR = OUTPUT_ROOT / "embeddings_pi"
EMB_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [41, 42, 43, 44, 45, 47]
BATCH_SIZE = 32
IMAGE_SIZE = 224

CLASS_NAMES = ["NOR", "MINF", "DCM", "HCM", "RV"]
SPLIT_TO_INT = {"train": 0, "val": 1, "test": 2}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def output_dir_for(seed: int) -> Path:
    if seed == 42:
        return OUTPUT_ROOT / "stage2_pi_effb0"
    return OUTPUT_ROOT / f"stage2_pi_effb0_seed{seed}"


class AllFramesDataset(Dataset):
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
                sidx = SPLIT_TO_INT[split]
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


def radial_fluence_map(B: int, H: int, W: int, device) -> torch.Tensor:
    """Isotropic radial-illumination map: 1 at center, ~0 at corners.
    Standard inverse-square-of-distance falloff. Shape (B, 1, H, W)."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device),
                              indexing="ij")
    r2 = xx ** 2 + yy ** 2
    phi = 1.0 / (1.0 + 4.0 * r2)
    phi = phi.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
    return phi.contiguous()


def cardiac_prior_channels(x_rgb: torch.Tensor, mode: str
                              ) -> torch.Tensor:
    """Build the 2-channel ACDC analytic prior. Returns (B, 2, H, W)
    in the same image space as `x_rgb` (which is in [0, 1] after
    `unnormalize`)."""
    B, _, H, W = x_rgb.shape
    device = x_rgb.device
    if mode == "radial":
        phi = radial_fluence_map(B, H, W, device)
        zero = torch.zeros_like(phi)
        return torch.cat([zero, phi], dim=1)
    elif mode == "none":
        zero = torch.zeros((B, 2, H, W), device=device)
        return zero
    else:
        raise ValueError(f"unknown prior_mode={mode}; expected radial|none")


def extract_embeddings_for_seed(seed: int, dataset: AllFramesDataset,
                                 prior_mode: str) -> Dict[str, np.ndarray]:
    out_dir = output_dir_for(seed)
    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing: {ckpt_path}. Train ACDC +PI backbone first.")

    print(f"[seed {seed}] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_args = ckpt["args"]
    class_names = ckpt["class_names"]
    if class_names != CLASS_NAMES:
        raise SystemExit(f"class_names mismatch on seed {seed}: "
                          f"expected {CLASS_NAMES}, got {class_names}")

    extra_channels = int(model_args.get("extra_channels", 2))
    print(f"[seed {seed}] extra_channels={extra_channels} "
          f"(input is {3+extra_channels}-channel)")

    from models_pi import ImageClassifierPI
    model = ImageClassifierPI(
        model_args["model_name"], num_classes=len(class_names),
        pretrained=False, extra_channels=extra_channels).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    n = len(dataset)

    sample_x, _, _, _ = dataset[0]
    sample_x = sample_x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_rgb = unnormalize(sample_x.float())
        phys = cardiac_prior_channels(x_rgb, prior_mode)
        if phys.shape[1] != extra_channels:
            raise SystemExit(f"prior gives {phys.shape[1]} channels but "
                             f"checkpoint expects {extra_channels}")
        x5 = torch.cat([sample_x, phys], dim=1)
        feat0 = model.backbone(x5)
    feat_dim = int(feat0.shape[-1])
    print(f"[seed {seed}] feature dim = {feat_dim}, n_frames = {n}")

    embeddings = np.zeros((n, feat_dim), dtype=np.float32)
    filenames: List[str] = [""] * n
    labels = np.zeros((n,), dtype=np.int64)
    splits = np.zeros((n,), dtype=np.int64)

    pos = 0
    t0 = time.time()
    with torch.no_grad():
        for x, fnames, ys, ss in loader:
            x = x.to(DEVICE, non_blocking=True)
            x_rgb = unnormalize(x.float())
            phys = cardiac_prior_channels(x_rgb, prior_mode)
            x_in = torch.cat([x, phys], dim=1)
            f = model.backbone(x_in)
            bs = x.size(0)
            embeddings[pos:pos + bs] = f.cpu().numpy()
            for k in range(bs):
                filenames[pos + k] = fnames[k]
            labels[pos:pos + bs] = ys.numpy()
            splits[pos:pos + bs] = ss.numpy()
            pos += bs
            if pos % (BATCH_SIZE * 50) == 0:
                rate = pos / max(0.001, (time.time() - t0))
                eta = (n - pos) / max(0.001, rate) / 60
                print(f"[seed {seed}] {pos}/{n}  rate={rate:.0f} fps  eta={eta:.1f} min")

    print(f"[seed {seed}] done in {(time.time() - t0)/60:.1f} min")
    return {
        "filenames": np.array(filenames),
        "labels": labels,
        "splits": splits,
        "embeddings": embeddings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_seeds", type=int, nargs="+", default=None)
    ap.add_argument("--prior_mode", type=str, default="radial",
                     choices=["radial", "none"],
                     help="Analytic prior used as the 2 extra channels.")
    args = ap.parse_args()

    seeds = args.only_seeds or SEEDS
    print(f"[main] device={DEVICE}  seeds={seeds}  prior_mode={args.prior_mode}")
    print(f"[main] source: ACDC +PI 5-channel input fusion")

    dataset = AllFramesDataset()
    print(f"[main] dataset has {len(dataset)} frames")
    if len(dataset) == 0:
        raise SystemExit(f"no frames in {DATA_ROOT}; run preprocess_acdc.py first")

    t_start = time.time()
    for seed in seeds:
        out_path = EMB_DIR / f"seed{seed}_embeddings.npz"
        if out_path.exists():
            print(f"[seed {seed}] cached at {out_path}, skipping")
            continue
        cache = extract_embeddings_for_seed(seed, dataset, args.prior_mode)
        np.savez_compressed(out_path, **cache)
        print(f"[seed {seed}] saved {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\n[main] all seeds done in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
