"""
Build per-frame embedding cache from the v2 +PI input-fusion checkpoints.
=========================================================================

The v2 paper's "input-fusion" arm uses a 5-channel input
(RGB + 2 prior channels) and a backbone with the first conv layer
expanded to accept the extra channels. The 6 seeds are at:
  D:/kvasir_capsule/outputs/stage2_pi_effb0[_seed{seed}]/best_model.pt

This script extracts the per-frame pooled feature (1280-d) from each
+PI backbone and saves a cache compatible with `train_cell_b.py`.

Spatial-channel C1 hypothesis: the v2 input-fusion arm gave the
strongest v2 macro-AUC lift (+0.023). If we feed those embeddings
into the temporal transformer (cell b architecture), we test whether
a spatial-channel C1 prior + temporal aggregation outperforms
RGB-only embeddings + temporal (cell b on RGB embeddings = 0.7788).

Output:
  D:/kvasir_capsule/outputs/embeddings_pi/seed{seed}_embeddings.npz
    same format as RGB embeddings (filenames, labels, splits, embeddings)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
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
OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs")
EMB_DIR = OUTPUT_ROOT / "embeddings_pi"
EMB_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [41, 42, 43, 44, 45, 47]
BATCH_SIZE = 32
IMAGE_SIZE = 224

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
SPLIT_TO_INT = {"train": 0, "val": 1, "test": 2}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def output_dir_for(seed: int) -> Path:
    if seed == 42:
        return OUTPUT_ROOT / "stage2_pi_effb0"
    return OUTPUT_ROOT / f"stage2_pi_effb0_seed{seed}"


class AllFramesDataset(Dataset):
    """Returns (3-channel ImageNet-normalized frame, filename, label, split).
    The +PI model will compute the analytic prior internally and prepend it
    to form the 5-channel input. Here we just provide RGB; the wrapper in
    `extract_embeddings_for_seed` handles the prior + concat."""

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


def make_pi_input(x_rgb_norm: torch.Tensor, args_dict: Dict) -> torch.Tensor:
    """Construct the 5-channel input the +PI checkpoint expects:
    RGB (already normalized) concatenated with 2 prior channels.
    The 2 prior channels are P_blood and Phi (the radial-fluence map)
    per the v2 paper's input-fusion arm."""
    from physics_prior import blood_probability, radial_fluence
    x_rgb = unnormalize(x_rgb_norm.float())   # to [0,1]
    p_blood = blood_probability(x_rgb,
                                  alpha=args_dict.get("physics_alpha", 4.0),
                                  lambda_eff=args_dict.get("physics_lambda_eff",
                                                             None))
    # The v2 input-fusion arm uses 2 extra channels: [P_blood, Phi]
    # If args_dict has only one extra channel (P_blood), this matches.
    # Inspect the checkpoint's first conv weight shape to determine
    # the actual number of extra channels.
    return torch.cat([x_rgb_norm, p_blood.unsqueeze(1)], dim=1)


def extract_embeddings_for_seed(seed: int, dataset: AllFramesDataset
                                 ) -> Dict[str, np.ndarray]:
    out_dir = output_dir_for(seed)
    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing: {ckpt_path}")

    print(f"[seed {seed}] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_args = ckpt["args"]
    class_names = ckpt["class_names"]
    if class_names != CLASS_NAMES:
        raise SystemExit(f"class_names mismatch on seed {seed}")

    extra_channels = int(model_args.get("extra_channels", 2))
    print(f"[seed {seed}] extra_channels={extra_channels} (5-ch model? {3+extra_channels})")

    from models_pi import ImageClassifierPI
    model = ImageClassifierPI(
        model_args["model_name"], num_classes=len(class_names),
        pretrained=False, extra_channels=extra_channels).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    n = len(dataset)

    # Determine feature dim. The +PI input-fusion model expects
    # `physics_channels(rgb01)` which returns a (B, extra_channels, H, W)
    # tensor concatenated to the RGB after normalization is reversed.
    from physics_prior import physics_channels
    sample_x, _, _, _ = dataset[0]
    sample_x = sample_x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x_rgb = unnormalize(sample_x.float())
        phys = physics_channels(x_rgb,
                                  alpha=model_args.get("physics_alpha", 4.0),
                                  lambda_eff=model_args.get("physics_lambda_eff", None),
                                  version=model_args.get("physics_prior_version", "v1"),
                                  pivot_v2=model_args.get("physics_pivot_v2", 0.30))
        # phys has (B, extra_channels, H, W) -- expected to match `extra_channels`
        if phys.shape[1] != extra_channels:
            raise SystemExit(f"physics_channels gave {phys.shape[1]} channels, "
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
            phys = physics_channels(x_rgb,
                                      alpha=model_args.get("physics_alpha", 4.0),
                                      lambda_eff=model_args.get("physics_lambda_eff", None),
                                      version=model_args.get("physics_prior_version", "v1"),
                                      pivot_v2=model_args.get("physics_pivot_v2", 0.30))
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
    args = ap.parse_args()

    seeds = args.only_seeds or SEEDS
    print(f"[main] device={DEVICE}  seeds={seeds}  source=+PI input-fusion")

    dataset = AllFramesDataset()
    print(f"[main] dataset has {len(dataset)} frames")

    t_start = time.time()
    for seed in seeds:
        out_path = EMB_DIR / f"seed{seed}_embeddings.npz"
        if out_path.exists():
            print(f"[seed {seed}] cached at {out_path}, skipping")
            continue
        cache = extract_embeddings_for_seed(seed, dataset)
        np.savez_compressed(out_path, **cache)
        print(f"[seed {seed}] saved {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\n[main] all seeds done in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
