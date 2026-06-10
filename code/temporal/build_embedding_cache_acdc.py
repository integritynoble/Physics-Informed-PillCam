"""
Build per-frame embedding cache from ACDC RGB backbones.
=========================================================

ACDC analogue of `build_embedding_cache.py`. Forward each of the 6
ACDC-fine-tuned RGB checkpoints over every cardiac-MRI slice and
cache the 1280-d pooled feature per frame. These embeddings drive
cells (b)/(c)/(d)/(e) ACDC.

Pre-conditions:
  - `preprocess_acdc.py` has populated D:/acdc/stage2_data/{train,val,test}/<class>/*.jpg
  - 6 RGB backbones exist at D:/acdc/outputs/stage2_rgb_effb0[_seed{seed}]/best_model.pt
    (these are produced by the to-be-written backbone-training run on
    ACDC slices, ~1 GPU-day per seed on RTX 5090; same script as
    capsule's train_stage2_pi.py with extra_channels=0 and ACDC paths)

Output:
    D:/acdc/outputs/embeddings/seed{seed}_embeddings.npz
        filenames:  (N,) array of frame filenames
        labels:     (N,) int -- class index 0..4 (NOR/MINF/DCM/HCM/RV)
        splits:     (N,) int -- 0=train, 1=val, 2=test
        embeddings: (N, 1280) float32

Compute: ~5-15 min per seed on RTX 5090 (ACDC is ~3-5x smaller than
Kvasir-Capsule by frame count).
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
EMB_DIR = OUTPUT_ROOT / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [41, 42, 43, 44, 45, 47]
BATCH_SIZE = 32
IMAGE_SIZE = 224

CLASS_NAMES = ["NOR", "MINF", "DCM", "HCM", "RV"]
SPLIT_TO_INT = {"train": 0, "val": 1, "test": 2}


def output_dir_for(seed: int) -> Path:
    if seed == 42:
        return OUTPUT_ROOT / "stage2_rgb_effb0"
    return OUTPUT_ROOT / f"stage2_rgb_effb0_seed{seed}"


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


def extract_embeddings_for_seed(seed: int, dataset: AllFramesDataset
                                 ) -> Dict[str, np.ndarray]:
    out_dir = output_dir_for(seed)
    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"[fatal] missing checkpoint: {ckpt_path}. "
                          f"Train the ACDC RGB backbone for seed {seed} first.")

    print(f"[seed {seed}] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_args = ckpt["args"]
    class_names = ckpt["class_names"]
    if class_names != CLASS_NAMES:
        raise SystemExit(f"[fatal] seed {seed} class_names mismatch: "
                          f"expected {CLASS_NAMES}, got {class_names}")

    from models import ImageClassifier
    model = ImageClassifier(model_args["model_name"],
                              num_classes=len(class_names),
                              pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    n = len(dataset)
    print(f"[seed {seed}] feature dim probing...")
    sample_x = dataset[0][0].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat0 = model.backbone(sample_x)
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
            f = model.backbone(x)
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
    print(f"[main] device={DEVICE}  seeds={seeds}  source=ACDC RGB")

    dataset = AllFramesDataset()
    print(f"[main] dataset has {len(dataset)} frames")
    if len(dataset) == 0:
        print(f"[main] FATAL: no frames in {DATA_ROOT}. Run preprocess_acdc.py first.")
        sys.exit(1)

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
