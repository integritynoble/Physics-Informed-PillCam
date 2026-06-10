"""
Build per-frame embedding cache for Track B (NMI temporal-prior plan).
=======================================================================

Direction-2 / Week 2 of Month 1.

For each of the 6 RGB checkpoints (`stage2_rgb_effb0[_seed{seed}]`),
run forward inference over all 47,238 labeled Kvasir-Capsule frames
(train + val + test) and cache the 1280-dim pooled backbone feature
per frame.

These embeddings become the C2 input to the temporal sequence
transformer in cells (b)-(e). The cache also enables cell (a)
reproduction: a linear probe on the cached embeddings should match
the existing per-frame distill checkpoint's test macro-AUC within
±0.002.

Output:
    D:/kvasir_capsule/outputs/embeddings/seed{seed}_embeddings.npz
        filenames:  (N,) array of frame filenames
        labels:     (N,) int -- class index 0..13
        splits:     (N,) int -- 0=train, 1=val, 2=test
        embeddings: (N, 1280) float32

Compute: ~5-30 min per seed depending on JPEG-decode throughput on
GTX 1660 Ti; ~1 hour total for 6 seeds.

Usage:
    python build_embedding_cache.py                  # all 6 seeds
    python build_embedding_cache.py --only_seeds 42  # one seed
"""

from __future__ import annotations

import argparse
import json
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
EMB_DIR = OUTPUT_ROOT / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [41, 42, 43, 44, 45, 47]
BATCH_SIZE = 32
IMAGE_SIZE = 224

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
SPLIT_TO_INT = {"train": 0, "val": 1, "test": 2}


def output_dir_for(seed: int) -> Path:
    if seed == 42:
        return OUTPUT_ROOT / "stage2_rgb_effb0"
    return OUTPUT_ROOT / f"stage2_rgb_effb0_seed{seed}"


class AllFramesDataset(Dataset):
    """Walks D:/kvasir_capsule/stage2_data/{train,val,test}/<class>/<file>.jpg
    and yields (image_tensor, filename, label_idx, split_idx).

    Uses gastroscopy_code's build_transforms(224, train=False) to match
    the original v2 inference pipeline exactly. Without this, PIL's
    bicubic resize gives slightly different pixel values than
    torchvision's bilinear default, causing per-seed AUC drift of
    ~0.005 vs the published v2 numbers."""

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
        try:
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Need pillow") from exc

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from PIL import Image
        path, fname, label, split = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)  # already (3, 224, 224) ImageNet-normalized
        return x.float(), fname, label, split


def extract_embeddings_for_seed(seed: int, dataset: AllFramesDataset
                                 ) -> Dict[str, np.ndarray]:
    """Load the seed's RGB checkpoint, forward over the dataset,
    return cache dict."""
    out_dir = output_dir_for(seed)
    ckpt_path = out_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"[fatal] missing checkpoint: {ckpt_path}")

    print(f"[seed {seed}] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model_args = ckpt["args"]
    class_names = ckpt["class_names"]
    if class_names != CLASS_NAMES:
        raise SystemExit(
            f"[fatal] seed {seed} class_names mismatch with expected: "
            f"got {class_names}")

    from models import ImageClassifier
    model = ImageClassifier(model_args["model_name"],
                              num_classes=len(class_names),
                              pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # ImageClassifier.forward returns logits via head(backbone(x)).
    # We want the BACKBONE's pooled output (the 1280-dim vector going
    # into head). Easiest: hook into model.backbone's output.
    # Looking at gastroscopy_code/models.py, ImageClassifier uses a
    # torchvision EffNet-B0 with classifier replaced by Identity, and
    # its head is Sequential(Dropout, Linear). So backbone(x) returns
    # the pooled 1280-dim vector directly.
    #
    # Try: model.backbone(x) -> pooled feature (1280-dim).

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    n = len(dataset)

    # Determine feature dim by running one batch
    sample_x, _, _, _ = dataset[0]
    with torch.no_grad():
        feat0 = model.backbone(sample_x.unsqueeze(0).to(DEVICE))
        feat0 = feat0.flatten(1)
    feat_dim = int(feat0.shape[-1])
    print(f"[seed {seed}] feature dim = {feat_dim}, n_frames = {n}")

    embeddings = np.zeros((n, feat_dim), dtype=np.float32)
    filenames: List[str] = [""] * n
    labels = np.zeros((n,), dtype=np.int64)
    splits = np.zeros((n,), dtype=np.int64)

    t0 = time.time()
    pos = 0
    with torch.no_grad():
        for x, fnames, ys, ss in loader:
            x = x.to(DEVICE, non_blocking=True)
            f = model.backbone(x).flatten(1)
            bs = x.size(0)
            embeddings[pos:pos + bs] = f.cpu().numpy()
            for k in range(bs):
                filenames[pos + k] = fnames[k]
            labels[pos:pos + bs] = ys.numpy()
            splits[pos:pos + bs] = ss.numpy()
            pos += bs
            if pos % (BATCH_SIZE * 50) == 0:
                rate = pos / max(0.001, (time.time() - t0))
                eta_min = (n - pos) / max(0.001, rate) / 60
                print(f"[seed {seed}] {pos}/{n}  rate={rate:.1f} fps  eta={eta_min:.1f} min")

    elapsed = time.time() - t0
    print(f"[seed {seed}] done in {elapsed/60:.1f} min")

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
    print(f"[main] device={DEVICE}  seeds={seeds}")

    dataset = AllFramesDataset(image_size=IMAGE_SIZE)
    print(f"[main] dataset has {len(dataset)} frames")

    # Class distribution sanity check
    cls_counts = defaultdict(lambda: defaultdict(int))
    for _, _, lbl, sp in dataset.samples:
        cls_counts[CLASS_NAMES[lbl]][sp] += 1
    print("[main] dataset class x split counts:")
    for c in CLASS_NAMES:
        d = cls_counts[c]
        print(f"        {c:25s}  train={d[0]:5d}  val={d[1]:5d}  test={d[2]:5d}")

    t_start = time.time()
    for seed in seeds:
        out_path = EMB_DIR / f"seed{seed}_embeddings.npz"
        if out_path.exists():
            print(f"[seed {seed}] already cached at {out_path}, skipping")
            continue
        cache = extract_embeddings_for_seed(seed, dataset)
        np.savez_compressed(out_path, **cache)
        print(f"[seed {seed}] saved to {out_path} "
              f"(size = {out_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\n[main] all seeds complete in {(time.time() - t_start)/60:.1f} min")
    print(f"[main] cache directory: {EMB_DIR}")


if __name__ == "__main__":
    main()
