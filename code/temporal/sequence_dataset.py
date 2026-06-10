"""
SequenceDataset -- N-frame temporal windows around labeled center frames
=========================================================================

Direction-2 / Week 1 Step 2.

For each labeled frame `(video_id, frame_number, class, split)` from
`video_index.json`, yield a window of N frames in the same video. The
window is sampled around the center frame; if the window extends past
the video's first or last labeled frame, we pad by repeating the
boundary frame.

Note on temporal granularity: the Kvasir-Capsule labeled dataset is a
sparse subset of each video's actual frame stream (capsule videos are
~30K frames; the labeled subset is ~1100 frames per video on average).
Within a video, consecutive entries in `video_index.json` are temporally
ordered but may have arbitrary time gaps. The model's positional
encoding can use either frame ordinal (1, 2, 3, ...) or absolute
`frame_number` (which preserves time-gap information).

Usage:
    from sequence_dataset import SequenceDataset, build_split
    train_ds = build_split("train", window=16, ...)

Outputs (one item):
    {
        "frames": (N, 3, 224, 224) float32  -- ImageNet-normalized RGB
        "frame_numbers": (N,) int           -- absolute frame_number per frame
        "label": int                        -- class index of the center frame
        "video_id": str
        "center_frame_number": int
        "split": str
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
INDEX_JSON = HERE / "video_index.json"
SPLIT_ROOT = Path("D:/kvasir_capsule/stage2_data")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Standard 14-class order matching existing v2 checkpoints
CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_index(index_path: Path = INDEX_JSON) -> Dict:
    return json.loads(index_path.read_text(encoding="utf-8"))


def filename_for(video_id: str, frame_number: int) -> str:
    return f"{video_id}_{frame_number}.jpg"


def find_image_path(filename: str, split_root: Path = SPLIT_ROOT
                     ) -> Optional[Path]:
    """Locate the JPG given just its filename. Walks
    SPLIT_ROOT/{train,val,test}/<class>/<filename>."""
    for split in ("train", "val", "test"):
        for class_dir in (split_root / split).iterdir():
            if not class_dir.is_dir():
                continue
            candidate = class_dir / filename
            if candidate.is_file():
                return candidate
    return None


class FilenameLookup:
    """One-time scan of stage2_data tree, gives O(1) lookup
    filename -> Path. Avoid the find_image_path linear scan per call."""

    def __init__(self, split_root: Path = SPLIT_ROOT):
        self.lookup: Dict[str, Path] = {}
        for split in ("train", "val", "test"):
            split_dir = split_root / split
            if not split_dir.is_dir():
                continue
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                for f in class_dir.iterdir():
                    if f.suffix.lower() == ".jpg":
                        self.lookup[f.name] = f

    def __getitem__(self, filename: str) -> Path:
        return self.lookup[filename]

    def __contains__(self, filename: str) -> bool:
        return filename in self.lookup


class SequenceDataset(Dataset):
    """Yields N-frame temporal windows around labeled center frames.

    Parameters
    ----------
    split : str
        "train", "val", or "test"
    window : int
        Number of frames in the temporal window (must be odd to have a
        well-defined center; if even, center is window//2 - 1)
    image_size : int
        Resize all frames to this size (square)
    use_train_augmentation : bool
        If True, apply train-time augmentations (matches stage2 pipeline)
    """

    def __init__(self,
                 split: str,
                 window: int = 16,
                 image_size: int = 224,
                 use_train_augmentation: bool = False,
                 index_path: Path = INDEX_JSON,
                 split_root: Path = SPLIT_ROOT):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        self.split = split
        self.window = window
        self.image_size = image_size
        self.split_root = split_root

        index = load_index(index_path)
        self.lookup = FilenameLookup(split_root)

        # For each video that contains frames in the requested split,
        # build the in-video sequence (only frames whose split matches).
        # The "center frames" we iterate over are exactly the labeled
        # frames from the requested split.
        self.windows: List[Dict] = []
        for video_id, frames in index["by_video"].items():
            split_frames = [f for f in frames if f["split"] == split]
            if not split_frames:
                continue
            # Sort by frame_number (already sorted in index, but be safe)
            split_frames.sort(key=lambda f: f["frame_number"])
            # The "context" we use for any center frame: ALL frames in this
            # split for this video, in temporal order. Filenames live in
            # stage2_data and are accessible by FilenameLookup.
            for i, center in enumerate(split_frames):
                # Build the window centered on i (with boundary padding)
                half = window // 2
                lo = max(0, i - half)
                hi = min(len(split_frames), i - half + window)
                # If we hit a boundary, expand the other side
                if hi - lo < window:
                    if lo == 0:
                        hi = min(len(split_frames), window)
                    elif hi == len(split_frames):
                        lo = max(0, len(split_frames) - window)
                window_frames = split_frames[lo:hi]
                # Pad by repeating boundary if still short (very short
                # videos in this split)
                while len(window_frames) < window:
                    window_frames.append(window_frames[-1])
                self.windows.append({
                    "video_id": video_id,
                    "center_idx_in_video_split": i,
                    "center_class": center["class"],
                    "center_frame_number": center["frame_number"],
                    "window_filenames": [
                        filename_for(video_id, f["frame_number"])
                        for f in window_frames],
                    "window_frame_numbers": [
                        f["frame_number"] for f in window_frames],
                    "split": split,
                })

        self.use_train_augmentation = use_train_augmentation

        # Defer the heavy PIL/torchvision imports
        try:
            from torchvision import transforms as T
            self._T = T
            if use_train_augmentation:
                self.transform = T.Compose([
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                ])
        except ImportError:
            raise RuntimeError("torchvision required: pip install torchvision")

        try:
            from PIL import Image
            self._Image = Image
        except ImportError:
            raise RuntimeError("Pillow required: pip install pillow")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:
        w = self.windows[idx]
        frames = []
        for fname in w["window_filenames"]:
            path = self.lookup[fname]
            img = self._Image.open(path).convert("RGB")
            tensor = self.transform(img)  # (3, H, W) in [0, 1]
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
            frames.append(tensor)
        frames_t = torch.stack(frames, dim=0)  # (N, 3, H, W)
        return {
            "frames": frames_t.float(),
            "frame_numbers": torch.tensor(w["window_frame_numbers"], dtype=torch.long),
            "label": CLASS_TO_IDX[w["center_class"]],
            "video_id": w["video_id"],
            "center_frame_number": w["center_frame_number"],
            "split": w["split"],
        }


# ---------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------

def _self_test():
    print("[test] building train SequenceDataset (window=16) ...")
    ds = SequenceDataset(split="train", window=16, image_size=224)
    print(f"[test] {len(ds)} train windows")
    print(f"[test] sample item:")
    item = ds[0]
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"        {k:25s}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"        {k:25s}: {v}")

    # Pick a Lymphangiectasia center frame and verify the window contains
    # nearby frame_numbers from the same video
    lym_idx = next((i for i in range(len(ds))
                     if ds.windows[i]["center_class"] == "Lymphangiectasia"),
                    None)
    if lym_idx is not None:
        item = ds[lym_idx]
        print(f"\n[test] Lymphangiectasia center idx={lym_idx}")
        print(f"        video_id = {item['video_id']}")
        print(f"        center frame_number = {item['center_frame_number']}")
        print(f"        window frame_numbers = {item['frame_numbers'].tolist()}")
        print(f"        label = {item['label']} ({CLASS_NAMES[item['label']]})")

    # Class distribution of train windows
    from collections import Counter
    counts = Counter(w["center_class"] for w in ds.windows)
    print("\n[test] train class distribution:")
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"        {c:25s}: {n}")


if __name__ == "__main__":
    _self_test()
