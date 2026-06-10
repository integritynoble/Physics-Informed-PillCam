"""
ACDC cardiac MRI cine -- preprocessing for Track B replication.
================================================================

Extracts 2D slices from the 4D NIfTI files in the ACDC challenge
dataset and organizes them in a Kvasir-Capsule-like train/val/test
directory layout so we can reuse the existing temporal pipeline
(`build_embedding_cache.py`, `build_c3_features.py`,
`train_cell_b.py`, etc.) with minimal code changes.

ACDC source layout (2017 challenge):
    {ACDC_RAW}/training/patient001/
        patient001_4d.nii.gz                   <- the 4D cine (3D + cycle)
        Info.cfg                               <- metadata (Group: NOR/MINF/DCM/HCM/RV)
    {ACDC_RAW}/training/patient002/
        ...
    {ACDC_RAW}/testing/                        <- (smaller test set, 50 patients)

Output layout (matches Kvasir-Capsule convention):
    D:/acdc/stage2_data/train/<class>/<filename>.jpg
    D:/acdc/stage2_data/val/<class>/<filename>.jpg
    D:/acdc/stage2_data/test/<class>/<filename>.jpg

Per-frame filenames encode patient_id and (slice, time) for the
temporal index:
    <patient_id>_s<slice_idx>_t<time_idx>.jpg
e.g.,
    patient034_s05_t12.jpg

JPG (quality 95) is used for compatibility with the capsule
pipeline's `.jpg`-only filesystem walker (`build_embedding_cache.py`).
Quality 95 preserves perceptual detail and is approximately
lossless for the percentile-clipped 8-bit grayscale that we
extract from each NIfTI slice.

This filename pattern is what `build_video_index.py` will key on
when building the cardiac-MRI sequence index. We treat each
(patient, slice) pair as a "video" and the cardiac cycle phase as
the temporal axis.

CLI:
    python preprocess_acdc.py \\
        --acdc_raw D:/acdc/raw \\
        --out D:/acdc/stage2_data \\
        --train_frac 0.70 --val_frac 0.15

The split is patient-level (no leak), stratified by class group.

Compute: ~30 min on CPU for the full ACDC dataset.

Dependencies:
    pip install nibabel pillow scikit-image numpy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ACDC class taxonomy (5 groups in the 2017 challenge)
ACDC_CLASS_NAMES = ["NOR", "MINF", "DCM", "HCM", "RV"]
ACDC_FULL_NAMES = {
    "NOR": "Normal",
    "MINF": "Myocardial infarction",
    "DCM": "Dilated cardiomyopathy",
    "HCM": "Hypertrophic cardiomyopathy",
    "RV": "Abnormal right ventricle",
}


def parse_info_cfg(info_path: Path) -> Dict[str, str]:
    """Parse ACDC's `Info.cfg` (one key:value per line, colon separator)."""
    out: Dict[str, str] = {}
    text = info_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def load_4d_nifti(path: Path) -> np.ndarray:
    """Returns a (X, Y, Z, T) array of float32 intensities."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("Need nibabel: pip install nibabel") from exc
    img = nib.load(str(path))
    arr = img.get_fdata().astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., None]  # treat as single time-step
    return arr


def normalize_slice(slc: np.ndarray) -> np.ndarray:
    """Per-slice 1st-99th percentile normalization → [0, 1]."""
    lo = np.percentile(slc, 1.0)
    hi = np.percentile(slc, 99.0)
    if hi <= lo:
        return np.zeros_like(slc, dtype=np.float32)
    s = np.clip((slc - lo) / (hi - lo), 0.0, 1.0)
    return s.astype(np.float32)


def slice_to_rgb(slc: np.ndarray, image_size: int = 224) -> np.ndarray:
    """Resize and replicate grayscale to 3-channel RGB in [0, 255] uint8."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Need pillow: pip install pillow") from exc
    norm = normalize_slice(slc)
    img = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    rgb = np.stack([np.asarray(img)] * 3, axis=-1)
    return rgb


def split_patients_by_class(patient_groups: Dict[str, str],
                             train_frac: float, val_frac: float,
                             seed: int = 42
                             ) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Patient-level stratified split. Returns (patient_id -> split,
    per-class split counts)."""
    rng = random.Random(seed)
    by_class: Dict[str, List[str]] = {c: [] for c in ACDC_CLASS_NAMES}
    for pid, cname in patient_groups.items():
        if cname not in by_class:
            continue
        by_class[cname].append(pid)

    pid_split: Dict[str, str] = {}
    counts = {f"{c}_{s}": 0 for c in ACDC_CLASS_NAMES
              for s in ("train", "val", "test")}
    for cname, pids in by_class.items():
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        for pid in pids[:n_train]:
            pid_split[pid] = "train"
            counts[f"{cname}_train"] += 1
        for pid in pids[n_train:n_train + n_val]:
            pid_split[pid] = "val"
            counts[f"{cname}_val"] += 1
        for pid in pids[n_train + n_val:]:
            pid_split[pid] = "test"
            counts[f"{cname}_test"] += 1
    return pid_split, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_raw", required=True,
                    help="ACDC raw dataset directory (containing training/ subfolder)")
    ap.add_argument("--out", required=True,
                    help="Output directory; will create train/val/test/<class>/ subfolders")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_testing_subset",
                    action="store_true",
                    help="If set, also extract the ACDC 'testing' subset; "
                         "those patients have group labels and can be folded "
                         "into our test split. Default: training-only.")
    args = ap.parse_args()

    raw = Path(args.acdc_raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources = [raw / "training"]
    if args.include_testing_subset and (raw / "testing").is_dir():
        sources.append(raw / "testing")

    # Pass 1: enumerate patients and read class groups
    patient_groups: Dict[str, str] = {}
    patient_dirs: Dict[str, Path] = {}
    for src in sources:
        for pdir in sorted(src.iterdir()):
            if not pdir.is_dir() or not pdir.name.startswith("patient"):
                continue
            info_path = pdir / "Info.cfg"
            if not info_path.exists():
                continue
            info = parse_info_cfg(info_path)
            group = info.get("Group", "").strip().upper()
            if group not in ACDC_CLASS_NAMES:
                print(f"[skip] {pdir.name}: unrecognized group '{group}'")
                continue
            patient_groups[pdir.name] = group
            patient_dirs[pdir.name] = pdir
    print(f"[acdc] {len(patient_groups)} patients with valid Group label")

    # Pass 2: patient-level stratified split
    pid_split, counts = split_patients_by_class(
        patient_groups, args.train_frac, args.val_frac, args.seed)
    print("[acdc] split (patient counts):")
    for c in ACDC_CLASS_NAMES:
        print(f"  {c}: train={counts[f'{c}_train']}  "
              f"val={counts[f'{c}_val']}  test={counts[f'{c}_test']}")

    # Pre-create class folders in each split
    for split in ("train", "val", "test"):
        for cname in ACDC_CLASS_NAMES:
            (out / split / cname).mkdir(parents=True, exist_ok=True)

    # Pass 3: extract slices
    n_frames_total = 0
    metadata = {
        "n_patients": len(patient_groups),
        "patient_split": pid_split,
        "class_counts": counts,
        "frames": [],
    }
    for pid, cname in sorted(patient_groups.items()):
        pdir = patient_dirs[pid]
        split = pid_split[pid]
        nii = pdir / f"{pid}_4d.nii.gz"
        if not nii.exists():
            print(f"[skip] {pid}: no 4d file")
            continue
        try:
            arr = load_4d_nifti(nii)
        except Exception as exc:
            print(f"[skip] {pid}: load error {exc}")
            continue
        X, Y, Z, T = arr.shape

        # Iterate over (z slice, t time) -- this is our (slice, frame_number)
        # pair for the temporal index.
        n_frames_pat = 0
        for z in range(Z):
            for t in range(T):
                slc = arr[:, :, z, t]
                rgb = slice_to_rgb(slc, image_size=args.image_size)
                fname = f"{pid}_s{z:02d}_t{t:02d}.jpg"
                fpath = out / split / cname / fname
                try:
                    from PIL import Image
                    Image.fromarray(rgb).save(fpath, "JPEG", quality=95)
                except Exception as exc:
                    print(f"[err] save {fpath}: {exc}")
                    continue
                metadata["frames"].append({
                    "filename": fname,
                    "patient_id": pid,
                    "slice": z,
                    "time": t,
                    "class": cname,
                    "split": split,
                })
                n_frames_pat += 1
        n_frames_total += n_frames_pat
        if len(metadata["frames"]) % 5000 < n_frames_pat:
            print(f"[acdc]  {pid} ({cname} {split}): {n_frames_pat} frames")
    print(f"[acdc] {n_frames_total} total frames extracted")

    # Save metadata for build_video_index.py to consume
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[acdc] metadata -> {meta_path}")


if __name__ == "__main__":
    main()
