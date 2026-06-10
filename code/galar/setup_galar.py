"""Stage Galar dataset frames into an ImageFolder layout for zero-shot evaluation.

Refreshed 2026-05-15 based on the downloaded Galar metadata + 80 per-video
Labels/N.csv files. Galar's actual format on disk is:

    galar_raw/
    ├── metadata.csv                 (per-video: File Name, Capsule System, Gender, Age)
    ├── Labels/
    │   ├── 1.csv                    (one CSV per video; frame-by-frame multi-label one-hot)
    │   ├── 2.csv
    │   ├── ...
    │   └── 80.csv
    └── Frames/                      (extracted from Galar_Frames_*.7z archives)
        └── <video_id>/<frame>.{jpg|png}     (verify the exact path structure on first extract)

Two CSV header variants exist:
  - 74 videos use 27 cols: index, z-line, pylorus, ampulla of vater, ileocecal valve,
    section, mouth, esophagus, stomach, small intestine, colon,
    ulcer, polyp, active bleeding, blood, erythema, erosion, angiectasia,
    IBD, foreign body, esophagitis, varices, hematin, celiac, cancer,
    lymphangioectasis, frame
  - 6 videos use 32 cols (adds 'bubbles, dirt, no view, reduced view, good view'
    at the front, then the same 27 cols)

The script reads the per-video CSVs, maps Galar columns to Kvasir-Capsule
classes via galar_class_mapping.json, optionally filters by capsule system
and anatomical section, and stages frames into:

    out_dir/test/
    ├── Angiectasia/
    ├── Blood - fresh/
    ├── Blood - hematin/
    ├── Erosion/
    ├── Erythema/
    ├── Foreign Body/
    ├── Ileocecal valve/
    ├── Lymphangiectasia/
    ├── Normal clean mucosa/
    ├── Polyp/
    ├── Pylorus/
    ├── Ulcer/
    └── Ampulla of Vater/

Multi-label frames (positive for multiple Kvasir-evaluable classes) are
assigned to the first claim by class iteration order, to keep ImageFolder's
single-label semantics. The multi-label nature is preserved by recording
the original Galar row's full label set in a metadata sidecar
(out_dir/test/_multilabel_index.csv) for downstream analysis.

USAGE
    python setup_galar.py \\
        --galar_root  /home2/s248103/galar_raw \\
        --out_dir     /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/data/galar_eval \\
        --mapping     /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/galar/galar_class_mapping.json \\
        --mode        hardlink

    python setup_galar.py ... --capsule_subset pillcam   # only PillCam videos (within-vendor)
    python setup_galar.py ... --capsule_subset olympus   # only Olympus videos (cross-vendor)
    python setup_galar.py ... --max_per_class 5000       # cap Normal clean mucosa for class balance
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
import re


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--galar_root", type=Path, required=True,
                   help="Directory containing metadata.csv + Labels/*.csv + Frames/* "
                        "(Frames/ produced by extracting Galar_Frames_*.7z archives)")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output staging directory")
    p.add_argument("--mapping", type=Path, required=True,
                   help="Path to galar_class_mapping.json")
    p.add_argument("--frames_subdir", default="Frames",
                   help="Name of the frames subdirectory inside galar_root (default: Frames)")
    p.add_argument("--split_name", default="test",
                   help="Name of the output split dir (default: test). Galar has no pre-defined "
                        "train/val/test split; we stage everything under one split for zero-shot.")
    p.add_argument("--mode", choices=["hardlink", "symlink", "copy"], default="hardlink",
                   help="How to materialize frames in out_dir")
    p.add_argument("--capsule_subset", choices=["all", "pillcam", "olympus"], default="all",
                   help="Stratify by capsule system (default: all). pillcam = SB3/SB2/Colon2; "
                        "olympus = Olympus only.")
    p.add_argument("--max_per_class", type=int, default=10000,
                   help="Maximum frames to keep per Kvasir class (default: 10000). "
                        "Caps the abundant Normal clean mucosa for class balance.")
    p.add_argument("--keep_section_filter_lesions", action="store_true",
                   help="Restrict lesion-class frames to section=='small intestine' "
                        "(matches Kvasir-Capsule scope strictly). Default: keep all sections.")
    p.add_argument("--require_good_view", action="store_true",
                   help="On the 6 videos with view-quality labels, drop frames not marked good_view. "
                        "On the other 74 videos, no filter is applied. Default: off.")
    p.add_argument("--video_ids", default=None,
                   help="Comma-separated subset of video IDs to process (e.g. '1,2,5,10'). "
                        "Default: all videos that have label CSV files present.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print planned counts without copying anything")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for any subsampling")
    return p.parse_args()


def load_mapping(mapping_path: Path) -> dict:
    return json.loads(mapping_path.read_text())


def _galar_label_to_kvasir(mapping: dict) -> dict[str, str]:
    """Build a dict from Galar column name (lowercase) -> Kvasir class name."""
    out = {}
    for row in mapping["_galar_to_kvasir_mapping_table"]:
        out[row["galar_column"].lower()] = row["kvasir_class"]
    return out


def _select_videos(meta_csv: Path, capsule_subset: str,
                   subset_ids: list[int] | None) -> set[int]:
    """Return set of video IDs to process based on capsule system filter."""
    keep_systems = None
    if capsule_subset == "pillcam":
        keep_systems = {"PillCam SB3", "PillCam SB2", "PillCam Colon2"}
    elif capsule_subset == "olympus":
        keep_systems = {"Olympus"}

    selected = set()
    with meta_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vid = int(row["File Name"])
            except (KeyError, ValueError):
                continue
            if subset_ids is not None and vid not in subset_ids:
                continue
            cap = (row.get("Capsule System") or "").strip()
            if keep_systems is None or cap in keep_systems:
                selected.add(vid)
    return selected


_PATHOLOGY_COLS = (
    "ulcer", "polyp", "active bleeding", "blood", "erythema",
    "erosion", "angiectasia", "IBD", "foreign body",
    "esophagitis", "varices", "hematin", "celiac", "cancer",
    "lymphangioectasis",
)
_NON_LESION_CLASSES = {"Ileocecal valve", "Pylorus", "Ampulla of Vater"}


def _classify_row(row: dict, _mapping: dict, gal2k: dict[str, str],
                  keep_section_filter_lesions: bool, require_good_view: bool) -> list[str]:
    """Decide which Kvasir class(es) this Galar frame row is positive for.
    Fast path: Galar columns are already lowercase as-stored, so we do a
    single direct dict lookup per label.
    """
    if require_good_view:
        gv = row.get("good view")
        if gv is not None and gv != "1":
            return []

    section = (row.get("section") or "").strip().lower()

    matched_classes: list[str] = []
    for gal_col, kvasir in gal2k.items():
        if row.get(gal_col) == "1":
            if keep_section_filter_lesions and kvasir not in _NON_LESION_CLASSES:
                if section != "small intestine":
                    continue
            matched_classes.append(kvasir)

    if matched_classes:
        return matched_classes

    # Derive Normal clean mucosa if no class claimed yet
    if section == "small intestine":
        for col in _PATHOLOGY_COLS:
            if row.get(col) == "1":
                return []
        return ["Normal clean mucosa"]
    return []


def _resolve_frame_path(galar_root: Path, frames_subdir: str,
                         video_id: int, frame_idx: int) -> Path | None:
    """Locate the frame file on disk. We try multiple common conventions until
    figshare/extraction reveals the exact path:
        Frames/<vid_id>/<frame_idx>.jpg
        Frames/<vid_id>/<frame_idx>.png
        Frames/<vid_id>/frame_<frame_idx>.jpg
        Frames/<vid_id>/<frame_idx:06d>.jpg
        Frames/video_<vid_id>/<frame_idx>.jpg
    """
    base = galar_root / frames_subdir
    # Galar 2024 figshare extracts as Frames/<vid>/frame_<idx:06d>.PNG (uppercase).
    # Earlier guesses (lowercase / no-prefix) kept as fall-throughs for safety.
    candidates = [
        base / str(video_id) / f"frame_{frame_idx:06d}.PNG",
        base / str(video_id) / f"frame_{frame_idx:06d}.png",
        base / str(video_id) / f"frame_{frame_idx:06d}.jpg",
        base / str(video_id) / f"frame_{frame_idx}.PNG",
        base / str(video_id) / f"frame_{frame_idx}.jpg",
        base / str(video_id) / f"{frame_idx:06d}.PNG",
        base / str(video_id) / f"{frame_idx:06d}.png",
        base / str(video_id) / f"{frame_idx:06d}.jpg",
        base / str(video_id) / f"{frame_idx}.jpg",
        base / str(video_id) / f"{frame_idx}.png",
        base / f"video_{video_id}" / f"frame_{frame_idx:06d}.PNG",
        base / f"video_{video_id}" / f"frame_{frame_idx:06d}.jpg",
        base / f"video_{video_id}" / f"{frame_idx}.jpg",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _place(src: Path, dst: Path, mode: str) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    if mode == "hardlink":
        try:
            os.link(src, dst); return True
        except OSError:
            pass
    if mode == "symlink":
        try:
            os.symlink(src, dst); return True
        except OSError:
            pass
    shutil.copy2(src, dst)
    return True


def main():
    args = parse_args()
    mapping = load_mapping(args.mapping)
    gal2k = _galar_label_to_kvasir(mapping)
    target_classes = mapping["_kvasir_evaluable_classes_target"]

    meta_csv = args.galar_root / "metadata.csv"
    labels_dir = args.galar_root / "Labels"
    if not meta_csv.is_file():
        raise SystemExit(f"missing metadata.csv at {meta_csv}")
    if not labels_dir.is_dir():
        raise SystemExit(f"missing Labels/ dir at {labels_dir}")

    subset_ids = None
    if args.video_ids:
        subset_ids = sorted(int(x) for x in args.video_ids.split(",") if x.strip())

    selected_videos = _select_videos(meta_csv, args.capsule_subset, subset_ids)
    print(f"[setup_galar] capsule_subset={args.capsule_subset}  -> {len(selected_videos)} videos selected")
    if subset_ids is not None:
        selected_videos &= set(subset_ids)
        print(f"[setup_galar] intersected with --video_ids -> {len(selected_videos)} videos remain")
    if not selected_videos:
        raise SystemExit("no videos to process after filters")

    out_split = args.out_dir / args.split_name
    out_split.mkdir(parents=True, exist_ok=True)
    for c in target_classes:
        (out_split / c).mkdir(parents=True, exist_ok=True)

    placed_per_class = Counter()
    missing_frame_files = 0
    multilabel_records: list[dict] = []
    rng_state = (args.seed * 1009) & 0xFFFFFFFF

    for vid_id in sorted(selected_videos):
        csv_path = labels_dir / f"{vid_id}.csv"
        if not csv_path.is_file():
            print(f"  WARN: missing label CSV for video {vid_id}: {csv_path}")
            continue
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                matched = _classify_row(
                    row, mapping, gal2k,
                    keep_section_filter_lesions=args.keep_section_filter_lesions,
                    require_good_view=args.require_good_view,
                )
                if not matched:
                    continue
                # Single-label placement: pick the first matched class to claim
                # the frame; record all matches for downstream multi-label
                # analysis.
                claim = matched[0]
                if placed_per_class[claim] >= args.max_per_class:
                    continue
                try:
                    frame_idx = int(row["frame"])
                except (KeyError, ValueError):
                    continue
                # In dry-run we skip filesystem stat()s entirely (saves
                # millions of Lustre roundtrips on a 3.5M-row pass).
                if args.dry_run:
                    placed_per_class[claim] += 1
                else:
                    src = _resolve_frame_path(args.galar_root, args.frames_subdir, vid_id, frame_idx)
                    if src is None:
                        missing_frame_files += 1
                        continue
                    dst = out_split / claim / f"v{vid_id:03d}_f{frame_idx:08d}{src.suffix}"
                    _place(src, dst, args.mode)
                    placed_per_class[claim] += 1
                multilabel_records.append({
                    "video_id": vid_id,
                    "frame": frame_idx,
                    "assigned_class": claim,
                    "all_matched_classes": "|".join(matched),
                    "section": (row.get("section") or "").strip(),
                })

    print(f"\n[setup_galar] placed frames per class (split={args.split_name}):")
    grand = 0
    for c in target_classes:
        n = placed_per_class.get(c, 0)
        print(f"  {c:25s} {n:>7d}")
        grand += n
    print(f"  {'TOTAL':25s} {grand:>7d}")
    if missing_frame_files:
        print(f"\n[setup_galar] WARNING: {missing_frame_files} frame files could not be located. "
              f"Verify the --frames_subdir layout (expected Frames/<vid>/<frame>.jpg or similar). "
              f"Inspect a sample with `find {args.galar_root / args.frames_subdir} -maxdepth 3 | head -20`")

    # Write the multi-label sidecar
    if multilabel_records and not args.dry_run:
        sidecar = out_split / "_multilabel_index.csv"
        with sidecar.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(multilabel_records[0].keys()))
            w.writeheader()
            w.writerows(multilabel_records)
        print(f"\n[setup_galar] wrote multi-label sidecar: {sidecar}")

    if args.dry_run:
        print("\n[setup_galar] DRY RUN — no files placed.")
    else:
        print(f"\n[setup_galar] DONE. Use --galar_test_dir {out_split} when running eval_zero_shot.py")


if __name__ == "__main__":
    sys.exit(main() or 0)
