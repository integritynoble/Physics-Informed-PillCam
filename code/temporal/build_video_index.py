"""
Build video-level frame index from Kvasir-Capsule metadata.csv
==============================================================

Direction-2 / Week 1 Step 1.

Reads `D:/kvasir_capsule/raw/metadata.csv` (47,238 rows) and produces a
JSON cache that maps every video_id to a sorted list of (frame_number,
class, split) tuples. This is the temporal axis the per-frame v2
pipeline discards by construction; with it we can reconstruct any
N-frame window around any labeled frame.

Side products:
- An inventory: # videos, frames per split, frames per class, video
  durations
- A consistency check: every labeled frame in
  D:/kvasir_capsule/stage2_data/{train,val,test}/<class>/<filename>
  must have a matching row in metadata.csv

Output:
  paper/nature-machine-intelligence/code/temporal/video_index.json
  paper/nature-machine-intelligence/code/temporal/video_index_inventory.md

Usage:
  python build_video_index.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
META_CSV = Path("D:/kvasir_capsule/raw/metadata.csv")
SPLIT_ROOT = Path("D:/kvasir_capsule/stage2_data")

OUT_JSON = HERE / "video_index.json"
OUT_MD = HERE / "video_index_inventory.md"


def read_metadata(csv_path: Path) -> List[Dict]:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append({
                "filename": row["filename"],
                "video_id": row["video_id"],
                "frame_number": int(row["frame_number"]),
                "finding_class": row["finding_class"].strip(),
            })
    return rows


def build_split_lookup(split_root: Path) -> Dict[str, str]:
    """Walk train/val/test/<class>/<filename>.jpg and return
    {filename: split} mapping. Filenames are unique across the dataset."""
    out: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        split_dir = split_root / split
        if not split_dir.is_dir():
            print(f"[warn] missing split: {split_dir}")
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for f in class_dir.iterdir():
                if f.suffix.lower() == ".jpg":
                    out[f.name] = split
    return out


def main():
    print(f"[index] reading {META_CSV}")
    meta = read_metadata(META_CSV)
    print(f"[index] {len(meta)} metadata rows")

    print(f"[index] building split lookup from {SPLIT_ROOT}")
    split_of = build_split_lookup(SPLIT_ROOT)
    print(f"[index] {len(split_of)} files indexed across all splits")

    # Group metadata rows by video_id, sort by frame_number
    by_video: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    missing_in_split = 0
    extra_in_split = set(split_of.keys())

    for row in meta:
        fname = row["filename"]
        sp = split_of.get(fname)
        if sp is None:
            missing_in_split += 1
        else:
            extra_in_split.discard(fname)
        by_video[row["video_id"]].append(
            (row["frame_number"], row["finding_class"], sp or "unlabeled")
        )

    for vid in by_video:
        by_video[vid].sort(key=lambda t: t[0])

    # Inventory
    n_videos = len(by_video)
    frames_per_video = {vid: len(v) for vid, v in by_video.items()}
    split_of_video: Dict[str, set] = {}
    for vid, frames in by_video.items():
        sps = set(s for _, _, s in frames if s != "unlabeled")
        split_of_video[vid] = sps

    # Class distribution overall
    class_counts: Counter = Counter()
    for vid, frames in by_video.items():
        for _, c, _ in frames:
            class_counts[c] += 1

    # Class distribution by split
    by_class_split: Dict[Tuple[str, str], int] = Counter()
    for vid, frames in by_video.items():
        for _, c, sp in frames:
            by_class_split[(c, sp)] += 1

    # Frames-per-split totals
    split_totals = Counter(s for vid, frames in by_video.items()
                           for _, _, s in frames)

    # Video assignment to splits — verify each video is in exactly one
    multi_split_videos = []
    for vid, sps in split_of_video.items():
        if len(sps) > 1:
            multi_split_videos.append((vid, sps))

    # Save the JSON cache
    cache = {
        "n_videos": n_videos,
        "n_frames": len(meta),
        "frames_per_video": frames_per_video,
        "split_of_video": {vid: sorted(list(sps)) for vid, sps in split_of_video.items()},
        "class_counts_global": dict(class_counts),
        "class_counts_by_split": {f"{c}__{s}": n for (c, s), n in by_class_split.items()},
        "split_totals": dict(split_totals),
        "by_video": {vid: [{"frame_number": fn, "class": c, "split": sp}
                            for fn, c, sp in frames]
                     for vid, frames in by_video.items()},
    }
    OUT_JSON.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"[index] cache -> {OUT_JSON}")

    # Inventory markdown
    md = []
    md.append("# Kvasir-Capsule video index\n")
    md.append(f"**Source:** `{META_CSV}` (47,238 frames per the dataset paper)")
    md.append(f"**Date:** 2026-05-06\n")

    md.append("## Top-level counts\n")
    md.append("| Item | Value |")
    md.append("|---|---:|")
    md.append(f"| Total frames in metadata | {len(meta)} |")
    md.append(f"| Total videos | {n_videos} |")
    md.append(f"| Frames in train split | {split_totals.get('train', 0)} |")
    md.append(f"| Frames in val split | {split_totals.get('val', 0)} |")
    md.append(f"| Frames in test split | {split_totals.get('test', 0)} |")
    md.append(f"| Frames *not* in any split (unlabeled-context) | "
              f"{split_totals.get('unlabeled', 0)} |")
    md.append(f"| Files in stage2_data without metadata row | "
              f"{len(extra_in_split)} |")
    md.append(f"| Metadata rows without stage2_data file | {missing_in_split} |")
    md.append(f"| Videos that span >1 split (data leak) | {len(multi_split_videos)} |")
    md.append("")

    md.append("## Class distribution (global)\n")
    md.append("| Class | Count |")
    md.append("|---|---:|")
    for c, n in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        md.append(f"| {c} | {n} |")
    md.append("")

    md.append("## Class x split distribution\n")
    splits = ["train", "val", "test", "unlabeled"]
    md.append("| Class | " + " | ".join(splits) + " |")
    md.append("|---|" + "|".join(["---:"] * len(splits)) + "|")
    for c in sorted(class_counts.keys()):
        row = [c] + [str(by_class_split.get((c, s), 0)) for s in splits]
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## Video-length distribution\n")
    lens = sorted(frames_per_video.values())
    md.append("| Stat | Value |")
    md.append("|---|---:|")
    md.append(f"| min frames/video | {lens[0]} |")
    md.append(f"| median frames/video | {lens[len(lens)//2]} |")
    md.append(f"| max frames/video | {lens[-1]} |")
    md.append(f"| mean frames/video | {sum(lens)/len(lens):.0f} |")
    md.append("")

    if multi_split_videos:
        md.append("## WARNING: videos that span multiple splits (data leak)\n")
        for vid, sps in multi_split_videos[:20]:
            md.append(f"- {vid}: {sorted(sps)}")
        md.append("")

    md.append("## What this enables\n")
    md.append("With per-video frame ordering and class labels indexed,")
    md.append("a `SequenceDataset` can yield N-frame windows around any")
    md.append("labeled frame for sequence-aware temporal modeling")
    md.append("(Direction 2 of the NMI plan). Unlabeled-context frames")
    md.append("(those not in train/val/test but present in metadata.csv)")
    md.append("can serve as adjacent-frame context without leaking labels.")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[index] inventory -> {OUT_MD}")

    print("\n[summary]")
    print(f"  videos = {n_videos}")
    print(f"  total frames = {len(meta)}")
    print(f"  train / val / test / unlabeled = "
          f"{split_totals.get('train', 0)} / {split_totals.get('val', 0)} / "
          f"{split_totals.get('test', 0)} / {split_totals.get('unlabeled', 0)}")
    print(f"  multi-split videos (data leak) = {len(multi_split_videos)}")
    print(f"  files in splits without metadata row = {len(extra_in_split)}")


if __name__ == "__main__":
    main()
