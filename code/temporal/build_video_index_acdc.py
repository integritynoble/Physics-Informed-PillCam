"""
Build per-(patient, slice) video index for ACDC cardiac MRI.
=============================================================

ACDC analogue of `build_video_index.py`. The capsule version keys
"video" by `video_id` from metadata.csv; here a "video" is a single
slice of a patient's cardiac cycle, indexed by `(patient_id, slice_idx)`.
The cardiac-phase index `t` is the frame ordering within each video.

Input:
  `D:/acdc/stage2_data/metadata.json` (produced by `preprocess_acdc.py`)

Output:
  `video_index_acdc.json` next to this script. Schema mirrors
  `video_index.json` but with the new keying:

  {
    "n_videos": int,                    # number of (patient, slice) pairs
    "n_frames": int,                    # total per-frame entries
    "split_totals": {train/val/test: int},
    "class_counts_global": {NOR: int, ...},
    "class_counts_by_split": {"NOR__train": int, ...},
    "split_of_video": {video_id: [splits]},
    "by_video": {
      "patient001_s05": [
         {"filename": "patient001_s05_t00.png",
          "frame_number": 0, "class": "DCM", "split": "train"},
         {"filename": "patient001_s05_t01.png",
          "frame_number": 1, "class": "DCM", "split": "train"},
         ...
      ],
      ...
    }
  }
"""

from __future__ import annotations

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
META_JSON = Path("D:/acdc/stage2_data/metadata.json")
OUT_JSON = HERE / "video_index_acdc.json"
OUT_MD = HERE / "video_index_acdc_inventory.md"


def main():
    if not META_JSON.exists():
        print(f"[acdc-index] FATAL: {META_JSON} not found. Run "
              f"`preprocess_acdc.py` first.")
        sys.exit(1)

    print(f"[acdc-index] reading {META_JSON}")
    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    frames = meta.get("frames", [])
    n_total = len(frames)
    print(f"[acdc-index] {n_total} per-frame entries")

    # Group by (patient_id, slice) -> sorted by time index
    by_video: Dict[str, List[Dict]] = defaultdict(list)
    for fr in frames:
        vid = f"{fr['patient_id']}_s{fr['slice']:02d}"
        by_video[vid].append({
            "filename": fr["filename"],
            "frame_number": int(fr["time"]),
            "class": fr["class"],
            "split": fr["split"],
        })
    for vid in by_video:
        by_video[vid].sort(key=lambda f: f["frame_number"])

    n_videos = len(by_video)

    # Inventory: split_of_video, class_counts, etc.
    split_of_video: Dict[str, List[str]] = {}
    multi_split_videos: List[Tuple[str, set]] = []
    for vid, fs in by_video.items():
        sps = set(f["split"] for f in fs)
        split_of_video[vid] = sorted(sps)
        if len(sps) > 1:
            multi_split_videos.append((vid, sps))

    class_counts: Counter = Counter()
    by_class_split: Counter = Counter()
    split_totals: Counter = Counter()
    for vid, fs in by_video.items():
        for f in fs:
            class_counts[f["class"]] += 1
            by_class_split[(f["class"], f["split"])] += 1
            split_totals[f["split"]] += 1

    cache = {
        "n_videos": n_videos,
        "n_frames": n_total,
        "frames_per_video": {vid: len(v) for vid, v in by_video.items()},
        "split_of_video": split_of_video,
        "class_counts_global": dict(class_counts),
        "class_counts_by_split": {f"{c}__{s}": n
                                    for (c, s), n in by_class_split.items()},
        "split_totals": dict(split_totals),
        "by_video": dict(by_video),
    }
    OUT_JSON.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"[acdc-index] cache -> {OUT_JSON}")

    # Markdown inventory
    md = []
    md.append("# ACDC video index\n")
    md.append(f"**Source:** `{META_JSON}` (produced by `preprocess_acdc.py`)")
    md.append(f"**Video definition:** one (patient, short-axis slice) per video; "
              f"cardiac-phase index `t` is the frame ordering within each video.")
    md.append("")
    md.append("## Top-level counts\n")
    md.append("| Item | Value |")
    md.append("|---|---:|")
    md.append(f"| Total frames | {n_total} |")
    md.append(f"| Total videos (patient, slice) pairs | {n_videos} |")
    md.append(f"| Frames in train | {split_totals.get('train', 0)} |")
    md.append(f"| Frames in val | {split_totals.get('val', 0)} |")
    md.append(f"| Frames in test | {split_totals.get('test', 0)} |")
    md.append(f"| Videos that span >1 split (data leak) | {len(multi_split_videos)} |")
    md.append("")
    md.append("## Class distribution (global)\n")
    md.append("| Class | Count |")
    md.append("|---|---:|")
    for c, n in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        md.append(f"| {c} | {n} |")
    md.append("")
    md.append("## Class x split distribution\n")
    splits = ["train", "val", "test"]
    md.append("| Class | " + " | ".join(splits) + " |")
    md.append("|---|" + "|".join(["---:"] * len(splits)) + "|")
    for c in sorted(class_counts.keys()):
        row = [c] + [str(by_class_split.get((c, s), 0)) for s in splits]
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    if by_video:
        lens = sorted(len(v) for v in by_video.values())
        md.append("## Video-length distribution (cardiac cycle phases per slice)\n")
        md.append("| Stat | Value |")
        md.append("|---|---:|")
        md.append(f"| min frames/video | {lens[0]} |")
        md.append(f"| median frames/video | {lens[len(lens)//2]} |")
        md.append(f"| max frames/video | {lens[-1]} |")
        md.append(f"| mean frames/video | {sum(lens)/len(lens):.1f} |")
        md.append("")
    if multi_split_videos:
        md.append("## WARNING: videos that span multiple splits (data leak)\n")
        for vid, sps in multi_split_videos[:20]:
            md.append(f"- {vid}: {sorted(sps)}")
        md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[acdc-index] inventory -> {OUT_MD}")
    print(f"\n[summary]")
    print(f"  videos = {n_videos}")
    print(f"  frames = {n_total}")
    print(f"  train / val / test = {split_totals.get('train', 0)} / "
          f"{split_totals.get('val', 0)} / {split_totals.get('test', 0)}")
    print(f"  multi-split videos = {len(multi_split_videos)}")


if __name__ == "__main__":
    main()
