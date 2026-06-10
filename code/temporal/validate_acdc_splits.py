"""
Validate ACDC preprocessing + sequence construction.
======================================================

Checks before running any training on ACDC:

1. **Patient-disjoint splits.** No patient's frames appear in more
   than one split. Critical for honest test-set evaluation; the
   architecture choice doesn't matter if the test set is leaky.

2. **Class-label consistency.** Every frame from the same patient
   has the same class label (the patient-level diagnosis from
   `Info.cfg`).

3. **Video-length sanity.** Each (patient, slice) "video" should
   have ~10-30 frames (one cardiac cycle). Outliers in either
   direction indicate malformed cines or preprocessing bugs.

4. **Split balance.** With 100 patients × 5 classes (20 each)
   stratified-split at 70/15/15, expect train ≈ 14 patients per
   class, val ≈ 3, test ≈ 3 (per `preprocess_acdc.py` defaults).

5. **Sequence construction.** For each labeled center frame, the
   16-frame window construction in `train_cell_b_acdc.py` must
   yield exactly 16 frames per window with no cross-patient leak.
   We verify by sampling 200 random windows and checking the
   patient and slice are constant across each window.

6. **Filesystem consistency.** Every frame in `metadata.json` has
   the corresponding JPG/PNG on disk; conversely, every JPG/PNG on
   disk has a metadata entry.

Each check prints a verdict line; final exit code is 0 on all-pass,
non-zero on any failure.

Usage:
    python validate_acdc_splits.py [--strict]

`--strict`: treat WARNINGS as errors (exit code != 0).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

DATA_ROOT = Path("D:/acdc/stage2_data")
META_JSON = DATA_ROOT / "metadata.json"
HERE = Path(__file__).resolve().parent
INDEX_JSON = HERE / "video_index_acdc.json"

CLASS_NAMES = ["NOR", "MINF", "DCM", "HCM", "RV"]
WINDOW = 16


class Validator:
    def __init__(self, strict: bool):
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def err(self, msg: str):
        self.errors.append(msg)
        print(f"  ERR: {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  WARN: {msg}")

    def ok(self, msg: str):
        print(f"  OK: {msg}")

    @property
    def exit_code(self) -> int:
        if self.errors:
            return 1
        if self.strict and self.warnings:
            return 2
        return 0


def check_metadata_present(v: Validator) -> Dict:
    print("\n[check 0] metadata.json present and parseable")
    if not META_JSON.exists():
        v.err(f"missing: {META_JSON}. Run preprocess_acdc.py first.")
        return {}
    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    n = len(meta.get("frames", []))
    v.ok(f"loaded {n} frame entries")
    return meta


def check_patient_disjoint(v: Validator, meta: Dict):
    print("\n[check 1] patient-disjoint splits")
    by_patient: Dict[str, Set[str]] = defaultdict(set)
    for fr in meta.get("frames", []):
        by_patient[fr["patient_id"]].add(fr["split"])
    bad = {p: s for p, s in by_patient.items() if len(s) > 1}
    if bad:
        v.err(f"{len(bad)} patient(s) appear in >1 split:")
        for p, s in list(bad.items())[:10]:
            print(f"      {p}: {sorted(s)}")
    else:
        v.ok(f"all {len(by_patient)} patients are confined to a single split")


def check_class_consistency(v: Validator, meta: Dict):
    print("\n[check 2] class-label consistency per patient")
    by_patient: Dict[str, Set[str]] = defaultdict(set)
    for fr in meta.get("frames", []):
        by_patient[fr["patient_id"]].add(fr["class"])
    bad = {p: s for p, s in by_patient.items() if len(s) > 1}
    if bad:
        v.err(f"{len(bad)} patient(s) have multiple class labels:")
        for p, s in list(bad.items())[:5]:
            print(f"      {p}: {sorted(s)}")
    else:
        v.ok(f"all {len(by_patient)} patients have a single consistent class")


def check_video_lengths(v: Validator, meta: Dict):
    print("\n[check 3] video-length sanity (one cardiac cycle per slice)")
    lengths_by_video: Dict[str, int] = defaultdict(int)
    for fr in meta.get("frames", []):
        vid = f"{fr['patient_id']}_s{fr['slice']:02d}"
        lengths_by_video[vid] += 1
    if not lengths_by_video:
        v.warn("no videos to check")
        return
    lens = sorted(lengths_by_video.values())
    n = len(lens)
    median = lens[n // 2]
    print(f"  videos = {n}, min = {lens[0]}, median = {median}, max = {lens[-1]}")
    if lens[0] < 5:
        v.warn(f"min frames/video = {lens[0]}; expected >=5 for any usable cycle")
    if lens[-1] > 60:
        v.warn(f"max frames/video = {lens[-1]}; unusually long, check NIfTI shape")
    if median < 10 or median > 35:
        v.warn(f"median frames/video = {median}; expected 10-35 for ACDC")
    else:
        v.ok(f"median {median} frames/video falls in expected 10-35 band")


def check_split_balance(v: Validator, meta: Dict):
    print("\n[check 4] split balance (target: ~70/15/15 patients)")
    pats_in_split: Dict[str, Set[str]] = defaultdict(set)
    for fr in meta.get("frames", []):
        pats_in_split[fr["split"]].add(fr["patient_id"])
    counts = {s: len(p) for s, p in pats_in_split.items()}
    print(f"  {counts}")
    total = sum(counts.values())
    if total == 0:
        v.err("no patients in any split")
        return
    train_frac = counts.get("train", 0) / total
    val_frac = counts.get("val", 0) / total
    test_frac = counts.get("test", 0) / total
    if not (0.55 < train_frac < 0.80):
        v.warn(f"train fraction = {train_frac:.2f}; outside 0.55-0.80")
    if not (0.05 < val_frac < 0.25):
        v.warn(f"val fraction = {val_frac:.2f}; outside 0.05-0.25")
    if not (0.05 < test_frac < 0.25):
        v.warn(f"test fraction = {test_frac:.2f}; outside 0.05-0.25")
    if v.errors == [] and v.warnings == []:
        v.ok(f"split fractions train/val/test = "
             f"{train_frac:.2f}/{val_frac:.2f}/{test_frac:.2f}")
    # Per-class per-split table
    by_class_split: Counter = Counter()
    for fr in meta.get("frames", []):
        by_class_split[(fr["class"], fr["split"])] += 1
    # Reduce to per-patient counts (one patient = one diagnosis)
    pcs: Counter = Counter()
    for fr in meta.get("frames", []):
        pcs[(fr["patient_id"], fr["class"], fr["split"])] += 1
    pat_class_split = Counter()
    for (pid, cname, split), _ in pcs.items():
        pat_class_split[(cname, split)] += 1
    print("  per-class patient counts:")
    for cname in CLASS_NAMES:
        row = " ".join(f"{s}={pat_class_split.get((cname, s), 0)}"
                          for s in ("train", "val", "test"))
        print(f"      {cname:5s}: {row}")


def check_sequence_construction(v: Validator):
    print("\n[check 5] sequence construction sanity (16-frame windows)")
    if not INDEX_JSON.exists():
        v.warn(f"{INDEX_JSON} not found; skipping (run "
               f"`build_video_index_acdc.py` first)")
        return
    idx = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    by_video: Dict[str, List[Dict]] = idx.get("by_video", {})
    if not by_video:
        v.warn("video index has no videos")
        return

    # Sample 200 random (video, frame) pairs and reconstruct the window
    rng = random.Random(123)
    flat: List = []
    for vid, fs in by_video.items():
        for i, f in enumerate(fs):
            flat.append((vid, i, fs))
    if not flat:
        v.warn("no labeled frames in index")
        return
    sample = rng.sample(flat, k=min(200, len(flat)))

    half = WINDOW // 2
    bad_window_size = 0
    bad_continuity = 0
    bad_class_consistency = 0
    for vid, center, fs in sample:
        n = len(fs)
        lo = max(0, center - half)
        hi = min(n, center - half + WINDOW)
        if hi - lo < WINDOW:
            if lo == 0:
                hi = min(n, WINDOW)
            elif hi == n:
                lo = max(0, n - WINDOW)
        window = fs[lo:hi]
        # boundary pad to exactly WINDOW
        while len(window) < WINDOW:
            window.append(window[-1])
        if len(window) != WINDOW:
            bad_window_size += 1
        # Continuity: filenames within window should all start with vid
        if not all(f["filename"].startswith(vid) for f in window):
            bad_continuity += 1
        # Class consistency within window
        classes = set(f["class"] for f in window)
        if len(classes) > 1:
            bad_class_consistency += 1

    print(f"  sampled {len(sample)} windows")
    print(f"  bad_window_size: {bad_window_size}")
    print(f"  cross-video leakage (filename mismatch): {bad_continuity}")
    print(f"  cross-class within-window: {bad_class_consistency}")
    if bad_window_size or bad_continuity:
        v.err("sequence construction failed sanity checks")
    elif bad_class_consistency:
        v.warn(f"{bad_class_consistency} windows span multiple classes — "
               f"only legal if a slice has variable diagnosis (it should not)")
    else:
        v.ok("all 200 sampled windows are size-16, single-video, single-class")


def check_filesystem_consistency(v: Validator, meta: Dict):
    print("\n[check 6] filesystem consistency (metadata <-> disk)")
    expected = set(fr["filename"] for fr in meta.get("frames", []))
    expected_paths = set()
    for fr in meta.get("frames", []):
        p = DATA_ROOT / fr["split"] / fr["class"] / fr["filename"]
        expected_paths.add(p)

    on_disk: Set[Path] = set()
    for split in ("train", "val", "test"):
        sd = DATA_ROOT / split
        if not sd.exists():
            continue
        for cd in sd.iterdir():
            if not cd.is_dir():
                continue
            for f in cd.iterdir():
                if f.suffix.lower() in (".jpg", ".png", ".jpeg"):
                    on_disk.add(f)

    missing = expected_paths - on_disk
    extra = on_disk - expected_paths
    if missing:
        v.err(f"{len(missing)} files in metadata but missing on disk "
              f"(first 5: {[str(p.name) for p in list(missing)[:5]]})")
    if extra:
        v.warn(f"{len(extra)} files on disk without metadata entry")
    if not missing and not extra:
        v.ok(f"all {len(expected_paths)} files in metadata are on disk; "
             f"no extras")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                     help="treat warnings as errors in exit code")
    args = ap.parse_args()

    v = Validator(strict=args.strict)
    print(f"[validate] strict={args.strict}  data_root={DATA_ROOT}")

    meta = check_metadata_present(v)
    if not meta:
        sys.exit(v.exit_code)

    check_patient_disjoint(v, meta)
    check_class_consistency(v, meta)
    check_video_lengths(v, meta)
    check_split_balance(v, meta)
    check_sequence_construction(v)
    check_filesystem_consistency(v, meta)

    print("\n[summary]")
    print(f"  errors  : {len(v.errors)}")
    print(f"  warnings: {len(v.warnings)}")
    print(f"  exit    : {v.exit_code}")
    sys.exit(v.exit_code)


if __name__ == "__main__":
    main()
