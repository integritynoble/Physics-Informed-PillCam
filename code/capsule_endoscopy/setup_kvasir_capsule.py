"""Prepare Kvasir-Capsule for training with train_stage2_pi.py.

Takes a locally downloaded Kvasir-Capsule release and arranges the labelled
frames into the ImageFolder layout expected by train_stage2_pi.py:

    {out_dir}/
      train/
        {class_1}/*.jpg
        {class_2}/*.jpg
        ...
      val/
        ...
      test/
        ...

Video-level split (no frames from the same source video appear in multiple
splits — essential to avoid train/test leakage on video datasets).

Because Kvasir-Capsule is released under terms that require explicit
acceptance (https://datasets.simula.no/kvasir-capsule/), this script does not
download the dataset automatically. Steps:

  1. Go to https://datasets.simula.no/kvasir-capsule/ and accept the terms.
  2. Download `labelled_images.zip` and `metadata.csv` to a local folder.
     (For the paper we only need `labelled_images.zip`; the raw videos are
     optional and very large.)
  3. Run this script:

         python setup_kvasir_capsule.py \
           --images_zip  /path/to/labelled_images.zip \
           --metadata    /path/to/metadata.csv \
           --out_dir     ./stage2_data \
           --mode        hardlink

Modes:
  - hardlink: fastest, no disk duplication (same volume only). Default.
  - copy:    portable, duplicates ~40 GB of images. Safe on OneDrive/cross-volume.
  - symlink: smallest, but Windows needs Developer Mode or admin.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


# Kvasir-Capsule v1.0 metadata.csv uses Title Case for several class names —
# verified by inspecting the released metadata.csv on 2026-04-24. Earlier
# revisions of this file used sentence case (matching the dataset paper text)
# but the actual data uses these strings, and ImageFolder's class_to_idx is
# built from the on-disk folder names, so we must align with the data.
KVASIR_CLASSES = [
    "Ampulla of Vater",
    "Angiectasia",
    "Blood - fresh",
    "Blood - hematin",
    "Erosion",
    "Erythema",
    "Foreign Body",
    "Ileocecal valve",
    "Lymphangiectasia",
    "Normal clean mucosa",
    "Polyp",
    "Pylorus",
    "Reduced Mucosal View",
    "Ulcer",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_zip", type=str, default=None,
                   help="Path to labelled_images.zip. If the zip is already extracted, "
                        "pass --images_dir instead.")
    p.add_argument("--images_dir", type=str, default=None,
                   help="Path to a folder that already contains the extracted labelled_images.")
    p.add_argument("--metadata", type=str, required=True,
                   help="Path to metadata.csv from the Kvasir-Capsule release.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--mode", type=str, default="hardlink",
                   choices=["hardlink", "symlink", "copy"])
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    # test_frac is 1 - train - val
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rare_class_strategy", type=str, default="video_only",
                   choices=["video_only", "frame_stratified"],
                   help="How to handle classes with fewer than --rare_class_threshold source videos. "
                        "video_only: original behavior — rare classes end up in train only. "
                        "frame_stratified: split frames within each video temporally "
                        "(first 70%% → train, next 15%% → val, last 15%% → test) so every "
                        "class is evaluable. Required when the headline ablation depends on "
                        "rare-class AUC (e.g. Polyp, Blood-fresh, Blood-hematin).")
    p.add_argument("--rare_class_threshold", type=int, default=3,
                   help="A class with fewer than this many source videos uses the rare-class "
                        "strategy. Default: 3 (i.e., classes with <3 source videos cannot be "
                        "cleanly video-stratified into train+val+test).")
    p.add_argument("--filename_col", type=str, default=None,
                   help="Override auto-detected filename column.")
    p.add_argument("--video_col", type=str, default=None,
                   help="Override auto-detected video id column.")
    p.add_argument("--class_col", type=str, default=None,
                   help="Override auto-detected class/finding column.")
    p.add_argument("--video_index", type=str, default=None,
                   help="Path to a canonical video-index JSON pinning the "
                        "{video_id: 'train'|'val'|'test'} mapping. When provided, "
                        "the random split allocator is bypassed entirely and the "
                        "split is read verbatim from the file. Use this for cross-"
                        "machine reproducibility (the GalKva-2026 benchmark ships "
                        "such a JSON under benchmark/kvasir_video_index_seed{N}.json).")
    p.add_argument("--split_manifest", type=str, default=None,
                   help="Path to the richer split-manifest JSON produced by "
                        "_build_split_manifest.py (schema_version 1). The loader "
                        "extracts the top-level `video_to_split` field and uses "
                        "it as the canonical assignment, equivalent to "
                        "--video_index against that nested dict. Strongly "
                        "preferred over --video_index because the manifest also "
                        "carries per-class counts, multilabel-frame flags, and a "
                        "content_sha256 for verification. The GalKva-2026 benchmark "
                        "ships such manifests under benchmark/.")
    p.add_argument("--dump_video_index", type=str, default=None,
                   help="If set, dump the generated {video_id: split} mapping to "
                        "the given JSON path after allocation. Use this to snapshot "
                        "a canonical split on one machine and share it via "
                        "--video_index elsewhere.")
    return p.parse_args()


def sanitize(name: str) -> str:
    """Filesystem-safe class folder name."""
    return name.replace(os.sep, "_").strip()


def _autodetect_columns(fieldnames: list[str]) -> tuple[str, str, str]:
    def pick(candidates):
        lower = {f.lower(): f for f in fieldnames}
        for c in candidates:
            if c in lower:
                return lower[c]
        return None

    fn_col = pick(["filename", "file", "image", "image_name"])
    vid_col = pick(["video_id", "video", "videoname"])
    cls_col = pick(["finding_class", "class", "label", "finding"])
    if not (fn_col and vid_col and cls_col):
        raise SystemExit(
            f"Could not auto-detect columns in {fieldnames}. "
            f"Pass --filename_col, --video_col, --class_col explicitly."
        )
    return fn_col, vid_col, cls_col


def load_metadata(path: str, filename_col, video_col, class_col) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
    print(f"[setup] metadata delimiter detected: {delimiter!r}")
    if not rows:
        raise SystemExit(f"Empty metadata: {path}")

    if not (filename_col and video_col and class_col):
        fn, vid, cls = _autodetect_columns(reader.fieldnames or [])
        filename_col = filename_col or fn
        video_col = video_col or vid
        class_col = class_col or cls
    print(f"[setup] metadata columns: filename={filename_col!r}  video={video_col!r}  class={class_col!r}")

    out = []
    for r in rows:
        cls = (r.get(class_col) or "").strip()
        if not cls or cls.lower() in ("", "none", "null", "nan"):
            continue
        out.append({
            "filename": r[filename_col].strip(),
            "video": r[video_col].strip(),
            "class": cls,
        })
    # Sanity: warn if the metadata carries class strings that don't match the
    # canonical KVASIR_CLASSES set. Catches schema drift if Simula updates the
    # release. Doesn't drop rows — surfaces the discrepancy and keeps going.
    unknown = sorted({r["class"] for r in out} - set(KVASIR_CLASSES))
    if unknown:
        print(f"[setup] WARNING: rows with non-canonical class labels: {unknown}")
    return out


def extract_zip_if_needed(images_zip: str | None, images_dir: str | None) -> Path:
    if images_dir:
        p = Path(images_dir)
        if not p.is_dir():
            raise SystemExit(f"--images_dir {p} not found")
        return p
    if not images_zip:
        raise SystemExit("Pass either --images_zip or --images_dir")

    src = Path(images_zip)
    if not src.is_file():
        raise SystemExit(f"--images_zip {src} not found")

    dest = src.with_suffix("")  # strip .zip
    if dest.is_dir():
        print(f"[setup] already extracted at {dest}, skipping unzip")
        return dest

    print(f"[setup] extracting {src} -> {dest}")
    dest.mkdir(exist_ok=True)
    with ZipFile(src) as zf:
        zf.extractall(dest)
    return dest


def _extract_per_class_tarballs(root: Path) -> int:
    """The Simula labeled-images zip contains 14 per-class .tar.gz files instead
    of raw images. If we see any unextracted tarballs anywhere under root, extract
    each one in place. Returns the number of tarballs extracted."""
    n = 0
    for tarball in list(root.rglob("*.tar.gz")):
        dest = tarball.parent
        marker = dest / (tarball.stem.removesuffix(".tar") + ".extracted")
        if marker.exists():
            continue
        print(f"[setup] extracting tarball {tarball.name} -> {dest}")
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(dest, filter="data")  # silence Py3.12+ DeprecationWarning
        marker.touch()
        n += 1
    return n


def index_images(root: Path) -> dict[str, Path]:
    """Map filename -> absolute Path, recursively. Last one wins on duplicates."""
    extracted = _extract_per_class_tarballs(root)
    if extracted:
        print(f"[setup] extracted {extracted} per-class tarball(s); re-indexing")
    idx: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            idx[p.name] = p
    return idx


def split_videos(meta: list[dict], train_frac: float, val_frac: float,
                 seed: int,
                 video_index_override: dict[str, str] | None = None) -> dict[str, str]:
    """Return {video_id: split_name}.

    Class-stratified video-level split. With Kvasir-Capsule's 43 source videos
    and 14 classes, a uniform random video shuffle leaves several classes
    represented in only one split — which makes per-class val/test AUC
    undefined. We greedily assign videos in three passes (train-first):
      Pass 1a: every class must have >=1 video in train (mandatory for learning).
      Pass 1b: every class with >=2 source videos must have >=1 in test.
      Pass 1c: every class with >=3 source videos must have >=1 in val.
      Pass 2: fill remaining videos by capacity deficit.
    Classes with only 1 source video remain training-only — there is no way
    to provide held-out examples without frame-level leakage. They are flagged
    in the caller's per-split count report.

    Cross-machine reproducibility (added 2026-05-18):
        Python set iteration order depends on PYTHONHASHSEED and on object
        identity, which differs between Windows and Linux invocations of this
        script. Even with the same `seed`, two machines reading the same
        metadata.csv can produce different video → split assignments. Two
        fixes:
          1. All consumption of class_videos[cls] and video_classes[v] is now
             sorted before iterating, so iteration order is deterministic.
          2. The optional `video_index_override` short-circuits the random
             allocation entirely. Pass a {video_id: split} dict (e.g. loaded
             from benchmark/video_index.json) to pin the assignment.

    """
    if video_index_override is not None:
        # Validate: keys must cover every video in meta; values must be in
        # {train, val, test}. Extra keys are allowed (e.g. videos not in the
        # current metadata.csv) and are simply not used.
        videos_in_meta = {r["video"] for r in meta}
        missing = videos_in_meta - set(video_index_override.keys())
        if missing:
            raise SystemExit(
                f"--video_index does not cover {len(missing)} video(s) present "
                f"in metadata.csv: {sorted(missing)[:5]}..."
            )
        bad_splits = {v: s for v, s in video_index_override.items()
                      if s not in ("train", "val", "test")}
        if bad_splits:
            raise SystemExit(
                f"--video_index has invalid split values for videos: {bad_splits}"
            )
        return {v: video_index_override[v] for v in videos_in_meta}

    rng = random.Random(seed)

    video_classes: dict[str, set[str]] = defaultdict(set)
    for r in meta:
        video_classes[r["video"]].add(r["class"])
    class_videos: dict[str, set[str]] = defaultdict(set)
    for v, cs in video_classes.items():
        for c in cs:
            class_videos[c].add(v)

    all_videos = sorted(video_classes.keys())
    n = len(all_videos)
    target = {
        "train": int(round(n * train_frac)),
        "val": int(round(n * val_frac)),
    }
    target["test"] = max(1, n - target["train"] - target["val"])

    assignment: dict[str, str] = {}

    def coverage(cls: str, split: str) -> int:
        return sum(1 for v in sorted(class_videos[cls])
                   if assignment.get(v) == split)

    def cap_left(split: str) -> int:
        used = sum(1 for s in assignment.values() if s == split)
        return target[split] - used

    def claim(cls: str, split: str) -> bool:
        # Sort before consuming so iteration order is deterministic across
        # Windows / Linux / different Python processes. The downstream
        # rng.shuffle(candidates) then provides the only source of randomness.
        candidates = sorted(v for v in class_videos[cls] if v not in assignment)
        if not candidates or cap_left(split) <= 0:
            return False
        # Prefer a video that also covers other under-served classes for this split
        def score(v: str) -> int:
            s = 0
            for c in sorted(video_classes[v]):
                if c == cls:
                    continue
                if coverage(c, split) == 0 and len(class_videos[c]) >= 2:
                    s += 1
            return s
        rng.shuffle(candidates)
        candidates.sort(key=score, reverse=True)
        assignment[candidates[0]] = split
        return True

    # Tie-break on alphabetical name so the rarity sort is deterministic
    # across machines (Python's sort is stable; the lambda result alone is
    # not unique when two classes have equal video counts).
    classes_by_rarity = sorted(class_videos.keys(),
                                key=lambda c: (len(class_videos[c]), c))

    # 1a: train must contain every class
    for cls in classes_by_rarity:
        if coverage(cls, "train") == 0:
            claim(cls, "train")
    # 1b: classes with >=2 source videos must reach test
    for cls in classes_by_rarity:
        if len(class_videos[cls]) >= 2 and coverage(cls, "test") == 0:
            claim(cls, "test")
    # 1c: classes with >=3 source videos must reach val
    for cls in classes_by_rarity:
        if len(class_videos[cls]) >= 3 and coverage(cls, "val") == 0:
            claim(cls, "val")

    # Pass 2: fill remaining by deficit (largest gap wins, alphabetical tie-break)
    remaining = sorted(v for v in all_videos if v not in assignment)
    rng.shuffle(remaining)
    for v in remaining:
        deficit = {s: cap_left(s) for s in target}
        # Stable max: largest deficit wins, train > val > test tie-break
        cand = max(("train", "val", "test"), key=lambda s: (deficit[s], -["train","val","test"].index(s)))
        assignment[v] = cand

    return assignment


def split_rare_class_frames(rows: list[dict], train_frac: float, val_frac: float) -> dict[str, str]:
    """Temporal-block frame split for a single rare class.

    Within each source video, sort the rare-class rows by frame number and
    assign the first train_frac → train, next val_frac → val, remainder → test.
    Returns {filename: split}.

    Rationale: random per-frame splitting is the highest-leakage frame split
    (adjacent frames are nearly identical). A contiguous-block split puts
    train/val/test on different time intervals of the same video, which is the
    weakest within-video correlation we can construct without leaving the
    video. Methodology follows Smedsrud et al. (2021) for rare classes.
    """
    out: dict[str, str] = {}
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)

    for vid, vrows in by_video.items():
        # Sort by integer frame number — extracted from filename suffix
        # "_<frame_idx>.jpg" if metadata.frame_number is unavailable.
        def frame_key(r: dict) -> int:
            try:
                stem = Path(r["filename"]).stem  # "<video>_<idx>"
                return int(stem.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                return 0
        vrows = sorted(vrows, key=frame_key)
        n = len(vrows)
        if n == 0:
            continue
        if n == 1:
            out[vrows[0]["filename"]] = "train"
            continue
        if n == 2:
            out[vrows[0]["filename"]] = "train"
            out[vrows[1]["filename"]] = "test"
            continue
        # n >= 3 — guarantee at least one frame in each split.
        n_tr = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_te = max(1, n - n_tr - n_val)
        # Resolve any over-allocation from rounding by trimming train.
        if n_tr + n_val + n_te > n:
            n_tr = n - n_val - n_te
        for i, r in enumerate(vrows):
            if i < n_tr:
                s = "train"
            elif i < n_tr + n_val:
                s = "val"
            else:
                s = "test"
            out[r["filename"]] = s
    return out


def assign_row_splits(meta: list[dict], train_frac: float, val_frac: float, seed: int,
                       rare_strategy: str, rare_threshold: int,
                       video_index_override: dict[str, str] | None = None) -> tuple[dict[int, str], dict]:
    """Decide per-row split. Returns ({row_index: split}, info_dict).

    - For 'common' classes (>= rare_threshold source videos), split by video
      (existing class-stratified video-level allocator).
    - For 'rare' classes:
        * rare_strategy == 'video_only' → still use video-level allocator
          (legacy behavior — rare classes typically end up in train only).
        * rare_strategy == 'frame_stratified' → temporal-block frame split
          within each video for that class.

    info_dict carries:
      'video_assignment': {video_id: split} (used for common rows)
      'rare_classes': sorted list of class names treated as rare
      'common_classes': sorted list of class names treated as common
    """
    class_videos: dict[str, set[str]] = defaultdict(set)
    for r in meta:
        class_videos[r["class"]].add(r["video"])

    rare_classes = {c for c, vs in class_videos.items() if len(vs) < rare_threshold}
    common_classes = {c for c in class_videos if c not in rare_classes}

    # Video-level allocation is computed from common-class metadata only when
    # rare_strategy == 'frame_stratified', so rare-class videos do not consume
    # val/test capacity that the common classes need. Under 'video_only' we
    # keep the original behavior (all classes participate in video allocation).
    if rare_strategy == "frame_stratified":
        meta_for_video_split = [r for r in meta if r["class"] in common_classes]
    else:
        meta_for_video_split = meta
    video_assignment = split_videos(meta_for_video_split, train_frac, val_frac, seed,
                                     video_index_override=video_index_override)

    row_splits: dict[int, str] = {}
    if rare_strategy == "frame_stratified" and rare_classes:
        for cls in sorted(rare_classes):
            cls_rows = [(i, r) for i, r in enumerate(meta) if r["class"] == cls]
            frame_split = split_rare_class_frames([r for _, r in cls_rows],
                                                    train_frac, val_frac)
            for i, r in cls_rows:
                row_splits[i] = frame_split.get(r["filename"], "train")

    for i, r in enumerate(meta):
        if i in row_splits:
            continue
        # Common-class row, or rare-class row under 'video_only' strategy.
        s = video_assignment.get(r["video"])
        if s is None:
            # Video was filtered out of the video allocator (rare-only video
            # under frame_stratified). Fall back to frame-stratified split for
            # this row's class so it is still placed somewhere.
            s = "train"
        row_splits[i] = s

    info = {
        "video_assignment": video_assignment,
        "rare_classes": sorted(rare_classes),
        "common_classes": sorted(common_classes),
    }
    return row_splits, info


def place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # fall through to copy
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    assert 0 < args.train_frac < 1 and 0 < args.val_frac < 1
    assert args.train_frac + args.val_frac < 1, "train+val must leave room for test"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_root = extract_zip_if_needed(args.images_zip, args.images_dir)
    print(f"[setup] indexing images under {images_root} ...")
    idx = index_images(images_root)
    print(f"[setup] indexed {len(idx):,} image files")

    meta = load_metadata(args.metadata, args.filename_col, args.video_col, args.class_col)
    print(f"[setup] {len(meta):,} labeled rows in metadata")

    class_counter = Counter(r["class"] for r in meta)
    print(f"[setup] class counts in metadata:")
    for c, n in class_counter.most_common():
        print(f"  {c:35s} {n:>6d}")

    video_index_override = None
    if args.video_index and args.split_manifest:
        raise SystemExit(
            "[setup] ERROR: --video_index and --split_manifest are mutually exclusive; "
            "pass only one."
        )
    if args.split_manifest:
        import json as _json
        with open(args.split_manifest, "r", encoding="utf-8") as f:
            manifest = _json.load(f)
        if "video_to_split" not in manifest:
            raise SystemExit(
                f"[setup] ERROR: --split_manifest {args.split_manifest} does not "
                f"contain a top-level `video_to_split` key. Expected schema_version 1 "
                f"manifest from _build_split_manifest.py."
            )
        video_index_override = manifest["video_to_split"]
        print(f"[setup] loaded canonical split-manifest from {args.split_manifest} "
              f"(schema_version {manifest.get('schema_version', '?')}, "
              f"{len(video_index_override)} videos pinned, "
              f"content_sha256 {manifest.get('content_sha256', '?')[:16]}…)")
        if manifest.get("leaks_detected"):
            print(f"[setup] WARNING: manifest reports {len(manifest['leaks_detected'])} "
                  f"video-level leaks; review before use.")
    elif args.video_index:
        import json as _json
        with open(args.video_index, "r", encoding="utf-8") as f:
            video_index_override = _json.load(f)
        # Accept either a flat {video_id: split} dict OR a manifest with
        # video_to_split nested (for backward compatibility).
        if isinstance(video_index_override, dict) and "video_to_split" in video_index_override:
            video_index_override = video_index_override["video_to_split"]
        print(f"[setup] loaded canonical video-index from {args.video_index} "
              f"({len(video_index_override)} videos pinned)")

    row_splits, split_info = assign_row_splits(
        meta, args.train_frac, args.val_frac, args.seed,
        rare_strategy=args.rare_class_strategy,
        rare_threshold=args.rare_class_threshold,
        video_index_override=video_index_override,
    )
    video_assignment = split_info["video_assignment"]
    video_counts = Counter(video_assignment.values())
    print(f"[setup] video-level split (common classes):  "
          f"train={video_counts['train']}  val={video_counts['val']}  test={video_counts['test']}")

    if args.dump_video_index:
        import json as _json
        os.makedirs(os.path.dirname(args.dump_video_index) or ".", exist_ok=True)
        with open(args.dump_video_index, "w", encoding="utf-8") as f:
            _json.dump(dict(sorted(video_assignment.items())), f, indent=2)
        print(f"[setup] dumped canonical video-index to {args.dump_video_index}")
    print(f"[setup] rare_class_strategy={args.rare_class_strategy} "
          f"(threshold<{args.rare_class_threshold} videos)")
    if split_info["rare_classes"]:
        if args.rare_class_strategy == "frame_stratified":
            print(f"[setup] rare classes (frame-stratified): {split_info['rare_classes']}")
        else:
            print(f"[setup] rare classes (video_only — typically train-only): {split_info['rare_classes']}")
    else:
        print(f"[setup] no rare classes detected")

    placed: dict[str, Counter] = defaultdict(Counter)
    missing = 0
    for i, r in enumerate(meta):
        fname = r["filename"]
        split = row_splits[i]
        cls_folder = sanitize(r["class"])

        src = idx.get(fname)
        if src is None:
            # try alternate extensions/basenames
            stem = Path(fname).stem
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                src = idx.get(stem + ext)
                if src is not None:
                    break
        if src is None:
            missing += 1
            continue

        dst = out_dir / split / cls_folder / fname
        place(src, dst, args.mode)
        placed[split][cls_folder] += 1

    print(f"[setup] placed frames per split:")
    for split in ("train", "val", "test"):
        total = sum(placed[split].values())
        print(f"  {split}: {total:,} frames across {len(placed[split])} classes")
    if missing:
        print(f"[setup] WARNING: {missing} rows had no matching image file in {images_root}")

    # Sanity: warn if any split is missing any class
    all_classes = set().union(*[set(placed[s].keys()) for s in ("train", "val", "test")])
    for s in ("train", "val", "test"):
        miss = all_classes - set(placed[s].keys())
        if miss:
            print(f"[setup] WARNING: split '{s}' is missing classes: {sorted(miss)}")

    print(f"[setup] done. Use --data_dir {out_dir} when running train_stage2_pi.py")


if __name__ == "__main__":
    main()
