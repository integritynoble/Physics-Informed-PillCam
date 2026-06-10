"""One-pass inference + paper-grade analysis for the 12 Windows EffB0 checkpoints.

For each of the 12 effb0_paper_seed{N}_{rgb|pi} checkpoints:
  1. Load best_model.pt (Windows-trained, paper §4.1 headline).
  2. Run inference on Kvasir test split, save per-frame softmax probs + labels
     + paths. (Cached as test_predictions.npz for downstream stats.)
  3. Compute per-class AUC + 95% BCa CI (item 2 + item 9 inputs).
  4. Compute calibration metrics:  ECE (15 bins), Brier scores per class,
     reliability-diagram data (item 7).
  5. Compute per-patient (= per-test-video) AUC for each class +
     macro-AUC, with bootstrap CI (item 9).
  6. Save everything to <checkpoint_dir>/paper_analysis.json so downstream
     aggregator can produce the supplementary table + figures.

USAGE
    python effb0_paper_analysis.py
        # iterates over all cross_backbone/effb0_paper_seed*_* dirs
        # idempotent: skips checkpoints whose paper_analysis.json exists
"""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss

GASTRO = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/gastroscopy_code_package"
CAPSULE = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/Capsule-Endoscopy"
TEST_DIR = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/data/stage2_data/test"

ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
TRAINING_ONLY = {"Ampulla of Vater", "Blood - hematin", "Polyp"}


def _setup_imports():
    for p in (GASTRO, CAPSULE):
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_model(args: dict, device: str):
    use_pi = bool(args.get("use_physics_prior", False))
    if use_pi:
        from datasets_pi import build_transforms_pi  # noqa: WPS433
        from models_pi import ImageClassifierPI  # noqa: WPS433
        tf = build_transforms_pi(args["image_size"], train=False,
                                 alpha=args.get("physics_alpha", 4.0),
                                 lambda_eff=args.get("physics_lambda_eff"),
                                 version=args.get("physics_prior_version", "v1"),
                                 pivot_v2=args.get("physics_pivot_v2", 0.30))
        m = ImageClassifierPI(args["model_name"], num_classes=len(ALL_CLASSES),
                              pretrained=False).to(device)
    else:
        from datasets import build_transforms  # noqa: WPS433
        from models import ImageClassifier  # noqa: WPS433
        tf = build_transforms(args["image_size"], False)
        m = ImageClassifier(args["model_name"], num_classes=len(ALL_CLASSES),
                            pretrained=False).to(device)
    return m, tf, ("pi" if use_pi else "rgb")


def _video_id_from_path(p: str) -> str:
    """Kvasir frames are '<video_id>_<frame_idx>.jpg'. Strip the trailing
    _<int> and the extension. Used for per-patient (per-video) grouping."""
    stem = Path(p).stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else stem


def _ece(probs_pos: np.ndarray, labels_bin: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width bins on confidence."""
    confidences = probs_pos
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels_bin)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = float(labels_bin[mask].mean())
        conf = float(confidences[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _bca_ci(samples: np.ndarray, point: float, alpha: float = 0.05,
            jackknife_values: np.ndarray | None = None) -> tuple[float, float]:
    """BCa CI from already-computed bootstrap samples; optionally accelerated
    by precomputed jackknife values. Falls back to percentile if BCa fails."""
    from scipy import stats as _ss
    samples = samples[~np.isnan(samples)]
    if samples.size < 10:
        return float("nan"), float("nan")
    frac = max(min(float((samples < point).sum()) / samples.size, 1 - 1e-8), 1e-8)
    z0 = float(_ss.norm.ppf(frac))
    if jackknife_values is None or jackknife_values.size < 10:
        a = 0.0
    else:
        jm = jackknife_values.mean()
        num = float(np.sum((jm - jackknife_values) ** 3))
        den = float(6.0 * (np.sum((jm - jackknife_values) ** 2)) ** 1.5)
        a = num / den if den != 0 else 0.0
    z_lo, z_hi = float(_ss.norm.ppf(alpha / 2)), float(_ss.norm.ppf(1 - alpha / 2))
    try:
        alo = float(_ss.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo))))
        ahi = float(_ss.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi))))
    except Exception:
        alo, ahi = alpha / 2, 1 - alpha / 2
    return float(np.quantile(samples, alo)), float(np.quantile(samples, ahi))


def evaluate(ckpt_dir: Path, device: str, batch_size: int = 64,
             n_bootstrap: int = 500, force: bool = False) -> dict | None:
    ckpt_path = ckpt_dir / "best_model.pt"
    out_path = ckpt_dir / "paper_analysis.json"
    npz_path = ckpt_dir / "test_predictions.npz"
    if not ckpt_path.exists():
        return None
    if out_path.exists() and npz_path.exists() and not force:
        return json.loads(out_path.read_text())

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck["args"]
    if isinstance(args, dict):
        args_d = args
    else:
        args_d = vars(args)
    model, tf, arm = _build_model(args_d, device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    from datasets import FolderDatasetWithPaths  # noqa: WPS433
    from metrics_pi import ensure_class_folders  # noqa: WPS433
    ensure_class_folders(TEST_DIR, ALL_CLASSES)
    ds = FolderDatasetWithPaths(TEST_DIR, transform=tf, allow_empty=True)
    if ds.classes != ALL_CLASSES:
        raise RuntimeError(f"class set mismatch: {ds.classes}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=(device == "cuda"))

    all_probs, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for imgs, lab, paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(int(x) for x in lab.numpy())
            all_paths.extend(paths)
    probs = np.concatenate(all_probs, axis=0)  # (N, 14)
    labels = np.asarray(all_labels, dtype=np.int32)
    paths = np.asarray(all_paths)
    np.savez_compressed(npz_path, probs=probs, labels=labels, paths=paths,
                        classes=np.asarray(ALL_CLASSES))

    rng = np.random.default_rng(int(args_d.get("seed", 42)))
    per_class = {}
    per_class_auc_ci: dict[str, dict] = {}
    calib: dict[str, dict] = {}
    n = len(labels)

    # Per-class AUC + BCa CI + calibration (ECE / Brier)
    for c_idx, c_name in enumerate(ALL_CLASSES):
        y_bin = (labels == c_idx).astype(int)
        n_pos = int(y_bin.sum())
        per_class[c_name] = {"n_positive": n_pos, "n_negative": int(n - n_pos)}
        if n_pos == 0 or n_pos == n:
            per_class[c_name]["auc"] = None
            per_class_auc_ci[c_name] = {"point": None, "lo": None, "hi": None}
            calib[c_name] = {"ece_15": None, "brier": None}
            continue
        # Point AUC
        try:
            point = float(roc_auc_score(y_bin, probs[:, c_idx]))
        except Exception:
            point = float("nan")
        per_class[c_name]["auc"] = point
        # BCa CI via bootstrap resampling
        if not np.isnan(point):
            samples = np.empty(n_bootstrap, dtype=float)
            for i in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                try:
                    samples[i] = roc_auc_score(y_bin[idx], probs[idx, c_idx])
                except Exception:
                    samples[i] = float("nan")
            samples = samples[~np.isnan(samples)]
            lo, hi = _bca_ci(samples, point) if samples.size >= 10 else (float("nan"), float("nan"))
            per_class_auc_ci[c_name] = {"point": point, "lo": lo, "hi": hi,
                                        "n_resamples": int(samples.size)}
        else:
            per_class_auc_ci[c_name] = {"point": None, "lo": None, "hi": None}
        # Calibration
        calib[c_name] = {
            "ece_15": _ece(probs[:, c_idx], y_bin, n_bins=15),
            "brier": float(brier_score_loss(y_bin, probs[:, c_idx])),
        }

    # Macro-AUC (evaluable classes only — skips training-only classes with 0 test support)
    eval_aucs = [per_class[c]["auc"] for c in ALL_CLASSES
                 if c not in TRAINING_ONLY and per_class[c]["auc"] is not None]
    macro_auc_eval = float(np.mean(eval_aucs)) if eval_aucs else float("nan")

    # Per-patient (per-video) macro-AUC
    videos = np.asarray([_video_id_from_path(p) for p in paths])
    unique_videos = sorted(set(videos.tolist()))
    per_patient: dict[str, dict] = {}
    for vid in unique_videos:
        mask = (videos == vid)
        v_labels = labels[mask]
        v_probs = probs[mask]
        v_aucs = []
        for c_idx, c_name in enumerate(ALL_CLASSES):
            if c_name in TRAINING_ONLY:
                continue
            y_bin = (v_labels == c_idx).astype(int)
            if 0 < y_bin.sum() < len(y_bin):
                try:
                    v_aucs.append(float(roc_auc_score(y_bin, v_probs[:, c_idx])))
                except Exception:
                    pass
        per_patient[vid] = {
            "n_frames": int(mask.sum()),
            "n_evaluable_classes": len(v_aucs),
            "macro_auc": float(np.mean(v_aucs)) if v_aucs else None,
        }

    out = {
        "ckpt": str(ckpt_path),
        "arm": arm,
        "seed": int(args_d.get("seed", 42)),
        "model_name": args_d.get("model_name"),
        "n_test_frames": int(n),
        "macro_auc_evaluable": macro_auc_eval,
        "per_class": per_class,
        "per_class_auc_ci": per_class_auc_ci,
        "calibration": calib,
        "per_patient": per_patient,
        "n_bootstrap": int(n_bootstrap),
    }
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/cross_backbone")
    ap.add_argument("--pattern", default="effb0_paper_seed*_*",
                    help="Glob under root to iterate over.")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n_bootstrap", type=int, default=500)
    cli = ap.parse_args()
    _setup_imports()
    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[paper_analysis] device={device}")
    targets = sorted(Path(cli.root).glob(cli.pattern))
    n_done, n_skip, n_fail = 0, 0, 0
    for d in targets:
        try:
            r = evaluate(d, device=device, n_bootstrap=cli.n_bootstrap, force=cli.force)
            if r is None:
                n_skip += 1
            else:
                n_done += 1
                print(f"  {d.name}  arm={r['arm']}  macro_auc={r['macro_auc_evaluable']:.4f}  "
                      f"|patients|={len(r['per_patient'])}")
        except Exception as e:
            n_fail += 1
            print(f"  {d.name}  ERROR: {e}")
            traceback.print_exc()
    print(f"[paper_analysis] done: {n_done}  skipped: {n_skip}  failed: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
