"""
PI-TTA v2, experiment 2 — spatial-feature-map distillation.
=============================================================

Follow-up to `pi_tta_v2_embedding_kd.py` (which ran null at the
pooled-embedding layer with ~11% gap recovery). This experiment moves
the alignment one layer earlier — to the pre-pool spatial feature map
(7x7x1280 for EfficientNet-B0). The cell (b+) result and the embedding-
KD null together suggest the +PI lift sits in the spatial structure
that the global pool discards.

Pipeline (per seed):

  1. ADAPTER TRAINING:
     - Sample a stratified 2000-frame subset of the train split.
     - Forward both backbones (RGB and +PI) over that subset; cache
       the spatial feature maps (~500 MB cache).
     - Train a 1x1 conv adapter g_phi: (1280, 7, 7) -> (1280, 7, 7),
       initialized as identity. Loss: MSE(g_phi(rgb_map), pi_map).
     - K=30 epochs of MSE training; expect ~5-10 min for adapter
       training itself.

  2. POOLED-EMBEDDING EXTRACTION (full dataset):
     - Forward RGB backbone over all 47K frames; apply adapter; pool;
       cache as (47K, 1280) of "adapted RGB" embeddings.
     - Reuse already-cached +PI embeddings for the upper-bound probe.

  3. LINEAR-PROBE COMPARISON:
     - Three probes, all trained on train, evaluated on test:
         probe_rgb     : on cached RGB embeddings (cell-(a) RGB baseline)
         probe_pi_real : on cached +PI embeddings (cell-(a) +PI upper bound)
         probe_pi_hat  : on adapted RGB embeddings (the new test)
     - Recovery = (probe_pi_hat - probe_rgb) / (probe_pi_real - probe_rgb).

Verdict thresholds:
  >=50% recovery: PI-TTA v2 spatial design is promising; proceed to
                   write the full TTA loop on the backbone.
  20-50%:        marginal; consider training-time KD instead.
  <20%:          null; drop PI-TTA from the NMI plan.

Compute: ~15-25 min per seed on GTX 1660 Ti. Cross-seed ~2 hr.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

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
RGB_OUT = Path("D:/kvasir_capsule/outputs")
EMB_PI_DIR = Path("D:/kvasir_capsule/outputs/embeddings_pi")
EMB_RGB_DIR = Path("D:/kvasir_capsule/outputs/embeddings")
ADAPTER_DIR = Path("D:/kvasir_capsule/outputs/pi_tta_v2_spatial")
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = HERE.parent.parent / "docs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [41, 42, 43, 44, 45, 47]
BATCH_SIZE = 32
IMAGE_SIZE = 224

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
N_CLASSES = len(CLASS_NAMES)
SPLIT_TO_INT = {"train": 0, "val": 1, "test": 2}

ADAPTER_TRAIN_FRAMES = 2000     # stratified subsample size
ADAPTER_EPOCHS = 30
ADAPTER_LR = 1e-3
ADAPTER_WD = 1e-4

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def rgb_dir_for(seed: int) -> Path:
    if seed == 42:
        return RGB_OUT / "stage2_rgb_effb0"
    return RGB_OUT / f"stage2_rgb_effb0_seed{seed}"


def pi_dir_for(seed: int) -> Path:
    if seed == 42:
        return RGB_OUT / "stage2_pi_effb0"
    return RGB_OUT / f"stage2_pi_effb0_seed{seed}"


class AllFramesDataset(Dataset):
    """Returns (3-ch normalized tensor, filename, class, split)."""
    def __init__(self, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        self.samples: List[Tuple[Path, str, int, int]] = []
        for split in ("train", "val", "test"):
            sd = DATA_ROOT / split
            if not sd.is_dir():
                continue
            for cd in sorted(sd.iterdir()):
                if not cd.is_dir() or cd.name not in CLASS_NAMES:
                    continue
                cidx = CLASS_NAMES.index(cd.name)
                sidx = SPLIT_TO_INT[split]
                for f in cd.iterdir():
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


def stratified_subsample_indices(dataset: AllFramesDataset,
                                    n_per_class_target: int = None,
                                    total_target: int = ADAPTER_TRAIN_FRAMES,
                                    seed: int = 0) -> List[int]:
    """Return indices for a stratified subset of the train split."""
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, (_, _, lbl, sp) in enumerate(dataset.samples):
        if sp == SPLIT_TO_INT["train"]:
            by_class[lbl].append(i)
    if n_per_class_target is None:
        n_per_class_target = max(1, total_target // len(by_class))
    out: List[int] = []
    for c, idxs in by_class.items():
        idxs_sorted = sorted(idxs)
        rng.shuffle(idxs_sorted)
        out.extend(idxs_sorted[:n_per_class_target])
    rng.shuffle(out)
    return out


def unnormalize(x_norm: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x_norm.device, dtype=x_norm.dtype)
    std = IMAGENET_STD.to(x_norm.device, dtype=x_norm.dtype)
    return (x_norm * std + mean).clamp(0.0, 1.0)


def load_models(seed: int):
    from models import ImageClassifier
    from models_pi import ImageClassifierPI

    rgb_ckpt = rgb_dir_for(seed) / "best_model.pt"
    pi_ckpt = pi_dir_for(seed) / "best_model.pt"
    if not rgb_ckpt.exists():
        raise FileNotFoundError(f"missing RGB ckpt: {rgb_ckpt}")
    if not pi_ckpt.exists():
        raise FileNotFoundError(f"missing +PI ckpt: {pi_ckpt}")

    print(f"[seed {seed}] loading RGB ckpt {rgb_ckpt.name}")
    ckpt_rgb = torch.load(rgb_ckpt, map_location=DEVICE, weights_only=False)
    rgb_args = ckpt_rgb["args"]
    rgb_model = ImageClassifier(rgb_args["model_name"],
                                  num_classes=len(ckpt_rgb["class_names"]),
                                  pretrained=False).to(DEVICE)
    rgb_model.load_state_dict(ckpt_rgb["model_state"], strict=True)
    rgb_model.eval()

    print(f"[seed {seed}] loading +PI ckpt {pi_ckpt.name}")
    ckpt_pi = torch.load(pi_ckpt, map_location=DEVICE, weights_only=False)
    pi_args = ckpt_pi["args"]
    extra_channels = int(pi_args.get("extra_channels", 2))
    pi_model = ImageClassifierPI(pi_args["model_name"],
                                    num_classes=len(ckpt_pi["class_names"]),
                                    pretrained=False,
                                    extra_channels=extra_channels).to(DEVICE)
    pi_model.load_state_dict(ckpt_pi["model_state"], strict=True)
    pi_model.eval()

    return rgb_model, pi_model, pi_args


def get_rgb_spatial(rgb_model, x: torch.Tensor) -> torch.Tensor:
    return rgb_model.backbone.features(x)   # (B, 1280, 7, 7)


def get_pi_spatial(pi_model, x_5ch: torch.Tensor) -> torch.Tensor:
    return pi_model.backbone.features(x_5ch)


def get_pi_input(x_rgb_norm: torch.Tensor, pi_args: Dict) -> torch.Tensor:
    """Build the 5-channel input the +PI model expects."""
    from physics_prior import physics_channels
    x_rgb = unnormalize(x_rgb_norm.float())
    phys = physics_channels(x_rgb,
                              alpha=pi_args.get("physics_alpha", 4.0),
                              lambda_eff=pi_args.get("physics_lambda_eff", None),
                              version=pi_args.get("physics_prior_version", "v1"),
                              pivot_v2=pi_args.get("physics_pivot_v2", 0.30))
    return torch.cat([x_rgb_norm, phys], dim=1)


class SpatialAdapter(nn.Module):
    """1x1 conv on the 7x7 spatial feature map. Initialized as identity."""

    def __init__(self, channels: int = 1280):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.eye_(self.conv.weight.view(channels, channels))
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def cache_spatial_pairs_for_subset(rgb_model, pi_model, pi_args,
                                       dataset: AllFramesDataset,
                                       indices: List[int]
                                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward both backbones over the indexed subset; return paired
    spatial maps (kept in CPU memory as float16 to save space)."""
    print(f"[adapter] caching spatial maps for {len(indices)} frames")
    n = len(indices)
    rgb_maps = torch.empty((n, 1280, 7, 7), dtype=torch.float16)
    pi_maps = torch.empty((n, 1280, 7, 7), dtype=torch.float16)
    t0 = time.time()
    pos = 0
    while pos < n:
        batch_idx = indices[pos: pos + BATCH_SIZE]
        xs = []
        for i in batch_idx:
            x, _fname, _lbl, _sp = dataset[i]
            xs.append(x)
        x_norm = torch.stack(xs, dim=0).to(DEVICE)
        with torch.no_grad():
            rgb_map = get_rgb_spatial(rgb_model, x_norm)
            x_pi_in = get_pi_input(x_norm, pi_args)
            pi_map = get_pi_spatial(pi_model, x_pi_in)
        bs = x_norm.size(0)
        rgb_maps[pos: pos + bs] = rgb_map.detach().cpu().to(torch.float16)
        pi_maps[pos: pos + bs] = pi_map.detach().cpu().to(torch.float16)
        pos += bs
        if pos % (BATCH_SIZE * 10) == 0 or pos >= n:
            elapsed = time.time() - t0
            rate = pos / max(0.001, elapsed)
            eta = (n - pos) / max(0.001, rate)
            print(f"[adapter] cached {pos}/{n}  rate={rate:.0f} fps  eta={eta:.0f}s")
    return rgb_maps, pi_maps


def train_adapter(rgb_maps: torch.Tensor, pi_maps: torch.Tensor,
                   seed: int) -> SpatialAdapter:
    torch.manual_seed(seed)
    adapter = SpatialAdapter(channels=1280).to(DEVICE)

    train_ds = TensorDataset(rgb_maps.to(torch.float32),
                                pi_maps.to(torch.float32))
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(adapter.parameters(), lr=ADAPTER_LR,
                              weight_decay=ADAPTER_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ADAPTER_EPOCHS)

    print(f"[adapter] training adapter for {ADAPTER_EPOCHS} epochs")
    t0 = time.time()
    for epoch in range(1, ADAPTER_EPOCHS + 1):
        adapter.train()
        running, seen = 0.0, 0
        for r_cpu, p_cpu in loader:
            r = r_cpu.to(DEVICE)
            p = p_cpu.to(DEVICE)
            pred = adapter(r)
            loss = F.mse_loss(pred, p)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * r.size(0)
            seen += r.size(0)
        sched.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == ADAPTER_EPOCHS:
            elapsed = (time.time() - t0)
            print(f"[adapter] epoch {epoch:2d}/{ADAPTER_EPOCHS}  "
                  f"train_mse={running/max(1,seen):.4f}  elapsed={elapsed:.0f}s")
    return adapter


def extract_adapted_pooled(rgb_model, adapter: SpatialAdapter,
                              dataset: AllFramesDataset
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward RGB backbone over ALL frames, apply adapter, pool, return
    (filenames, labels, splits, adapted_pooled_embeddings)."""
    print(f"[apply] forwarding RGB backbone + adapter over all {len(dataset)} frames")
    n = len(dataset)
    feats = np.zeros((n, 1280), dtype=np.float32)
    fnames: List[str] = [""] * n
    labels = np.zeros((n,), dtype=np.int64)
    splits = np.zeros((n,), dtype=np.int64)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    adapter.eval()
    pos = 0
    t0 = time.time()
    with torch.no_grad():
        for x, fn, y, s in loader:
            x = x.to(DEVICE, non_blocking=True)
            rgb_map = get_rgb_spatial(rgb_model, x)
            adapted = adapter(rgb_map)
            pooled = F.adaptive_avg_pool2d(adapted, (1, 1)).flatten(1)
            bs = x.size(0)
            feats[pos: pos + bs] = pooled.cpu().numpy()
            for k in range(bs):
                fnames[pos + k] = fn[k]
            labels[pos: pos + bs] = y.numpy()
            splits[pos: pos + bs] = s.numpy()
            pos += bs
            if pos % (BATCH_SIZE * 50) == 0:
                rate = pos / max(0.001, (time.time() - t0))
                eta = (n - pos) / max(0.001, rate) / 60
                print(f"[apply] {pos}/{n}  rate={rate:.0f} fps  eta={eta:.1f} min")
    return np.array(fnames), labels, splits, feats


def macro_auc(probs: np.ndarray, labels: np.ndarray
              ) -> Tuple[float, Dict[str, float]]:
    from sklearn.metrics import roc_auc_score
    pc: Dict[str, float] = {}
    for j, c in enumerate(CLASS_NAMES):
        y = (labels == j).astype(np.int32)
        if y.sum() == 0 or y.sum() == len(y):
            pc[c] = float("nan")
            continue
        pc[c] = float(roc_auc_score(y, probs[:, j]))
    vals = [v for v in pc.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan"), pc


def linear_probe(e_train, y_train, e_test, y_test, seed):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0,
                              class_weight="balanced",
                              random_state=seed, solver="lbfgs")
    clf.fit(e_train, y_train)
    probs = clf.predict_proba(e_test)
    return macro_auc(probs, y_test)


def evaluate_seed(seed: int) -> Dict:
    print(f"\n{'='*60}\nSEED = {seed} (PI-TTA v2 spatial KD)\n{'='*60}")
    rgb_model, pi_model, pi_args = load_models(seed)
    dataset = AllFramesDataset()
    print(f"[main] dataset has {len(dataset)} frames")

    # Phase 1: cache spatial-map pairs for stratified train subset
    indices = stratified_subsample_indices(dataset, total_target=ADAPTER_TRAIN_FRAMES,
                                              seed=seed)
    print(f"[main] adapter-training subset = {len(indices)} frames "
          f"(stratified by class)")
    rgb_maps, pi_maps = cache_spatial_pairs_for_subset(
        rgb_model, pi_model, pi_args, dataset, indices)

    # Phase 2: train adapter
    adapter = train_adapter(rgb_maps, pi_maps, seed)
    torch.save(adapter.state_dict(), ADAPTER_DIR / f"adapter_seed{seed}.pt")

    # Phase 3: forward RGB backbone + adapter over all frames
    fn_a, lb_a, sp_a, e_pi_hat = extract_adapted_pooled(rgb_model, adapter, dataset)

    # Free spatial maps cache
    del rgb_maps, pi_maps
    torch.cuda.empty_cache()

    # Phase 4: load existing RGB and +PI pooled-embedding caches; align by
    # filename to e_pi_hat
    print(f"[main] loading existing pooled embedding caches")
    npz_rgb = np.load(EMB_RGB_DIR / f"seed{seed}_embeddings.npz")
    npz_pi = np.load(EMB_PI_DIR / f"seed{seed}_embeddings.npz")
    rgb_fn = list(npz_rgb["filenames"])
    pi_fn = list(npz_pi["filenames"])
    e_rgb_arr = np.array(npz_rgb["embeddings"])
    e_pi_arr = np.array(npz_pi["embeddings"])
    rgb_lbl = np.array(npz_rgb["labels"])
    rgb_split = np.array(npz_rgb["splits"])
    npz_rgb.close(); npz_pi.close()

    fn_rgb_to_idx = {fn: i for i, fn in enumerate(rgb_fn)}
    fn_pi_to_idx = {fn: i for i, fn in enumerate(pi_fn)}

    # Reindex to the order of fn_a so all three feature arrays are aligned
    rgb_idx = [fn_rgb_to_idx[fn] for fn in fn_a if fn in fn_rgb_to_idx]
    pi_idx = [fn_pi_to_idx[fn] for fn in fn_a if fn in fn_pi_to_idx]
    keep = [i for i, fn in enumerate(fn_a)
            if fn in fn_rgb_to_idx and fn in fn_pi_to_idx]
    e_rgb = e_rgb_arr[rgb_idx]
    e_pi = e_pi_arr[pi_idx]
    e_pi_hat_kept = e_pi_hat[keep]
    labels = lb_a[keep]
    splits = sp_a[keep]

    # Linear probes
    train = (splits == 0); test = (splits == 2)
    print(f"[probe] train n={train.sum()}, test n={test.sum()}")
    auc_rgb, _ = linear_probe(e_rgb[train], labels[train],
                                  e_rgb[test], labels[test], seed)
    auc_pi_real, _ = linear_probe(e_pi[train], labels[train],
                                       e_pi[test], labels[test], seed)
    auc_pi_hat, _ = linear_probe(e_pi_hat_kept[train], labels[train],
                                       e_pi_hat_kept[test], labels[test], seed)

    gap = auc_pi_real - auc_rgb
    recovered = auc_pi_hat - auc_rgb
    frac = recovered / max(1e-6, gap) if gap > 0 else float("nan")
    print(f"[seed {seed}] probe_rgb     = {auc_rgb:.4f}")
    print(f"[seed {seed}] probe_pi_real = {auc_pi_real:.4f}  (gap = {gap:+.4f})")
    print(f"[seed {seed}] probe_pi_hat  = {auc_pi_hat:.4f}  "
          f"(recovered = {recovered:+.4f}, frac = {frac*100:.1f}%)")

    return {
        "seed": seed,
        "auc_rgb": auc_rgb,
        "auc_pi_real": auc_pi_real,
        "auc_pi_hat": auc_pi_hat,
        "gap": gap,
        "recovered": recovered,
        "frac_recovered": frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    seeds = args.only_seeds or SEEDS
    print(f"[main] device={DEVICE}  seeds={seeds}")
    print(f"[main] PI-TTA v2 spatial-KD experiment")

    results: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        try:
            r = evaluate_seed(seed)
            results.append(r)
        except FileNotFoundError as exc:
            print(f"[seed {seed}] skip: {exc}")
            continue

    if not results:
        print("[main] no results; exiting")
        return

    arr_rgb = np.array([r["auc_rgb"] for r in results])
    arr_pi_real = np.array([r["auc_pi_real"] for r in results])
    arr_pi_hat = np.array([r["auc_pi_hat"] for r in results])
    arr_recovered = np.array([r["recovered"] for r in results])
    arr_frac = np.array([r["frac_recovered"] for r in results
                          if not np.isnan(r["frac_recovered"])])

    print(f"\n[main] cross-seed:")
    print(f"  probe_rgb     = {arr_rgb.mean():.4f} +- {arr_rgb.std():.4f}")
    print(f"  probe_pi_real = {arr_pi_real.mean():.4f} +- {arr_pi_real.std():.4f}")
    print(f"  probe_pi_hat  = {arr_pi_hat.mean():.4f} +- {arr_pi_hat.std():.4f}")
    print(f"  recovered     = {arr_recovered.mean():+.4f} +- {arr_recovered.std():.4f}")
    if len(arr_frac) > 0:
        print(f"  frac of gap   = {arr_frac.mean()*100:.1f}% +- {arr_frac.std()*100:.1f}%")

    cs_recovered = arr_recovered.mean()
    cs_gap = arr_pi_real.mean() - arr_rgb.mean()
    cs_frac = (cs_recovered / cs_gap * 100) if cs_gap > 0.001 else float("nan")
    if cs_recovered >= 0.005:
        verdict = (f"**PROMISING.** Spatial-feature-map KD recovers "
                   f"{cs_recovered:+.4f} macro-AUC over RGB baseline. "
                   f"PI-TTA v2 has empirical support; the per-image "
                   f"feature-alignment design is worth implementing as "
                   f"a real test-time adaptation loop.")
    elif cs_recovered >= 0.002:
        verdict = (f"**MARGINAL.** Spatial KD recovers {cs_recovered:+.4f}, "
                   f"slightly above noise. PI-TTA v2 is a borderline "
                   f"investment; consider reframing as training-time KD "
                   f"(cleaner result for the same gain).")
    else:
        verdict = (f"**NULL.** Spatial KD recovers {cs_recovered:+.4f} "
                   f"(within ±0.005 noise). Combined with the embedding-"
                   f"KD null, PI-TTA v2 is empirically dead for capsule. "
                   f"NMI plan should rely on Track B + the four-variant "
                   f"C1 boundary as the methodological contribution.")

    md = []
    md.append("# PI-TTA v2, experiment 2 — spatial-feature-map KD report\n")
    md.append("**Date:** 2026-05-07")
    md.append("**Question:** Does a 1x1 conv adapter trained to map "
              "RGB spatial feature maps to +PI spatial feature maps "
              "recover the cell (b)/(b+) gap?")
    md.append("**Method:** stratified 2000-frame train subset → cache "
              "spatial maps from both backbones → train 1x1 conv adapter "
              "(MSE) → apply to all RGB frames → linear probe.")
    md.append("")
    md.append("## Per-seed results\n")
    md.append("| Seed | probe_rgb | probe_pi_real | probe_pi_hat | gap | recovered | frac of gap |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        frac = r["frac_recovered"]
        frac_str = f"{frac*100:.1f}%" if not np.isnan(frac) else "n/a"
        md.append(f"| {r['seed']} | {r['auc_rgb']:.4f} "
                  f"| {r['auc_pi_real']:.4f} | {r['auc_pi_hat']:.4f} "
                  f"| {r['gap']:+.4f} | {r['recovered']:+.4f} | {frac_str} |")
    md.append("")
    md.append(f"**Cross-seed mean:**")
    md.append(f"- probe_rgb     = {arr_rgb.mean():.4f} ± {arr_rgb.std():.4f}")
    md.append(f"- probe_pi_real = {arr_pi_real.mean():.4f} ± {arr_pi_real.std():.4f}")
    md.append(f"- probe_pi_hat  = {arr_pi_hat.mean():.4f} ± {arr_pi_hat.std():.4f}")
    md.append(f"- recovered     = {arr_recovered.mean():+.4f} ± {arr_recovered.std():.4f}")
    if not np.isnan(cs_frac):
        md.append(f"- frac of mean gap = {cs_frac:.1f}%")
    md.append("")
    md.append(f"## Verdict\n")
    md.append(verdict)
    md.append("")
    md.append(f"## Comparison with embedding-level KD (experiment 1)\n")
    md.append("| Layer | Recovery (cross-seed) |")
    md.append("|---|---:|")
    md.append("| Pooled-embedding (experiment 1) | +0.003 ± 0.023 (~11% of gap) |")
    md.append(f"| Spatial-feature-map (this) | "
              f"{arr_recovered.mean():+.4f} ± {arr_recovered.std():.4f} "
              f"({cs_frac:.1f}% of mean gap)" if not np.isnan(cs_frac)
              else f"| Spatial-feature-map (this) | {arr_recovered.mean():+.4f} ± "
                   f"{arr_recovered.std():.4f} |")
    md.append("")
    md.append(f"**Total compute:** {(time.time() - t0)/60:.1f} min for "
              f"{len(results)} seeds.")

    out = REPORT_DIR / "pi_tta_v2_spatial_kd_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out}")


if __name__ == "__main__":
    main()
