#!/usr/bin/env bash
#SBATCH --job-name=gi-dinov2-lp
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# DINOv2 linear-probe baseline for the medIA paper Table 2.
# Pre-empts the "did 2024-2025 foundation models obviate task-specific
# priors?" reviewer concern. Frozen DINOv2-base (vitb14, ImageNet-pretrained),
# 768-d -> 14-class linear classifier, 6 seeds, same data and 11-evaluable-class
# protocol as the rest of the paper.
#
# Phase 1: extract 768-d embeddings for all ~47K frames -> embeddings_cache.npz
#          (~1-2 GPU-h, one-time)
# Phase 2: train linear classifier per seed (~5 min each on V100) x 6 seeds
#          (~30 min wall total)
# Time budget: 6 h for safety.

set -euo pipefail

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
DATA_DIR="$NEW_ROOT/data/stage2_data"
OUT_ROOT="$NEW_ROOT/outputs/dinov2_linear"

mkdir -p "$NEW_ROOT/outputs/slurm_logs" "$OUT_ROOT"

module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

echo "[job] node=$(hostname)  date=$(date)"
python -c "import torch; print('[job] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

cd "$NEW_ROOT/code/Capsule-Endoscopy"

echo "[job] === Phase 1: extract DINOv2 embeddings (one-time) ==="
if [ -f "$OUT_ROOT/embeddings_cache.npz" ]; then
    echo "[job] cache already present, skipping extraction"
else
    python dinov2_linear_probe.py --extract_embeddings \
        --data_dir "$DATA_DIR" \
        --out_dir  "$OUT_ROOT"
fi

echo
echo "[job] === Phase 2: train linear probe across 6 seeds ==="
SEEDS=(41 42 43 44 45 47)
for seed in "${SEEDS[@]}"; do
    if [ -f "$OUT_ROOT/seed${seed}/test_metrics.json" ]; then
        echo "[job] SKIP  seed=${seed}  (already complete)"
        continue
    fi
    echo "[job] === train seed=${seed} === $(date)"
    python dinov2_linear_probe.py --train \
        --out_dir "$OUT_ROOT" \
        --seed    "$seed" \
        --epochs  40 \
        --lr      1e-3 \
        --batch_size 512
done

echo
echo "[job] === Cross-seed aggregate ==="
python - <<'PY'
import json
from pathlib import Path
import statistics as st
root = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/dinov2_linear")
seeds = [41, 42, 43, 44, 45, 47]
aucs = []
for s in seeds:
    p = root / f"seed{s}" / "test_metrics.json"
    if not p.exists():
        print(f"  seed{s}: missing")
        continue
    d = json.loads(p.read_text())
    aucs.append(d["macro_auc"])
    print(f"  seed{s}: macro_auc = {d['macro_auc']:.4f}  n_test={d['n_test_frames']}")
if len(aucs) >= 2:
    mean = st.mean(aucs); sd = st.stdev(aucs)
    print(f"\n  DINOv2 linear-probe cross-seed macro-AUC: {mean:.4f} +- {sd:.4f}  (n={len(aucs)})")
    summary = {"cross_seed_macro_auc_mean": mean, "cross_seed_macro_auc_sd": sd,
               "per_seed_macro_auc": dict(zip(seeds[:len(aucs)], aucs)),
               "n_seeds": len(aucs), "model": "dinov2_vitb14", "method": "linear_probe_frozen"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
PY

echo
echo "[job] finished at $(date)"
echo "[job] summary in: $OUT_ROOT/summary.json"
