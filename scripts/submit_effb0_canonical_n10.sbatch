#!/usr/bin/env bash
#SBATCH --job-name=gi-effb0-canonical-n10
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# Full 10-seed retrain on the canonical Kvasir split (Windows-equivalent
# video assignments via --split_manifest). All 20 runs (10 seeds × 2 arms)
# use the headline recipe:  AdamW lr 1e-3, weight_decay 1e-4, cosine
# annealing, batch 32, 30 epochs, early-stop patience 5.
#
# Goal: produce a fully cluster-reproducible n=10 set of EffB0 checkpoints
# that anyone can regenerate from the released benchmark
# (kvasir_split_manifest_2026-05-18.json + this sbatch).
#
# Output dirs: cross_backbone/effb0_canonical_seed{N}_{rgb|pi}/
#
# Wall-clock: ~20-25 min per run × 20 runs = ~7-10h sequential.
# Time budget: 24h for safety.

set -euo pipefail

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
DATA_DIR=/project/BME/Zaman_lab/s248103/stage2_data_canonical
OUT_ROOT="$NEW_ROOT/outputs/cross_backbone"
GASTRO_DIR="$NEW_ROOT/code/gastroscopy_code_package"

mkdir -p "$NEW_ROOT/outputs/slurm_logs" "$OUT_ROOT"

module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

echo "[job] node=$(hostname)  date=$(date)"
python -c "import torch; print('[job] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
[ -d "$DATA_DIR" ] || { echo "[job] ERROR: canonical data dir missing: $DATA_DIR" >&2; exit 1; }

cd "$NEW_ROOT/code/Capsule-Endoscopy"

# Paper seeds + 4 new seeds = 10 total
SEEDS=(41 42 43 44 45 47 48 49 50 51)
ARMS=("rgb" "pi")

n_done=0; n_skip=0; n_fail=0
for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
        OUT_DIR="$OUT_ROOT/effb0_canonical_seed${seed}_${arm}"
        if [ -f "$OUT_DIR/best_model.pt" ] && [ -f "$OUT_DIR/test_metrics.json" ]; then
            echo "[job] SKIP seed=${seed} arm=${arm}  (already complete)"
            n_skip=$((n_skip + 1))
            continue
        fi
        mkdir -p "$OUT_DIR"
        echo
        echo "[job] === START effb0_canonical  seed=${seed}  arm=${arm} === $(date)"
        PI_ARG=""
        if [ "$arm" = "pi" ]; then PI_ARG="--use_physics_prior"; fi
        if python train_stage2_pi.py \
                --data_dir "$DATA_DIR" \
                --model_name efficientnet_b0 \
                --gastroscopy_code_dir "$GASTRO_DIR" \
                --output_dir "$OUT_DIR" \
                --seed "$seed" \
                --epochs 30 \
                --batch_size 32 \
                --lr 1e-3 \
                --weight_decay 1e-4 \
                --scheduler cosine \
                --pretrained \
                --early_stopping_patience 5 \
                --num_workers 8 \
                $PI_ARG 2>&1 | tail -8; then
            n_done=$((n_done + 1))
            echo "[job] === DONE effb0_canonical seed=${seed} arm=${arm} === $(date)"
        else
            n_fail=$((n_fail + 1))
            echo "[job] === FAILED effb0_canonical seed=${seed} arm=${arm} === $(date)"
        fi
    done
done

echo
echo "[job] summary: trained=${n_done}  skipped=${n_skip}  failed=${n_fail}"
echo "[job] finished at $(date)"
echo
echo "[job] next: post-processing inference + Galar zero-shot + aggregation"
