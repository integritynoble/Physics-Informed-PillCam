#!/usr/bin/env bash
#SBATCH --job-name=gi-cnxt-chab
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# ConvNeXt-Tiny channel ablation: does the EffB0 training-time-bias
# mechanism transfer to a different backbone?
#
# Two single-channel arms (6 seeds each), matched-recipe (15 ep, lr 1e-4,
# batch 32, cosine, weight_decay 1e-4) on the canonical Kvasir split:
#   B. P_blood-only: RGB + P_blood     (--ablate_input_channels 4 zeroes Φ)
#   C. Φ-only      : RGB + Φ           (--ablate_input_channels 3 zeroes P_blood)
#
# Output dirs:
#   B  →  cross_backbone/convnext_tiny_seed{N}_pi_pblood_only
#   C  →  cross_backbone/convnext_tiny_seed{N}_pi_phi_only
#
# Full +PI and RGB-only baselines already exist as
#   convnext_tiny_seed{N}_pi / convnext_tiny_seed{N}_rgb (Table 4 cross-backbone).
#
# Wall-clock estimate: ~30 min/run × 12 runs = ~6 h.

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

cd "$NEW_ROOT/code/Capsule-Endoscopy"

SEEDS=(41 42 43 44 45 47)

run_one() {
    local seed="$1"
    local tag="$2"
    local ablate="$3"
    local OUT_DIR="$OUT_ROOT/convnext_tiny_seed${seed}_pi_${tag}"
    if [ -f "$OUT_DIR/best_model.pt" ] && [ -f "$OUT_DIR/test_metrics.json" ]; then
        echo "[job] SKIP seed=${seed} tag=${tag}  (already complete)"
        return 0
    fi
    mkdir -p "$OUT_DIR"
    echo
    echo "[job] === START seed=${seed} tag=${tag} ablate=${ablate} === $(date)"
    python train_stage2_pi.py \
        --data_dir "$DATA_DIR" \
        --model_name convnext_tiny \
        --gastroscopy_code_dir "$GASTRO_DIR" \
        --output_dir "$OUT_DIR" \
        --seed "$seed" \
        --epochs 15 \
        --batch_size 32 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --scheduler cosine \
        --pretrained \
        --early_stopping_patience 5 \
        --num_workers 8 \
        --use_physics_prior \
        --ablate_input_channels "$ablate" 2>&1 | tail -8
    echo "[job] === DONE  seed=${seed} tag=${tag} === $(date)"
}

# B. P_blood-only: zero Φ (channel 4)
for s in "${SEEDS[@]}"; do
    run_one "$s" pblood_only 4
done

# C. Φ-only: zero P_blood (channel 3)
for s in "${SEEDS[@]}"; do
    run_one "$s" phi_only 3
done

echo
echo "[job] finished at $(date)"
echo "[job] next: post-hoc mAUC + ablation aggregation"
