#!/usr/bin/env bash
#SBATCH --job-name=gi-cnxt-chab30
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# ConvNeXt-Tiny channel ablation at the LONGER EffB0-headline recipe
# (30 epochs, lr 1e-3, cosine, weight_decay 1e-4, batch 32, early stop 5).
# Companion to submit_convnext_channel_ablation.sbatch (the matched-recipe
# 15 ep / lr 1e-4 run).
#
# Tests whether the ConvNeXt-T result that P_blood-only > full +PI is
# recipe-dependent or architecture-fundamental. Training-curve evidence
# from the 15 ep runs (val-F1 peaks at ep 7-8) suggested they were NOT
# undertrained, but this directly verifies under the EffB0-equivalent
# recipe.
#
# Two arms (6 seeds each):
#   A. Full +PI    : RGB + P_blood + Φ  (5 channels, no ablation)
#                    → cross_backbone/convnext_tiny_seed{N}_pi_30ep_lr1e3
#   B. P_blood-only: --ablate_input_channels 4 (zeros Φ)
#                    → cross_backbone/convnext_tiny_seed{N}_pi_pblood_only_30ep_lr1e3
#
# Wall-clock estimate: ~95 min/run × 12 runs = ~19 h.

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
    local tag="$2"        # 'pi' or 'pblood_only'
    local extra_args="$3" # ablation flag string, possibly empty
    local OUT_DIR
    if [ "$tag" = "pi" ]; then
        OUT_DIR="$OUT_ROOT/convnext_tiny_seed${seed}_pi_30ep_lr1e3"
    else
        OUT_DIR="$OUT_ROOT/convnext_tiny_seed${seed}_pi_${tag}_30ep_lr1e3"
    fi
    if [ -f "$OUT_DIR/best_model.pt" ] && [ -f "$OUT_DIR/test_metrics.json" ]; then
        echo "[job] SKIP seed=${seed} tag=${tag}  (already complete)"
        return 0
    fi
    mkdir -p "$OUT_DIR"
    echo
    echo "[job] === START seed=${seed} tag=${tag} === $(date)"
    python train_stage2_pi.py \
        --data_dir "$DATA_DIR" \
        --model_name convnext_tiny \
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
        --use_physics_prior \
        $extra_args 2>&1 | tail -8
    echo "[job] === DONE  seed=${seed} tag=${tag} === $(date)"
}

# A. Full +PI (no ablation)
for s in "${SEEDS[@]}"; do
    run_one "$s" pi ""
done

# B. P_blood-only: zero Φ (channel 4)
for s in "${SEEDS[@]}"; do
    run_one "$s" pblood_only "--ablate_input_channels 4"
done

echo
echo "[job] finished at $(date)"
echo "[job] next: eval_cross_backbone_auc.py + comparison vs the matched-recipe ablation"
