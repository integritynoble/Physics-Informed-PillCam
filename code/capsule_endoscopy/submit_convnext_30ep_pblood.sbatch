#!/usr/bin/env bash
#SBATCH --job-name=gi-cnxt-30-pb
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# ConvNeXt-T P_blood-only at the EffB0 headline recipe (30 ep, lr 1e-3).
# Sister job to submit_convnext_30ep_pi.sbatch — runs in parallel to
# halve wall time vs the original combined-arms sbatch.
# 6 seeds × ~95 min/run = ~9.5 h.

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
for seed in "${SEEDS[@]}"; do
    OUT_DIR="$OUT_ROOT/convnext_tiny_seed${seed}_pi_pblood_only_30ep_lr1e3"
    if [ -f "$OUT_DIR/best_model.pt" ] && [ -f "$OUT_DIR/test_metrics.json" ]; then
        echo "[job] SKIP seed=${seed} (already complete)"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "[job] === START seed=${seed} arm=pblood_only === $(date)"
    python train_stage2_pi.py \
        --data_dir "$DATA_DIR" --model_name convnext_tiny \
        --gastroscopy_code_dir "$GASTRO_DIR" --output_dir "$OUT_DIR" \
        --seed "$seed" --epochs 30 --batch_size 32 --lr 1e-3 \
        --weight_decay 1e-4 --scheduler cosine --pretrained \
        --early_stopping_patience 5 --num_workers 8 \
        --use_physics_prior --ablate_input_channels 4 2>&1 | tail -8
    echo "[job] === DONE  seed=${seed} arm=pblood_only === $(date)"
done
echo "[job] finished at $(date)"
