#!/usr/bin/env bash
#SBATCH --job-name=gi-effb0-headline
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# Retrain EfficientNet-B0 with the medIA paper's HEADLINE recipe so we have
# cluster-local checkpoints that reproduce the +0.023 in-domain lift and
# can be evaluated cross-vendor on Galar.
#
# Recipe (matches paper §3.3 verbatim):
#   AdamW   lr=1e-3   weight_decay=1e-4
#   cosine annealing
#   batch size 32
#   30 epochs (vs the matched-recipe 15)
#   early stopping patience=5 on val macro-F1-evaluable
#   inverse-frequency class weighting
#   deterministic
#
# This is DIFFERENT from the cross-backbone-matched recipe (lr 1e-4, 15 epochs)
# used for ResNet-18 and ConvNeXt-Tiny. Output dirs are named to avoid clobbering
# the matched-recipe runs:
#
#   cross_backbone/effb0_headline_seed{N}_{rgb|pi}
#
# Per-run wall: ~40 min on V100 (30 epochs vs 15). 12 runs sequential -> ~8h.
# Time budget: 16h for safety.

set -euo pipefail

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
DATA_DIR="$NEW_ROOT/data/stage2_data"
OUT_ROOT="$NEW_ROOT/outputs/cross_backbone"
GASTRO_DIR="$NEW_ROOT/code/gastroscopy_code_package"

mkdir -p "$NEW_ROOT/outputs/slurm_logs" "$OUT_ROOT"

module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

echo "[job] node=$(hostname)  date=$(date)"
python -c "import torch; print('[job] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
[ -d "$DATA_DIR" ] || { echo "[job] ERROR: data dir missing: $DATA_DIR" >&2; exit 1; }

cd "$NEW_ROOT/code/Capsule-Endoscopy"

SEEDS=(41 42 43 44 45 47)
ARMS=("rgb" "pi")

n_done=0; n_skipped=0; n_failed=0
for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
        OUT_DIR="$OUT_ROOT/effb0_headline_seed${seed}_${arm}"
        if [ -f "$OUT_DIR/best_model.pt" ] && [ -f "$OUT_DIR/test_metrics.json" ]; then
            echo "[job] SKIP  ${arm}  seed=${seed}  (already complete)"
            n_skipped=$((n_skipped + 1))
            continue
        fi
        mkdir -p "$OUT_DIR"
        echo "[job] === START effb0_headline  arm=${arm}  seed=${seed} === $(date)"
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
            echo "[job] === DONE  effb0_headline  arm=${arm}  seed=${seed} === $(date)"
        else
            n_failed=$((n_failed + 1))
            echo "[job] === FAILED effb0_headline  arm=${arm}  seed=${seed} === $(date)"
        fi
    done
done

echo
echo "[job] summary: trained=${n_done}  skipped=${n_skipped}  failed=${n_failed}"
echo "[job] finished at $(date)"
echo
echo "[job] next: compute in-domain mAUC then Galar zero-shot on these checkpoints"
echo "  cd $NEW_ROOT/code/Capsule-Endoscopy && python eval_cross_backbone_auc.py"
echo "  cd $NEW_ROOT/code/galar          && python eval_zero_shot.py --galar_test_dir $NEW_ROOT/data/galar_eval/test"
echo "  python aggregate_galar_results.py"
