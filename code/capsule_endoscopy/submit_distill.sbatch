#!/usr/bin/env bash
#SBATCH --job-name=gi-distill-6s
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# Distillation training for cell (e+) - 6 seeds {41,42,43,44,45,47}
# Recipe: efficientnet_b0, 30 epochs, batch 24, lr 1e-4, distill_lambda 1.0, cosine, no AMP
# Idempotent (run_distill.sh skips seeds with test_metrics.json)
# Expected: ~30-60 min per seed on V100s -> ~3-6 h total

set -euo pipefail

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
mkdir -p "$NEW_ROOT/outputs/slurm_logs"

module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

echo "[job] node=$(hostname) date=$(date)"
echo "[job] python=$(which python)"
python -c "import torch; print('[job] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

DATA_DIR="$NEW_ROOT/data/stage2_data"
for split in train val test; do
    [ -d "$DATA_DIR/$split" ] || { echo "[job] ERROR: missing $DATA_DIR/$split" >&2; exit 1; }
done
echo "[job] data OK"

cd "$NEW_ROOT/code/Capsule-Endoscopy"
bash run_distill.sh

echo "[job] finished at $(date)"
done_count=$(ls $NEW_ROOT/outputs/stage2_distill_effb0*/test_metrics.json 2>/dev/null | wc -l)
echo "[job] completed seeds: $done_count / 6"
