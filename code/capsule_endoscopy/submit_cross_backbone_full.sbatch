#!/usr/bin/env bash
#SBATCH --job-name=gi-xbb-full
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# Full cross-backbone replication on V100 32 GB:
#   ResNet-18 + ConvNeXt-Tiny  x  seeds 41,42,43  x  arms {rgb, pi} = 12 runs
# Recipe (kept IDENTICAL to the original GTX 1660 Ti runs for cross-seed comparability):
#   --lr 1e-4  --batch_size 32  --epochs 15  --scheduler cosine  no mixed_precision
# Runner is idempotent: skips any run with test_metrics.json (so resubmits resume).
# Expected wall-clock: ~24 h on a single V100s. Time budget set to 3 days for safety.

set -euo pipefail

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
mkdir -p "$NEW_ROOT/outputs/slurm_logs"

module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

echo "[job] node=$(hostname) date=$(date)"
echo "[job] python=$(which python)"
python -c "import torch; print('[job] torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

DATA_DIR="$NEW_ROOT/data/stage2_data"
for split in train val test; do
    if [ ! -d "$DATA_DIR/$split" ]; then
        echo "[job] ERROR: missing $DATA_DIR/$split" >&2
        exit 1
    fi
done
echo "[job] data OK: train=$(find $DATA_DIR/train -name '*.jpg' | wc -l) val=$(find $DATA_DIR/val -name '*.jpg' | wc -l) test=$(find $DATA_DIR/test -name '*.jpg' | wc -l)"

cd "$NEW_ROOT/code/Capsule-Endoscopy"
bash run_cross_backbone.sh

echo "[job] finished at $(date)"
echo "[job] completed runs: $(ls $NEW_ROOT/outputs/cross_backbone/*/test_metrics.json 2>/dev/null | wc -l) / 12"
