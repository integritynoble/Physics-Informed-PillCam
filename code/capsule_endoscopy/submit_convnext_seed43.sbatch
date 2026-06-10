#!/usr/bin/env bash
#SBATCH --job-name=gi-convnext-s43
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# Resume the cross-backbone batch on this cluster.
# Idempotent: the runner skips any run with test_metrics.json (already done).
# Filtered to "convnext_tiny", so seed-43 rgb + pi are the only NEW runs.

set -euo pipefail

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
mkdir -p "$NEW_ROOT/outputs/slurm_logs"

# --- modules + env ----------------------------------------------------------
module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

echo "[job] node=$(hostname) date=$(date)"
echo "[job] python=$(which python)"
python -c "import torch; print('[job] torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# --- preflight: data must be staged ----------------------------------------
DATA_DIR="$NEW_ROOT/data/stage2_data"
if [ ! -d "$DATA_DIR/train" ] || [ ! -d "$DATA_DIR/val" ] || [ ! -d "$DATA_DIR/test" ]; then
    echo "[job] ERROR: $DATA_DIR is missing train/val/test." >&2
    echo "[job] Run setup_kvasir_capsule.py first (see SETUP.md \xc2\xa72b)." >&2
    exit 1
fi
echo "[job] data OK: $(find $DATA_DIR/train -name '*.jpg' | wc -l) train frames, $(find $DATA_DIR/val -name '*.jpg' | wc -l) val, $(find $DATA_DIR/test -name '*.jpg' | wc -l) test"

# --- run --------------------------------------------------------------------
cd "$NEW_ROOT/code/Capsule-Endoscopy"
bash run_cross_backbone.sh convnext_tiny

echo "[job] finished at $(date)"
