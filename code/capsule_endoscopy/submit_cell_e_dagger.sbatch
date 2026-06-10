#!/usr/bin/env bash
#SBATCH --job-name=gi-cell-e-dagger
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# Cell (e+) pipeline: extract 1280-d pooled features from the 6 distill backbones
# (deploy_mode=True), then train the 6-seed temporal head with C3 residual.
# Expected: ~1 h for extraction across 6 seeds + ~15 min for temporal training on V100s.

set -euo pipefail

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
mkdir -p "$NEW_ROOT/outputs/slurm_logs"

module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

echo "[job] node=$(hostname) date=$(date)"
python -c "import torch; print('[job] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

cd /home2/s248103/abraham/GI/GI_Multi_Task/paper/nature-machine-intelligence/code/temporal

echo "[job] === step 1: build_embedding_cache_distill.py ==="
python build_embedding_cache_distill.py

echo "[job] === step 2: train_cell_e_dagger.py ==="
python train_cell_e_dagger.py

echo "[job] finished at $(date)"
