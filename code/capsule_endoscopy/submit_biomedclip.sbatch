#!/usr/bin/env bash
#SBATCH --job-name=gi-biomedclip
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# BiomedCLIP image-encoder linear probe on Kvasir-Capsule canonical split.
# Medical-domain foundation-model baseline; companion to DINOv2 linear probe.
# Embedding extraction (~1-2 GPU-h) + 6 linear-classifier seeds (~5 min each).

set -euo pipefail

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
DATA_DIR=/project/BME/Zaman_lab/s248103/stage2_data_canonical
OUT_DIR="$NEW_ROOT/outputs/biomedclip_linear"

mkdir -p "$NEW_ROOT/outputs/slurm_logs" "$OUT_DIR"
module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

echo "[job] node=$(hostname)  date=$(date)"
python -c "import torch; print('[job] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

cd "$NEW_ROOT/code/Capsule-Endoscopy"
python biomedclip_linear_probe.py --data_dir "$DATA_DIR" --out_dir "$OUT_DIR"
echo "[job] finished at $(date)"
