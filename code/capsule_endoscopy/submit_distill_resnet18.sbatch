#!/usr/bin/env bash
#SBATCH --job-name=gi-disti-r18
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out
#SBATCH --error=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.err

# Distillation for resnet18 backbone, 6 seeds {41,42,43,44,45,47}.
# Outputs to stage2_distill_resnet18/ (seed 42) and stage2_distill_resnet18_seed{N}/ otherwise.

set -euo pipefail

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
mkdir -p "$NEW_ROOT/outputs/slurm_logs"

module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

echo "[job] node=$(hostname) date=$(date)"
python -c "import torch; print('[job] torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

cd "$NEW_ROOT/code/Capsule-Endoscopy"
bash run_distill.sh resnet18

echo "[job] finished at $(date)"
done_count=$(ls $NEW_ROOT/outputs/stage2_distill_resnet18*/test_metrics.json 2>/dev/null | wc -l)
echo "[job] resnet18 distill completed seeds: $done_count / 6"
