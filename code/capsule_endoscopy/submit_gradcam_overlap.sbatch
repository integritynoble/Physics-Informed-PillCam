#!/usr/bin/env bash
#SBATCH --job-name=gi-gradcam-overlap
#SBATCH --partition=GPUv100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/slurm_logs/%x-%j.out

set -euo pipefail
module purge
module load anaconda3/2023.09-0 cuda/12.6.2 cudnn/9.2.0.82-12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate giproject

cd /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/Capsule-Endoscopy
python gradcam_prior_overlap.py --seed 42 --n_per_class 50
