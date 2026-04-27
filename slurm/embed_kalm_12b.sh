#!/bin/bash
# kalm-12b (11.76B params, 3840-dim)
# GPU: A100 required — model weights ~23.5GB, does NOT fit on V100-32GB
# batch_size=2, checkpoint_interval=50
# Expected runtime: ~6-8 hours for all 6 conditions
#
#SBATCH --job-name=micro1_embed_kalm_12b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=/projects/PalikotLab/micro1/logs/embed_kalm_12b_%j.out
#SBATCH --error=/projects/PalikotLab/micro1/logs/embed_kalm_12b_%j.err

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

export HF_HOME=/projects/PalikotLab/hf_cache
export TRANSFORMERS_CACHE=/projects/PalikotLab/hf_cache
export HF_DATASETS_CACHE=/projects/PalikotLab/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source activate /projects/PalikotLab/envs/micro1
export PATH=/projects/PalikotLab/envs/micro1/bin:$PATH

mkdir -p /projects/PalikotLab/micro1/logs
cd /projects/PalikotLab/micro1

python embed_opensource.py --model kalm-12b
