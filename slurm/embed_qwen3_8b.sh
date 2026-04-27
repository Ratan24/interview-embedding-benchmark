#!/bin/bash
# qwen3-8b (7.57B params, 4096-dim)
# GPU: A100 — model weights ~15GB, V100 too tight for long sequences
# batch_size=4, checkpoint_interval=50
# Expected runtime: ~5-7 hours for all 6 conditions
#
#SBATCH --job-name=micro1_embed_qwen3_8b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=/projects/PalikotLab/micro1/logs/embed_qwen3_8b_%j.out
#SBATCH --error=/projects/PalikotLab/micro1/logs/embed_qwen3_8b_%j.err

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

python embed_opensource.py --model qwen3-8b
