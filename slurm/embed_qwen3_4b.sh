#!/bin/bash
# qwen3-4b (4.02B params, 2560-dim)
# GPU: V100-SXM2 32GB — model weights ~8GB, batch_size=4
# Expected runtime: ~4-6 hours for all 6 conditions
#
#SBATCH --job-name=micro1_embed_qwen3_4b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=/projects/PalikotLab/micro1/logs/embed_qwen3_4b_%j.out
#SBATCH --error=/projects/PalikotLab/micro1/logs/embed_qwen3_4b_%j.err

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

python embed_opensource.py --model qwen3-4b
