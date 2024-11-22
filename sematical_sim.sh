#!/bin/bash
#SBATCH --job-name=dimple_evaluation_sematical_similarity
#SBATCH --account=eecs498s006f24_class
#SBATCH --partition=spgpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30g
#SBATCH --output=dimple_evaluation_sematical_similarity.log

# Load Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simcse_env

# Optional: display GPU information
nvidia-smi

# Run the evaluation script
# python3 evaluate.py
python3 semantic_similarity.py