#!/bin/bash
# The interpreter used to execute the script

# SLURM job options
#SBATCH --job-name=dimple_evaluation          # Job name
#SBATCH --account=eecs498s006f24_class        # Account name
#SBATCH --partition=spgpu                     # Partition name
#SBATCH --time=08:00:00                       # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gpus=a40:1                          # Type and number of GPUs
#SBATCH --cpus-per-task=4                     # Number of CPUs per task
#SBATCH --mem=30g                             # Memory allocated to the job
#SBATCH --output=dimple_evaluation.log        # Output log file

# Load Conda and activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dataset_evaluation

# Optional: display GPU information
nvidia-smi

# Run the evaluation script
python3 evaluate.py