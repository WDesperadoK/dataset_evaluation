#!/bin/bash
# Set up the environment for SimCSE dataset evaluation

# Stop on any error
set -e

# Load Conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
else
  echo "Conda not found. Please ensure Miniconda/Anaconda is installed."
  exit 1
fi

# Create a new Conda environment if it doesn't exist
if ! conda info --envs | grep -q "simcse_env"; then
  echo "Creating new Conda environment 'simcse_env'..."
  conda create -n simcse_env python=3.9 -y
fi

# Activate the environment
echo "Activating Conda environment 'simcse_env'..."
conda activate simcse_env

# Install dependencies from requirements file
if [ -f "sim_requirements.txt" ]; then
  echo "Installing dependencies from sim_requirements.txt..."
  pip install -r sim_requirements.txt
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu125
else
  echo "Requirements file 'sim_requirements.txt' not found. Exiting."
  exit 1
fi

# Install SimCSE
echo "Installing SimCSE..."
pip install simcse

echo "SimCSE environment setup complete."
