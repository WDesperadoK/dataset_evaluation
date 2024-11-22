#!/bin/bash
# Set up the environment for dataset evaluation

# Load Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create a new Conda environment if it doesn't exist
if ! conda info --envs | grep -q "dataset_evaluation"; then
  conda create -n dataset_evaluation python=3.9 -y
fi

# Activate the environment
conda activate dataset_evaluation

# Install dependencies
pip install -r requirements.txt

# Install Stanford CoreNLP dependencies (if not installed)
if [ ! -d "stanford-corenlp-4.5.4" ]; then
  
  unzip stanford-corenlp-4.5.4.zip
fi

echo "Environment setup complete."
