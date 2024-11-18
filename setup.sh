#!/bin/bash
# run this script for set-up
echo "Start Setting up"

source ~/miniconda3/bin/activate
conda init --all
conda create -n dataset_evaluation python=3.12
conda activate dataset_evaluation
conda install pip
pip install -r requirements.txt
python3 -m nltk.downloader punkt
python3 -m spacy download en_core_web_sm

echo "Set up complete"