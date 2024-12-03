#!/bin/bash
#SBATCH --job-name=dimple_evaluation_ted
#SBATCH --account=eecs498s006f24_class
#SBATCH --partition=spgpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30g
#SBATCH --output=dimple_evaluation_ted.log


# Optional: display GPU information
nvidia-smi

# Run the evaluation script
python ted_score_data_loader.py
javac -d apted/build/classes/java/main   -cp "apted/build/libs/apted.jar:libs/gson-2.10.jar:stanford-corenlp/stanford-corenlp-4.5.4/*"   custom/*.java
java -cp ".:apted/build/libs/apted.jar:apted/build/classes/java/main:libs/gson-2.10.jar:stanford-corenlp/stanford-corenlp-4.5.4/*"   at.unisalzburg.dbresearch.apted.custom.TED_with_CoreNLP_Dataset