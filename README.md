# Dataset Evaluation

This is the repo for dataset evaluation of the Impossible Distillation - LoRA Replication Project

# Replication

Evaluate DIMPLE on semantic similarity and Tree Edit Distance

# Extension

Evaluation self-generated dataset on the same benchmarks

# Run the following code in your terminal

```bash/home/shihuis/dataset_evaluation/apted/src/main/java/node
bash setup.sh
sbatch evaluate.sh
bash setup_simcse.sh
sbatch sematical_sim.sh
bash setup_java.sh
sbatch ted_score.sh
```
