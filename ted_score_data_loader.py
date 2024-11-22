from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Jaehun/DIMPLE", split="train")

# Save to JSON
dataset.to_json("dimple_train.json")
