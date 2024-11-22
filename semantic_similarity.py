import numpy as np
import re
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from simcse import SimCSE

# Initialize the SimCSE model
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the DIMPLE dataset
ds = load_dataset("Jaehun/DIMPLE", split="train")
texts = ds['text']
paraphrases = ds['paraphrase']
print("Dataset Loaded")

def preprocess_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_in_batches(data, batch_size=5000):
    """Yield batches of preprocessed text."""
    for i in range(0, len(data), batch_size):
        yield [preprocess_text(text) for text in data[i:i + batch_size]]

# Preprocess texts and paraphrases
texts = list(preprocess_in_batches(texts))
paraphrases = list(preprocess_in_batches(paraphrases))

def calculate_semantic_similarity_in_batches(texts, paraphrases, batch_size=5000):
    """Calculate cosine similarity in smaller chunks."""
    n = len(texts)
    similarities = []
    
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_paraphrases = paraphrases[i:i + batch_size]
        
        # Flatten any nested structures (if present)
        batch_texts = [str(text) for text in batch_texts]
        batch_paraphrases = [str(paraphrase) for paraphrase in batch_paraphrases]
        
        # Ensure valid batch lengths
        if len(batch_texts) > 0 and len(batch_paraphrases) > 0:
            sim = model.similarity(batch_texts, batch_paraphrases)
            similarities.append(np.mean(np.diag(sim)))  # Average of diagonal similarities

    return np.mean(similarities)  # Average across batches


def evaluate_dataset(texts, paraphrases):
    results = {}
    results['Cosine Similarity'] = calculate_semantic_similarity_in_batches(texts, paraphrases)
    return results

# Run the evaluation
results = evaluate_dataset(texts, paraphrases)

# Display final results
print("\nFinal Evaluation Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
