from itertools import combinations
import math
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.tree import Tree
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
import string
from transformers import AutoTokenizer, AutoModel
from stanfordcorenlp import StanfordCoreNLP
from apted import APTED, Config
from apted.helpers import Tree as AptedTree

# Initialize SpaCy and NLTK resources
nltk.download('punkt')
nlp = StanfordCoreNLP('./stanford-corenlp-4.5.4')

# Load the SimCSE model and tokenizer
model_name = "princeton-nlp/sup-simcse-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load the DIMPLE dataset
ds = load_dataset("Jaehun/DIMPLE", split="train[0:100]")
texts = ds['text']
paraphrases = ds['paraphrase']


# Semantic Similarity (Cosine Similarity)
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def calculate_semantic_similarity(texts, paraphrases):
    print("Calculating Semantic Similarity using SimCSE...", flush=True)
    similarities = []
    for i, (text, paraphrase) in enumerate(zip(texts, paraphrases), 1):
        embed_text = get_embedding(text)
        embed_paraphrase = get_embedding(paraphrase)
        similarity = cosine_similarity(embed_text.detach().numpy(), embed_paraphrase.detach().numpy())[0][0]
        similarities.append(similarity)
        if i <= 10:
            print(f"Pair {i}: Semantic Similarity = {similarity:.4f}", flush=True)
    avg_similarity = np.mean(similarities)
    print(f"Average Semantic Similarity: {avg_similarity:.4f}", flush=True)
    return avg_similarity

# Lexical Diversity (H2 and H3 - Entropy-based Diversity)

# Calculate n-gram entropy
def calculate_ngram_entropy(words, n):
    if not words:
        return 0  # Return 0 if tokenization failed
    ngrams = list(zip(*[words[i:] for i in range(n)]))
    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())
    probabilities = [count / total_ngrams for count in ngram_counts.values()]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


# Mean Segmental Type-Token Ratio (MSTTR)
def calculate_msttr(texts, paraphrases, segment_length=50):
    print("Calculating MSTTR...", flush=True)
    msttr_values = []
    for i, (text, paraphrase) in enumerate(zip(texts, paraphrases), 1):
        combined_text = text + " " + paraphrase
        words = word_tokenize(combined_text)
        segments = [words[i:i + segment_length] for i in range(0, len(words), segment_length)]
        ttr = [len(set(segment)) / len(segment) for segment in segments if len(segment) > 0]
        msttr_values.append(np.mean(ttr))
        if i <= 10:
            print(f"MSTTR for pair {i}: {msttr_values[-1]:.4f}", flush=True)
    return np.mean(msttr_values)

# Jaccard Similarity
def calculate_jaccard_similarity(texts, paraphrases):
    print("Calculating Jaccard Similarity...", flush=True)
    jaccard_sims = []
    for i, (text, paraphrase) in enumerate(zip(texts, paraphrases), 1):
        similarity = len(set(text.split()) & set(paraphrase.split())) / len(set(text.split()) | set(paraphrase.split()))
        jaccard_sims.append(similarity)
        if i <= 10:
            print(f"Jaccard Similarity for pair {i}: {similarity:.4f}", flush=True)
    return np.mean(jaccard_sims)

# TED
class AptedConfig(Config):
    def rename(self, node1_label, node2_label):
        return 1 if node1_label != node2_label else 0

    def insert(self, node_label):
        return 1

    def delete(self, node_label):
        return 1

def nltk_tree_to_apted_tree(nltk_tree):
    node = AptedTree(nltk_tree.label())
    for child in nltk_tree:
        if isinstance(child, Tree):
            node.children.append(nltk_tree_to_apted_tree(child))
        else:
            node.children.append(AptedTree(child))
    return node

def prune_tree_to_depth(tree, max_depth, current_depth=1):
    if current_depth >= max_depth:
        return Tree(tree.label(), [])
    else:
        pruned_children = []
        for child in tree:
            if isinstance(child, Tree):
                pruned_child = prune_tree_to_depth(child, max_depth, current_depth + 1)
                pruned_children.append(pruned_child)
            else:
                pruned_children.append(child)
        return Tree(tree.label(), pruned_children)

def calculate_ted_average(texts, paraphrases):
    total_ted3_distance = 0
    total_tedf_distance = 0
    num_pairs = len(texts)

    config = AptedConfig()

    for text, paraphrase in zip(texts, paraphrases):
        # Parse sentences using Stanford CoreNLP
        parse_str1 = nlp.parse(text)
        parse_str2 = nlp.parse(paraphrase)

        # Convert parse strings to NLTK Trees
        tree1 = Tree.fromstring(parse_str1)
        tree2 = Tree.fromstring(parse_str2)

        # Compute TED-F (full trees)
        apted_tree1 = nltk_tree_to_apted_tree(tree1)
        apted_tree2 = nltk_tree_to_apted_tree(tree2)
        tedf_distance = APTED(apted_tree1, apted_tree2, config).compute_edit_distance()
        print(tedf_distance)
        total_tedf_distance += tedf_distance

        # Compute TED-3 (top-3 layers)
        pruned_tree1 = prune_tree_to_depth(tree1, 3)
        pruned_tree2 = prune_tree_to_depth(tree2, 3)
        apted_tree1_pruned = nltk_tree_to_apted_tree(pruned_tree1)
        apted_tree2_pruned = nltk_tree_to_apted_tree(pruned_tree2)
        ted3_distance = APTED(apted_tree1_pruned, apted_tree2_pruned, config).compute_edit_distance()
        print(ted3_distance)
        total_ted3_distance += ted3_distance

    # Calculate average distances
    average_ted3 = total_ted3_distance / num_pairs
    average_tedf = total_tedf_distance / num_pairs

    return average_ted3, average_tedf

# Evaluation function
def evaluate_dataset(texts, paraphrases):
    results = {}
    # results['Cosine Similarity'] = calculate_semantic_similarity(texts, paraphrases)
    # print("Calculating Entropy...", flush=True)
    # results['H2'] = np.mean([calculate_ngram_entropy(paraphrase, 2) for paraphrase in paraphrases])
    # results['H3'] = np.mean([calculate_ngram_entropy(paraphrase, 3) for paraphrase in paraphrases])
    # print(f"H2 is {results['H2']}")
    # print(f"H3 is {results['H3']}")
    # results['MSTTR'] = calculate_msttr(texts, paraphrases)
    # results['Jaccard Similarity'] = calculate_jaccard_similarity(texts, paraphrases)
    print("Calculating TED Scores...", flush=True)
    results['TED-3'], results['TED-F'] = calculate_ted_average(texts, paraphrases)
    return results

# Run the evaluation
results = evaluate_dataset(texts, paraphrases)

# Display final results
print("\nFinal Evaluation Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
