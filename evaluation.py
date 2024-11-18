from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
from collections import Counter
import numpy as np
import zss

ds = load_dataset("Jaehun/DIMPLE")
# Take a subset of the dataset for evaluation
ds = ds['train'][:100] 
# print(train_subset)

# Cosine Similarity

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(1)

def cosine_similarity(embed1, embed2):
    return (embed1 @ embed2.T) / (embed1.norm() * embed2.norm())

# Lexical Diversity

def ngram_entropy(text, n):
    words = nltk.word_tokenize(text)
    grams = nltk.ngrams(words, n)
    frequency = Counter(grams)
    probabilities = [f / sum(frequency.values()) for f in frequency.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def msttr(text, segment_length=100):
    words = nltk.word_tokenize(text)
    if len(words) < segment_length:
        return float('nan')  # Not enough tokens to evaluate MSTTR
    segment_types = [len(set(words[i:i + segment_length])) for i in range(0, len(words), segment_length)]
    return np.mean(segment_types) / segment_length

def jaccard_similarity(text1, text2):
    tokens1 = set(nltk.word_tokenize(text1))
    tokens2 = set(nltk.word_tokenize(text2))
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

# Syntactic Diversity

def parse_to_tree(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    tree = nltk.chunk.ne_chunk(pos_tags)
    return tree

def simple_distance(A, B):
    return 1 if A != B else 0

def compute_tree_edit_distance(tree1, tree2):
    return zss.simple_distance(tree1, tree2, lambda n: n, lambda n: n.label(), simple_distance)

def compute_ted_top_layers(tree1, tree2, depth=3):
    """Compute tree edit distance only up to a certain depth."""
    def get_children(node):
        if node.height() <= depth:
            return list(node)
        else:
            return []

    return zss.simple_distance(tree1, tree2, get_children=get_children, get_label=lambda x: x.label(), label_dist=simple_distance)


results = []
for i in range(len(ds['text'])):
    text1 = ds['text'][i]
    paraphrase = ds['paraphrase'][i]

    # Compute embeddings for semantic similarity
    embed1 = get_embedding(text1)
    embed2 = get_embedding(paraphrase)
    semantic_similarity = cosine_similarity(embed1, embed2).item()

    # Lexical diversity
    h2_entropy = ngram_entropy(text1, 2)
    h3_entropy = ngram_entropy(text1, 3)
    msttr_value = msttr(text1)
    jaccard_sim = jaccard_similarity(text1, paraphrase)

    # Syntactic diversity (PROBLEMS! counldn't run)
    tree1 = parse_to_tree(text1)
    tree2 = parse_to_tree(paraphrase)
    ted_f = compute_tree_edit_distance(tree1, tree2)
    ted_3 = compute_ted_top_layers(tree1, tree2, depth=3)

    result = {
        'semantic_similarity': semantic_similarity,
        'h2_entropy': h2_entropy,
        'h3_entropy': h3_entropy,
        'msttr': msttr_value,
        'jaccard_similarity': jaccard_sim,
        'ted_3': ted_3,
        'ted_f': ted_f
    }
    results.append(result)
    print(result)

# Average results
sum_semantic_similarity = 0
sum_h2_entropy = 0
sum_h3_entropy = 0
sum_msttr = 0
sum_jaccard = 0
sum_ted_3 = 0
sum_ted_f = 0

# Sum up all results
for result in results:
    sum_semantic_similarity += result['semantic_similarity']
    sum_h2_entropy += result['h2_entropy']
    sum_h3_entropy += result['h3_entropy']
    sum_msttr += result['msttr']
    sum_jaccard += result['jaccard_similarity']
    sum_ted_3 += result['ted_3']
    sum_ted_f += result['ted_f']

# Compute averages
average_semantic_similarity = sum_semantic_similarity / len(results)
average_h2_entropy = sum_h2_entropy / len(results)
average_h3_entropy = sum_h3_entropy / len(results)
average_msttr = sum_msttr / len(results)
average_jaccard = sum_jaccard / len(results)
average_ted_3 = sum_ted_3 / len(results)
average_ted_f = sum_ted_f / len(results)

# Store or print averages
average_results = {
    'average_semantic_similarity': average_semantic_similarity,
    'average_h2_entropy': average_h2_entropy,
    'average_h3_entropy': average_h3_entropy,
    'average_msttr': average_msttr,
    'average_jaccard_similarity': average_jaccard,
    'average_ted_3': average_ted_3,
    'average_ted_f': average_ted_f
}

print(average_results)


