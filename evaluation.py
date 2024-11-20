from datasets import load_dataset
from collections import Counter
import math
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Load dataset
ds = load_dataset("Jaehun/DIMPLE")
subset = ds['train']
# [:1000000]
paraphrases = subset['paraphrase']
original_paraphrase = paraphrases

### Preprocessing Function ###
def preprocess_text(text):
    """
    Preprocess text: remove punctuation while keeping all words.
    :param text: Original text.
    :return: Cleaned text as a string.
    """
    # Remove punctuation using regex but keep all words
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation while retaining alphanumeric characters and spaces
    return text

# Preprocess all paraphrases
paraphrases = [preprocess_text(text) for text in paraphrases]

### H2 and H3 Entropy Functions ###
def calculate_ngram_entropy(paraphrases, n):
    """
    Calculate n-gram entropy (H_n) for a given list of paraphrases.
    :param paraphrases: List of strings (paraphrase column).
    :param n: Size of n-grams (e.g., 2 for bigrams, 3 for trigrams).
    :return: n-gram entropy (H_n).
    """
    ngram_counts = Counter()
    total_ngrams = 0

    # Count n-grams in all paraphrases
    for text in paraphrases:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        ngram_counts.update(ngrams)
        total_ngrams += len(ngrams)

    # Calculate entropy
    entropy = 0
    for ngram, count in ngram_counts.items():
        prob = count / total_ngrams
        entropy -= prob * math.log2(prob)

    return entropy

### MSTTR Calculation with Fixed-Length Segments ###
def calculate_msttr_fixed_segments(paraphrases, segment_length=100):
    """
    Calculate MSTTR for combined text with fixed-length segments.
    :param paraphrases: List of strings (paraphrase column).
    :param segment_length: Number of tokens per segment.
    :return: MSTTR value for the dataset.
    """
    # Combine all paraphrases into one large text
    combined_text = " ".join(paraphrases)
    #print(combined_text)
    tokens = word_tokenize(combined_text)
    print(f"Total tokens: {len(tokens)}")
    print(f"Total segments: {len(tokens) // 100}")
    print(tokens[:100]) 

    # Extract segments of fixed length
    ttr_values = []
    for i in range(0, len(tokens), segment_length):
        segment = tokens[i:i+segment_length]
        if len(segment) > 0:
            ttr = len(set(segment)) / len(segment)
            ttr_values.append(ttr)

    return sum(ttr_values) / len(ttr_values) if ttr_values else 0


def calculate_jaccard_similarity(source, paraphrase):
    """
    Calculate the Jaccard similarity at the token level between source and paraphrase texts.
    :param source: Original source text (string).
    :param paraphrase: Paraphrase text (string).
    :return: Jaccard similarity score (float).
    """
    # Tokenize the source and paraphrase into words
    source_tokens = set(word_tokenize(source))
    paraphrase_tokens = set(word_tokenize(paraphrase))
    
    # Calculate intersection and union of token sets
    intersection = source_tokens.intersection(paraphrase_tokens)
    union = source_tokens.union(paraphrase_tokens)
    
    # Compute Jaccard similarity
    return len(intersection) / len(union) if len(union) > 0 else 0.0

def calculate_jaccard_for_dataset(sources, paraphrases):
    """
    Calculate the average Jaccard similarity between source and paraphrase texts in the dataset.
    :param sources: List of source texts (strings).
    :param paraphrases: List of paraphrase texts (strings).
    :return: Average Jaccard similarity score (float).
    """
    jaccard_scores = [
        calculate_jaccard_similarity(source, paraphrase)
        for source, paraphrase in zip(sources, paraphrases)
    ]
    return sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

sources = subset['text']
avg_jaccard = calculate_jaccard_for_dataset(sources, original_paraphrase)
print(f"Average Jaccard similarity: {avg_jaccard:.4f}")

### Main Evaluation ###
# Calculate H2 (2-gram entropy)
h2 = calculate_ngram_entropy(paraphrases, 2)
print(f"H2 (2-gram entropy): {h2:.4f}")

# Calculate H3 (3-gram entropy)
h3 = calculate_ngram_entropy(paraphrases, 3)
print(f"H3 (3-gram entropy): {h3:.4f}")

# Calculate MSTTR with 100-word segments
msttr = calculate_msttr_fixed_segments(original_paraphrase, segment_length=100) * 100  # Convert to percentage
print(f"MSTTR (100-word segments): {msttr:.2f}")



