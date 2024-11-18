import math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import spacy
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from datasets import load_dataset

# Initialize SpaCy and NLTK resources
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Load the DIMPLE dataset and only select a subset of 10 rows
ds = load_dataset("Jaehun/DIMPLE", split="train[:100]")  # Load the first 100 samples
texts = ds['text']
paraphrases = ds['paraphrase']

# Semantic Similarity (Cosine Similarity)
def calculate_cosine_similarity(texts, paraphrases):
    sentence_embeddings_texts = [nlp(text).vector for text in texts]
    sentence_embeddings_paraphrases = [nlp(paraphrase).vector for paraphrase in paraphrases]
    cosine_similarities = [
        cosine_similarity([sentence_embeddings_texts[i]], [sentence_embeddings_paraphrases[i]])[0][0]
        for i in range(len(texts))
    ]
    avg_cosine_similarity = np.mean(cosine_similarities)
    return avg_cosine_similarity

# Lexical Diversity (H2 and H3 - Entropy-based Diversity)
def calculate_entropy(text, n=2):
    words = text.split()
    n_grams = zip(*[words[i:] for i in range(n)])
    n_gram_counts = Counter(n_grams)
    total_n_grams = sum(n_gram_counts.values())
    entropy = -sum((count / total_n_grams) * math.log2(count / total_n_grams) for count in n_gram_counts.values())
    return entropy

# Lexical Diversity (Mean Segmental Type-Token Ratio - MSTTR)
def calculate_msttr(texts, paraphrases, segment_length=50):
    msttr_values = []
    for text, paraphrase in zip(texts, paraphrases):
        combined_text = text + " " + paraphrase  # Combine for evaluation
        words = word_tokenize(combined_text)
        segments = [words[i:i + segment_length] for i in range(0, len(words), segment_length)]
        ttr = [len(set(segment)) / len(segment) for segment in segments if len(segment) > 0]
        msttr_values.append(np.mean(ttr))
    return np.mean(msttr_values)

# Lexical Diversity (Jaccard Similarity)
def calculate_jaccard_similarity(texts, paraphrases):
    jaccard_sims = [
        len(set(text.split()) & set(paraphrase.split())) / len(set(text.split()) | set(paraphrase.split()))
        for text, paraphrase in zip(texts, paraphrases)
    ]
    return np.mean(jaccard_sims)

# Syntactic Diversity (TED-3 and TED-F)
def calculate_ted_similarity(texts, paraphrases):
    ted_3_scores = []
    ted_f_scores = []
    smoothing = SmoothingFunction().method1  # Apply smoothing

    for text, paraphrase in zip(texts, paraphrases):
        doc1 = nlp(text)
        doc2 = nlp(paraphrase)
        
        # Sentence-based comparison for TED-F
        sentences1 = [sent.text for sent in doc1.sents]
        sentences2 = [sent.text for sent in doc2.sents]
        
        # Full sentence comparison (TED-F) with smoothing
        bleu_scores_f = [
            sentence_bleu([sent1], sent2, smoothing_function=smoothing)
            for sent1, sent2 in zip(sentences1, sentences2)
        ]
        ted_f_scores.append(sum(bleu_scores_f) / len(bleu_scores_f) if bleu_scores_f else 0)
        
        # 3-gram comparison (TED-3) with smoothing
        ngrams1 = [text[i:i+3] for text in sentences1 for i in range(len(text) - 2)]
        ngrams2 = [text[i:i+3] for text in sentences2 for i in range(len(text) - 2)]
        bleu_scores_3 = [
            sentence_bleu([ngram1], ngram2, smoothing_function=smoothing)
            for ngram1, ngram2 in zip(ngrams1, ngrams2)
        ]
        ted_3_scores.append(sum(bleu_scores_3) / len(bleu_scores_3) if bleu_scores_3 else 0)
    
    ted_f_avg = sum(ted_f_scores) / len(ted_f_scores) if ted_f_scores else 0
    ted_3_avg = sum(ted_3_scores) / len(ted_3_scores) if ted_3_scores else 0
    return ted_3_avg, ted_f_avg

# Evaluation function
def evaluate_dataset(texts, paraphrases):
    results = {}
    results['Cosine Similarity'] = calculate_cosine_similarity(texts, paraphrases)
    results['H2'] = np.mean([calculate_entropy(text, n=2) for text in texts])
    results['H3'] = np.mean([calculate_entropy(text, n=3) for text in texts])
    results['MSTTR'] = calculate_msttr(texts, paraphrases)
    results['Jaccard Similarity'] = calculate_jaccard_similarity(texts, paraphrases)
    results['TED-3'], results['TED-F'] = calculate_ted_similarity(texts, paraphrases)
    return results

# Run the evaluation on the subset of the DIMPLE dataset
results = evaluate_dataset(texts, paraphrases)

# Display results
print("Evaluation Results for DIMPLE Dataset:")
for metric, value in results.items():
    print(f"{metric}: {value:.2f}")
