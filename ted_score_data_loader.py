import nltk
from nltk.corpus import stopwords
import re
from datasets import load_dataset

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

# Preprocess texts and paraphrases
ds = ds.map(lambda x: {
    'text': preprocess_text(x['text']),
    'paraphrase': preprocess_text(x['paraphrase'])
})

# Save the preprocessed dataset to JSON
ds.to_json("dimple_train.json", force_ascii=False, orient="records")
print("Preprocessed dataset saved as dimple_train.json")
