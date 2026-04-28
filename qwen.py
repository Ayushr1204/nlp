#!/usr/bin/env python3
"""
🔥 UNIVERSAL NLP HACKATHON SCRIPT 🔥
Handles CSV, TXT, and URL inputs with full NLP pipeline
Run with: python nlp_universal.py --input_type csv --path "your_dataset.csv"
"""

# ==================== IMPORTS ====================
import os
import re
import sys
import argparse
import warnings
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from transformers import pipeline

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

# Download NLTK resources (first run only)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# ==================== PLACEHOLDERS ====================
CSV_PATH = "your_dataset.csv"
TEXT_COLUMN = "TEXT_COLUMN"
LABEL_COLUMN = "LABEL_COLUMN"
TXT_PATH = "your_text_file.txt"
URL_PATH = "https://example.com/article"

# ==================== DATA LOADING ====================
def load_data(input_type: str, path: str) -> pd.DataFrame:
    """
    Load data from CSV, TXT file, or URL into unified DataFrame.
    Returns DataFrame with columns: ['text', 'label']
    """
    print(f"\n[LOAD] Loading data from {input_type.upper()}: {path}")
    
    try:
        if input_type == "csv":
            return _load_csv(path)
        elif input_type == "text":
            return _load_text(path)
        elif input_type == "url":
            return _load_url(path)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        # Return fallback dummy data for robustness
        return _create_dummy_data()


def _load_csv(path: str) -> pd.DataFrame:
    """Load and validate CSV dataset."""
    print(f"[CSV] Reading: {path}")
    df = pd.read_csv(path, encoding='utf-8', errors='ignore')
    print(f"[CSV] Columns found: {list(df.columns)}")
    
    if TEXT_COLUMN not in df.columns:
        raise KeyError(f"Text column '{TEXT_COLUMN}' not found. Available: {list(df.columns)}")
    
    # Create unified dataframe
    result_df = pd.DataFrame()
    result_df['text'] = df[TEXT_COLUMN].astype(str)
    
    if LABEL_COLUMN in df.columns:
        result_df['label'] = df[LABEL_COLUMN]
        print(f"[CSV] Labels found: {result_df['label'].nunique()} unique classes")
    else:
        print(f"[CSV] Warning: Label column '{LABEL_COLUMN}' not found. Creating dummy labels.")
        result_df['label'] = 0  # Dummy label
    
    print(f"[CSV] Loaded {len(result_df)} samples")
    return result_df


def _load_text(path: str) -> pd.DataFrame:
    """Load plain text file and split into sentences."""
    print(f"[TXT] Reading: {path}")
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"[TXT] Raw content length: {len(content)} characters")
    
    # Split into sentences for samples
    sentences = sent_tokenize(content)
    print(f"[TXT] Split into {len(sentences)} sentences")
    
    df = pd.DataFrame({'text': [s.strip() for s in sentences if s.strip()]})
    df['label'] = 0  # Dummy labels for unlabeled text
    print(f"[TXT] Created {len(df)} samples with dummy labels")
    return df


def _load_url(path: str) -> pd.DataFrame:
    """Scrape and clean text from web URL."""
    print(f"[URL] Scraping: {path}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(path, headers=headers, timeout=30)
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script, style, nav, footer elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    # Extract text and clean
    text = soup.get_text(separator=' ', strip=True)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    print(f"[URL] Extracted text length: {len(text)} characters")
    
    # Split into paragraphs/sentences
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short fragments
    
    print(f"[URL] Created {len(sentences)} valid samples")
    
    df = pd.DataFrame({'text': sentences})
    df['label'] = 0  # Dummy labels
    return df


def _create_dummy_data() -> pd.DataFrame:
    """Create fallback dummy data for robustness."""
    print("[FALLBACK] Creating dummy dataset for testing")
    dummy_texts = [
        "This is a sample positive text for testing.",
        "Another example with negative sentiment perhaps.",
        "Neutral statement without strong emotions.",
        "Great product, highly recommend it!",
        "Terrible experience, would not buy again."
    ]
    return pd.DataFrame({
        'text': dummy_texts,
        'label': [1, 0, 0, 1, 0]  # Balanced dummy labels
    })


# ==================== PREPROCESSING ====================
def preprocess_text(text: str, show_example: bool = False) -> str:
    """
    Full preprocessing pipeline:
    - Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    original = text[:100] if show_example else None
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs, mentions, special chars (keep letters and spaces)
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    processed = ' '.join(tokens)
    
    if show_example and original:
        print(f"\n[PREPROCESS] Example transformation:")
        print(f"  BEFORE: {original}...")
        print(f"  AFTER:  {processed[:100]}...")
    
    return processed


def apply_preprocessing(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Apply preprocessing to entire dataframe with progress."""
    print(f"\n[PREPROCESS] Processing {len(df)} texts...")
    
    # Show before/after for first sample
    if len(df) > 0:
        preprocess_text(df[text_col].iloc[0], show_example=True)
    
    df['text_clean'] = df[text_col].apply(preprocess_text)
    
    # Handle missing
    df['text_clean'] = df['text_clean'].fillna('')
    
    print(f"[PREPROCESS] Done. Sample cleaned text: '{df['text_clean'].iloc[0][:80]}...'")
    return df


# ==================== VECTORIZATION & SIMILARITY ====================
def vectorize_text(texts: List[str], max_features: int = 1000):
    """Convert texts to TF-IDF vectors."""
    print(f"\n[VECTORIZER] Creating TF-IDF vectors (max_features={max_features})")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    vectors = vectorizer.fit_transform(texts)
    print(f"[VECTORIZER] Shape: {vectors.shape}")
    print(f"[VECTORIZER] Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return vectorizer, vectors


def compute_similarity(vectors, n_samples: int = 3):
    """Compute and display cosine similarity matrix."""
    print(f"\n[SIMILARITY] Computing cosine similarity for first {n_samples} samples")
    
    # Limit to available samples
    n = min(n_samples, vectors.shape[0])
    subset = vectors[:n]
    
    sim_matrix = cosine_similarity(subset)
    
    print(f"[SIMILARITY] Matrix ({n}x{n}):")
    print(np.round(sim_matrix, 3))
    
    # Interpretation
    print("[SIMILARITY] Interpretation:")
    print("  - Diagonal = 1.0 (self-similarity)")
    print("  - Off-diagonal: closer to 1 = more similar, closer to 0 = less similar")
    print("  - Use this to detect duplicate/near-duplicate samples")
    
    return sim_matrix


# ==================== CLASSIFICATION ====================
def train_model(X, y, model_type: str = 'logistic'):
    """Train classification model if labels are meaningful."""
    print(f"\n[TRAIN] Training {model_type.upper()} classifier")
    
    # Check if labels are dummy (all same)
    if len(np.unique(y)) == 1:
        print("[TRAIN] Warning: Only one unique label detected. Skipping meaningful training.")
        return None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"[TRAIN] Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Select model
    if model_type == 'naive_bayes':
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Train
    model.fit(X_train, y_train)
    print(f"[TRAIN] Model trained successfully")
    
    return model, (X_test, y_test)


def evaluate_model(y_true, y_pred, labels=None):
    """Print evaluation metrics."""
    print(f"\n[EVALUATION] Classification Results:")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)
    
    print(f"\n  Classification Report:")
    target_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))] if labels is None else labels
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


# ==================== TRANSFORMER INFERENCE ====================
def run_transformer(texts: List[str], n_samples: int = 5, task: str = 'sentiment-analysis'):
    """Run pretrained transformer pipeline on samples."""
    print(f"\n[TRANSFORMER] Running pretrained '{task}' pipeline on {n_samples} samples")
    
    try:
        # Load pipeline (cached after first run)
        classifier = pipeline(task, model="distilbert-base-uncased-finetuned-sst-2-english" if task == 'sentiment-analysis' else None)
        
        # Sample texts
        samples = texts[:min(n_samples, len(texts))]
        
        print(f"[TRANSFORMER] Input samples:")
        for i, txt in enumerate(samples, 1):
            print(f"  {i}. {txt[:100]}...")
        
        # Run inference
        results = classifier(samples)
        
        print(f"\n[TRANSFORMER] Predictions:")
        for i, (txt, res) in enumerate(zip(samples, results), 1):
            if isinstance(res, list):
                res = res[0]  # Handle list output
            label = res.get('label', res)
            score = res.get('score', 'N/A')
            print(f"  {i}. Label: {label:15s} | Confidence: {score:.4f if isinstance(score, float) else score}")
        
        return results
        
    except Exception as e:
        print(f"[TRANSFORMER] Error (continuing without transformer): {e}")
        return None


# ==================== MAIN PIPELINE ====================
def main():
    """Orchestrate full NLP pipeline."""
    print("=" * 70)
    print("🔥 UNIVERSAL NLP HACKATHON SCRIPT 🔥")
    print("=" * 70)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Universal NLP Pipeline')
    parser.add_argument('--input_type', type=str, default='csv', 
                        choices=['csv', 'text', 'url'],
                        help='Type of input: csv, text, or url')
    parser.add_argument('--path', type=str, default=CSV_PATH,
                        help='Path to data (CSV file, TXT file, or URL)')
    parser.add_argument('--text_col', type=str, default=TEXT_COLUMN,
                        help='Name of text column (for CSV)')
    parser.add_argument('--label_col', type=str, default=LABEL_COLUMN,
                        help='Name of label column (for CSV)')
    parser.add_argument('--model', type=str, default='logistic',
                        choices=['logistic', 'naive_bayes'],
                        help='Classifier type')
    
    args = parser.parse_args()
    
    # Update global placeholders if provided
    global TEXT_COLUMN, LABEL_COLUMN
    TEXT_COLUMN = args.text_col
    LABEL_COLUMN = args.label_col
    
    # ==================== STEP 1: LOAD DATA ====================
    df = load_data(args.input_type, args.path)
    print(f"[MAIN] Loaded DataFrame: {df.shape}")
    print(df.head(2).to_string())
    
    # ==================== STEP 2: PREPROCESS ====================
    df = apply_preprocessing(df, text_col='text')
    
    # ==================== STEP 3: VECTORIZATION ====================
    vectorizer, tfidf_vectors = vectorize_text(df['text_clean'].tolist())
    
    # ==================== STEP 4: SIMILARITY ====================
    compute_similarity(tfidf_vectors, n_samples=3)
    
    # ==================== STEP 5: CLASSIFICATION ====================
    X = tfidf_vectors
    y = df['label'].values
    
    model, test_data = train_model(X, y, model_type=args.model)
    
    if model and test_data:
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred)
    else:
        print("[MAIN] Skipping classification evaluation (insufficient labels)")
    
    # ==================== STEP 6: TRANSFORMER ====================
    run_transformer(df['text'].tolist(), n_samples=5)
    
    # ==================== STEP 7: COMPARISON & INSIGHTS ====================
    print("\n" + "=" * 70)
    print("📊 COMPARISON & INSIGHTS")
    print("=" * 70)
    print("""
    🔹 Traditional ML (TF-IDF + Logistic Regression):
       ✅ Fast training/inference (seconds)
       ✅ Interpretable features (top TF-IDF terms)
       ✅ Low resource requirements
       ❌ Limited semantic understanding
       ❌ Requires manual feature engineering
    
    🔹 Transformer Models (BERT/DistilBERT):
       ✅ Deep semantic understanding
       ✅ State-of-the-art accuracy on many tasks
       ✅ Zero-shot capability with pretrained models
       ❌ Slower inference (GPU recommended)
       ❌ Higher memory/compute requirements
       ❌ Less interpretable (black-box)
    
    💡 RECOMMENDATION:
       - Start with TF-IDF + LR for quick prototyping
       - Use transformers for final model if accuracy is critical
       - Consider ensemble: use transformer predictions as features for traditional ML
    """)
    
    print("\n✅ Pipeline completed successfully!")
    print(f"📁 Output: Processed {len(df)} samples")
    print(f"🔧 Tip: Modify placeholders (CSV_PATH, TEXT_COLUMN, etc.) for your dataset")
    
    return df


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    result = main()