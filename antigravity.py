"""
================================================================================
 UNIVERSAL NLP PIPELINE — Hackathon-Ready
 Supports: CSV | Plain Text | Web URL
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import time
import string
import warnings
import numpy as np
import pandas as pd

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Web scraping
import requests
from bs4 import BeautifulSoup

# Transformers
from transformers import pipeline as hf_pipeline

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# NLTK DOWNLOADS (run once)
# ──────────────────────────────────────────────────────────────────────────────
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


# ==============================================================================
# PLACEHOLDERS — CHANGE THESE BEFORE RUNNING
# ==============================================================================
CSV_PATH        = "your_dataset.csv"        # path to your CSV file
TEXT_COLUMN     = "TEXT_COLUMN"              # column name containing text
LABEL_COLUMN    = "LABEL_COLUMN"            # column name containing labels
TEXT_FILE_PATH  = "your_text_file.txt"       # path to a .txt file
URL             = "https://example.com/article"  # URL to scrape


# ==============================================================================
# 1. DATA LOADING — load_data()
# ==============================================================================
def load_data(input_type: str) -> pd.DataFrame:
    """
    Load data from CSV, plain text, or web URL and return a unified DataFrame
    with columns: ['text', 'label'].
    """
    print(f"\n{'='*70}")
    print(f"  📂  LOADING DATA  —  input_type = '{input_type}'")
    print(f"{'='*70}")

    if input_type == "csv":
        return _load_csv()
    elif input_type == "text":
        return _load_text()
    elif input_type == "url":
        return _load_url()
    else:
        raise ValueError(f"Unknown input_type: '{input_type}'. Use 'csv', 'text', or 'url'.")


def _load_csv() -> pd.DataFrame:
    """Load a CSV file and map to the common schema."""
    try:
        raw = pd.read_csv(CSV_PATH)
        print(f"  ✅ Loaded CSV  →  {raw.shape[0]} rows, {raw.shape[1]} columns")
        print(f"  Columns: {list(raw.columns)}")

        df = pd.DataFrame()
        df["text"] = raw[TEXT_COLUMN].astype(str)

        if LABEL_COLUMN in raw.columns:
            df["label"] = raw[LABEL_COLUMN]
            print(f"  Labels found  →  {df['label'].nunique()} unique classes")
        else:
            df["label"] = np.nan
            print("  ⚠️  No label column found — labels set to NaN")

        return df

    except FileNotFoundError:
        print(f"  ❌ File not found: {CSV_PATH}")
        raise
    except KeyError as e:
        print(f"  ❌ Column not found: {e}. Available columns listed above.")
        raise


def _load_text() -> pd.DataFrame:
    """Load a plain text file, split into sentences, return DataFrame."""
    try:
        with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        sentences = sent_tokenize(content)
        # Remove very short fragments (< 10 chars)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]

        print(f"  ✅ Loaded text file  →  {len(sentences)} sentences extracted")
        df = pd.DataFrame({"text": sentences, "label": np.nan})
        return df

    except FileNotFoundError:
        print(f"  ❌ File not found: {TEXT_FILE_PATH}")
        raise


def _load_url() -> pd.DataFrame:
    """Scrape a web URL, extract text paragraphs, return DataFrame."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(URL, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        paragraphs = [p for p in paragraphs if len(p) >= 20]

        # Fallback: split all visible text into sentences
        if len(paragraphs) < 3:
            full_text = soup.get_text(separator="\n", strip=True)
            paragraphs = sent_tokenize(full_text)
            paragraphs = [s.strip() for s in paragraphs if len(s.strip()) >= 20]

        print(f"  ✅ Scraped URL  →  {len(paragraphs)} text segments extracted")
        df = pd.DataFrame({"text": paragraphs, "label": np.nan})
        return df

    except requests.RequestException as e:
        print(f"  ❌ Failed to fetch URL: {e}")
        raise


# ==============================================================================
# 2. PREPROCESSING — preprocess_text()
# ==============================================================================
STOP_WORDS = set()
LEMMATIZER = WordNetLemmatizer()

def _init_nlp_resources():
    """Lazy-init stopwords (avoids import-time errors)."""
    global STOP_WORDS
    if not STOP_WORDS:
        try:
            STOP_WORDS = set(stopwords.words("english"))
        except Exception:
            STOP_WORDS = set()


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single text string:
      1. Lowercase
      2. Remove punctuation
      3. Tokenize
      4. Remove stopwords
      5. Lemmatize
    """
    _init_nlp_resources()

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Stopword removal
    tokens = [t for t in tokens if t not in STOP_WORDS and t.isalpha()]

    # 5. Lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the entire DataFrame, show before/after."""
    print(f"\n{'='*70}")
    print("  🧹  PREPROCESSING")
    print(f"{'='*70}")

    # Handle missing values
    before_count = len(df)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"  Rows before cleaning: {before_count}  →  after: {len(df)}")

    # Show BEFORE samples
    print("\n  ── BEFORE preprocessing ──")
    for i, row in df.head(3).iterrows():
        print(f"    [{i}] {row['text'][:120]}...")

    # Apply preprocessing
    df["text_clean"] = df["text"].apply(preprocess_text)

    # Show AFTER samples
    print("\n  ── AFTER preprocessing ──")
    for i, row in df.head(3).iterrows():
        print(f"    [{i}] {row['text_clean'][:120]}...")

    return df


# ==============================================================================
# 3. TF-IDF + SIMILARITY — vectorize_text() / compute_similarity()
# ==============================================================================
def vectorize_text(df: pd.DataFrame):
    """Fit a TF-IDF vectorizer and return the matrix + vectorizer."""
    print(f"\n{'='*70}")
    print("  📊  TF-IDF VECTORIZATION")
    print(f"{'='*70}")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df["text_clean"])

    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Vocabulary size:     {len(vectorizer.vocabulary_)}")

    return tfidf_matrix, vectorizer


def compute_similarity(tfidf_matrix, n_samples: int = 3):
    """
    Compute cosine similarity between the first n_samples.
    
    Interpretation:
      - 1.0  → identical documents
      - 0.0  → completely unrelated
      - > 0.5 → fairly similar topics/language
    """
    print(f"\n{'='*70}")
    print(f"  🔗  COSINE SIMILARITY (first {n_samples} samples)")
    print(f"{'='*70}")

    n = min(n_samples, tfidf_matrix.shape[0])
    subset = tfidf_matrix[:n]
    sim_matrix = cosine_similarity(subset)

    sim_df = pd.DataFrame(
        sim_matrix,
        index=[f"Doc_{i}" for i in range(n)],
        columns=[f"Doc_{i}" for i in range(n)],
    )
    print(sim_df.round(4).to_string())

    # Quick interpretation
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i][j]
            level = "HIGH" if score > 0.5 else ("MODERATE" if score > 0.2 else "LOW")
            print(f"  Doc_{i} ↔ Doc_{j}:  {score:.4f}  ({level} similarity)")

    return sim_matrix


# ==============================================================================
# 4. CLASSIFICATION MODEL — train_model() / evaluate_model()
# ==============================================================================
def train_model(tfidf_matrix, labels):
    """
    Train Logistic Regression + Naive Bayes if real labels exist.
    Returns trained models and test data for evaluation.
    """
    print(f"\n{'='*70}")
    print("  🤖  CLASSIFICATION MODEL TRAINING")
    print(f"{'='*70}")

    has_labels = labels.notna().any()

    if not has_labels:
        print("  ⚠️  No real labels available.")
        print("  → Creating dummy binary labels (0/1) for demonstration purposes.")
        labels = pd.Series(np.random.randint(0, 2, size=tfidf_matrix.shape[0]))

    # Encode labels if they're strings
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels.astype(str))
    class_names = le.classes_

    print(f"  Classes: {list(class_names)}")
    print(f"  Total samples: {len(y)}")

    # Need at least 2 classes for classification
    if len(set(y)) < 2:
        print("  ❌ Only one class present — cannot train a classifier.")
        return None, None, None, None, None

    # Split
    test_size = min(0.2, max(0.1, 2 / len(y)))  # handle tiny datasets
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]}   Test: {X_test.shape[0]}")

    # Logistic Regression
    print("\n  ── Logistic Regression ──")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_time = time.time() - t0
    print(f"  Training time: {lr_time:.3f}s")

    # Naive Bayes
    print("\n  ── Multinomial Naive Bayes ──")
    t0 = time.time()
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_time = time.time() - t0
    print(f"  Training time: {nb_time:.3f}s")

    return lr, nb, X_test, y_test, class_names


def evaluate_model(model, X_test, y_test, class_names, model_name="Model"):
    """Evaluate a trained classifier and print metrics."""
    print(f"\n  ── Evaluation: {model_name} ──")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")

    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {cm}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], zero_division=0))

    return acc


# ==============================================================================
# 5. TRANSFORMER MODEL — run_transformer()
# ==============================================================================
def run_transformer(df: pd.DataFrame, n_samples: int = 5):
    """
    Run a pretrained HuggingFace sentiment-analysis pipeline on a few samples.
    Uses distilbert-base-uncased-finetuned-sst-2-english by default.
    """
    print(f"\n{'='*70}")
    print(f"  🚀  TRANSFORMER INFERENCE  (top {n_samples} samples)")
    print(f"{'='*70}")

    try:
        t0 = time.time()
        classifier = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,  # CPU; set to 0 for GPU
        )
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.2f}s")

        samples = df["text"].head(n_samples).tolist()

        t0 = time.time()
        results = classifier(samples, truncation=True, max_length=512)
        infer_time = time.time() - t0

        print(f"  Inference time: {infer_time:.3f}s  ({infer_time/len(samples):.3f}s per sample)\n")

        for i, (text, res) in enumerate(zip(samples, results)):
            preview = text[:100] + ("..." if len(text) > 100 else "")
            print(f"  [{i}] {preview}")
            print(f"      → {res['label']}  (confidence: {res['score']:.4f})\n")

        return results, infer_time

    except Exception as e:
        print(f"  ❌ Transformer inference failed: {e}")
        print("  Tip: run  pip install transformers torch  if not installed.")
        return None, None


# ==============================================================================
# 6. COMPARISON & INSIGHTS
# ==============================================================================
def print_comparison(lr_acc, nb_acc, lr_time_info, nb_time_info, transformer_time):
    """
    Print a summary comparing traditional ML vs Transformer approaches.

    Insights:
      ┌──────────────────────┬──────────────────────┬──────────────────────────┐
      │ Aspect               │ Traditional ML       │ Transformer              │
      ├──────────────────────┼──────────────────────┼──────────────────────────┤
      │ Speed (training)     │ Very fast (< 1s)     │ Slow (minutes–hours)     │
      │ Speed (inference)    │ Very fast             │ Slower per sample        │
      │ Accuracy             │ Good with features    │ Often SOTA               │
      │ Interpretability     │ High (feature weights)│ Low (black box)          │
      │ Data requirement     │ Works on small data   │ Needs more data / pretrained │
      │ Feature engineering  │ Manual (TF-IDF, etc.) │ Automatic (embeddings)   │
      └──────────────────────┴──────────────────────┴──────────────────────────┘
    """
    print(f"\n{'='*70}")
    print("  📈  COMPARISON: Traditional ML vs Transformer")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<30} {'Logistic Reg':<18} {'Naive Bayes':<18} {'Transformer':<18}")
    print(f"  {'─'*84}")

    if lr_acc is not None:
        print(f"  {'Accuracy':<30} {lr_acc:<18.4f} {nb_acc:<18.4f} {'N/A (zero-shot)':<18}")
    else:
        print(f"  {'Accuracy':<30} {'N/A':<18} {'N/A':<18} {'N/A (zero-shot)':<18}")

    t_str = f"{transformer_time:.3f}s" if transformer_time else "N/A"
    print(f"  {'Inference Speed':<30} {'< 0.01s':<18} {'< 0.01s':<18} {t_str:<18}")
    print(f"  {'Interpretability':<30} {'High':<18} {'High':<18} {'Low':<18}")
    print(f"  {'Feature Engineering':<30} {'Manual (TF-IDF)':<18} {'Manual (TF-IDF)':<18} {'Automatic':<18}")

    print("""
  💡 Key Takeaways:
     • Traditional ML (LR / NB) is fast, interpretable, and works well on
       structured / small datasets with manual feature engineering (TF-IDF).
     • Transformers capture deep contextual semantics automatically.
       They excel on nuanced tasks (sentiment, NLI) but are slower and less
       transparent.
     • For hackathons: start with TF-IDF + LR as a strong baseline, then
       upgrade to Transformers for the final submission if time permits.
    """)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    """
    End-to-end NLP pipeline.

    ── HOW TO USE ──
    1. Set the input_type variable below to "csv", "text", or "url".
    2. Update the corresponding placeholder at the top of this file.
    3. Run the script:  python nlp_pipeline.py
    """

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  CHANGE THIS to "csv", "text", or "url" as needed              ║
    # ╚══════════════════════════════════════════════════════════════════╝
    input_type = "csv"

    print("\n" + "█" * 70)
    print("  🧠  UNIVERSAL NLP PIPELINE  —  Hackathon Edition")
    print("█" * 70)

    # ── Step 1: Load Data ──
    try:
        df = load_data(input_type)
    except Exception as e:
        print(f"\n  💀 FATAL: Could not load data — {e}")
        return

    print(f"\n  DataFrame shape: {df.shape}")
    print(df.head())

    # ── Step 2: Preprocess ──
    try:
        df = preprocess_dataframe(df)
    except Exception as e:
        print(f"\n  ❌ Preprocessing failed: {e}")
        return

    # ── Step 3: TF-IDF Vectorization ──
    try:
        tfidf_matrix, vectorizer = vectorize_text(df)
    except Exception as e:
        print(f"\n  ❌ Vectorization failed: {e}")
        return

    # ── Step 4: Cosine Similarity ──
    try:
        sim_matrix = compute_similarity(tfidf_matrix, n_samples=3)
    except Exception as e:
        print(f"\n  ❌ Similarity computation failed: {e}")

    # ── Step 5: Classification ──
    lr_acc, nb_acc = None, None
    try:
        result = train_model(tfidf_matrix, df["label"])
        if result[0] is not None:
            lr, nb, X_test, y_test, class_names = result

            print(f"\n{'='*70}")
            print("  📋  MODEL EVALUATION")
            print(f"{'='*70}")
            lr_acc = evaluate_model(lr, X_test, y_test, class_names, "Logistic Regression")
            nb_acc = evaluate_model(nb, X_test, y_test, class_names, "Multinomial Naive Bayes")
    except Exception as e:
        print(f"\n  ❌ Classification failed: {e}")

    # ── Step 6: Transformer ──
    transformer_results, transformer_time = None, None
    try:
        transformer_results, transformer_time = run_transformer(df, n_samples=5)
    except Exception as e:
        print(f"\n  ❌ Transformer step failed: {e}")

    # ── Step 7: Comparison ──
    try:
        print_comparison(lr_acc, nb_acc, None, None, transformer_time)
    except Exception as e:
        print(f"\n  ❌ Comparison step failed: {e}")

    print("\n" + "█" * 70)
    print("  ✅  PIPELINE COMPLETE")
    print("█" * 70 + "\n")


# ==============================================================================
if __name__ == "__main__":
    main()
