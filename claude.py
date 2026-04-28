# ============================================================
# NLP HACKATHON PIPELINE - UNIVERSAL INPUT HANDLER
# ============================================================
# PLACEHOLDERS TO UPDATE:
#   - DATA_SOURCE      → file path, URL, or raw text
#   - DATA_SOURCE_TYPE → "csv", "txt", "pdf", "url", "raw"
#   - TEXT_COLUMN      → only for CSV: column with text
#   - LABEL_COLUMN     → only for CSV: column with labels
#   - TASK_TYPE        → "classification" or "unsupervised"
# ============================================================

import pandas as pd
import numpy as np
import re
import string
import warnings
import os
import sys
import requests
import io
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Transformers
from transformers import pipeline

# Download NLTK data
print("Downloading NLTK resources...")
for pkg in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

# ============================================================
# ██████████  CONFIGURATION — UPDATE THESE  ██████████████████
# ============================================================

# --- INPUT SOURCE ---
DATA_SOURCE      = "https://www.gutenberg.org/files/1342/1342-0.txt"  # <-- UPDATE
# Examples:
#   CSV file   : "data/reviews.csv"
#   Text file  : "book.txt"
#   PDF file   : "article.pdf"
#   URL        : "https://example.com/article"  (web page or .txt/.pdf link)
#   Raw string : Just paste text directly into RAW_TEXT below

DATA_SOURCE_TYPE = "url"
# Options: "csv" | "txt" | "pdf" | "url" | "raw"

RAW_TEXT = """
Paste your raw text here if DATA_SOURCE_TYPE is set to 'raw'.
This can be multiple paragraphs of any length.
The pipeline will automatically chunk it into sentences for analysis.
"""

# --- CSV-ONLY SETTINGS (ignored for other types) ---
TEXT_COLUMN  = "TEXT_COLUMN"   # <-- UPDATE if CSV
LABEL_COLUMN = "LABEL_COLUMN"  # <-- UPDATE if CSV

# --- PIPELINE SETTINGS ---
TASK_TYPE    = "unsupervised"  # "classification" (CSV with labels) | "unsupervised" (books/articles)
CHUNK_SIZE   = 3               # sentences per chunk for non-CSV sources
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_CLUSTERS   = 5               # for KMeans clustering (unsupervised mode)

# ============================================================
# SECTION 0: UNIVERSAL DATA LOADER
# ============================================================

def load_from_csv(path):
    """Load labeled dataset from a CSV file."""
    print(f"  → Reading CSV: {path}")
    df = pd.read_csv(path)
    print(f"  → Shape: {df.shape} | Columns: {list(df.columns)}")
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != '']
    print(f"  → {len(df)} valid rows after cleaning")
    return df, TEXT_COLUMN, LABEL_COLUMN, "classification"


def load_from_txt(path):
    """Load plain text file and chunk into sentences."""
    print(f"  → Reading TXT file: {path}")
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    print(f"  → Characters read: {len(raw):,}")
    return raw


def load_from_pdf(path):
    """Extract text from a PDF file using PyPDF2 or pdfplumber."""
    print(f"  → Reading PDF: {path}")
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            raw = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
        print(f"  → Extracted {len(raw):,} characters from {len(pdf.pages)} pages")
    except ImportError:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(path)
            raw = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
            print(f"  → Extracted {len(raw):,} characters from {len(reader.pages)} pages (PyPDF2)")
        except ImportError:
            print("  ✗ No PDF library found. Install pdfplumber: pip install pdfplumber")
            sys.exit(1)
    return raw


def load_from_url(url):
    """Download content from a URL — handles HTML pages and direct .txt/.pdf links."""
    print(f"  → Fetching URL: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (NLP-Hackathon-Bot/1.0)'}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get('Content-Type', '')
    print(f"  → Content-Type: {content_type} | Size: {len(resp.content):,} bytes")

    if 'pdf' in content_type or url.lower().endswith('.pdf'):
        # PDF from URL
        tmp = "/tmp/hackathon_download.pdf"
        with open(tmp, 'wb') as f:
            f.write(resp.content)
        return load_from_pdf(tmp)

    elif 'text/plain' in content_type or url.lower().endswith('.txt'):
        raw = resp.content.decode('utf-8', errors='ignore')
        print(f"  → Plain text: {len(raw):,} characters")
        return raw

    else:
        # HTML page — strip tags
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Remove scripts, styles, nav
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            raw = soup.get_text(separator='\n')
        except ImportError:
            # Fallback: regex tag stripping
            raw = re.sub(r'<[^>]+>', ' ', resp.text)
        raw = re.sub(r'\n{3,}', '\n\n', raw).strip()
        print(f"  → Extracted HTML text: {len(raw):,} characters")
        return raw


def chunk_text_to_df(raw_text, chunk_size=CHUNK_SIZE):
    """Convert raw text into a DataFrame of sentence-chunks for analysis."""
    sentences = sent_tokenize(raw_text)
    # Remove very short/empty sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    print(f"  → Total sentences detected: {len(sentences)}")

    # Group into chunks of N sentences
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)

    df = pd.DataFrame({'text': chunks})
    df['chunk_id'] = range(len(df))
    print(f"  → Created {len(df)} chunks (chunk_size={chunk_size})")
    return df, 'text', None, 'unsupervised'


def load_data():
    """Master loader — routes to correct handler based on DATA_SOURCE_TYPE."""
    print("\n" + "="*60)
    print("SECTION 0: UNIVERSAL DATA LOADING")
    print("="*60)
    print(f"  Source Type : {DATA_SOURCE_TYPE.upper()}")
    print(f"  Task Mode   : {TASK_TYPE.upper()}")

    src = DATA_SOURCE_TYPE.lower().strip()

    if src == "csv":
        return load_from_csv(DATA_SOURCE)

    elif src == "txt":
        raw = load_from_txt(DATA_SOURCE)
        return chunk_text_to_df(raw)

    elif src == "pdf":
        raw = load_from_pdf(DATA_SOURCE)
        return chunk_text_to_df(raw)

    elif src == "url":
        raw = load_from_url(DATA_SOURCE)
        return chunk_text_to_df(raw)

    elif src == "raw":
        print("  → Using inline RAW_TEXT")
        raw = RAW_TEXT.strip()
        print(f"  → Length: {len(raw):,} characters")
        return chunk_text_to_df(raw)

    else:
        print(f"  ✗ Unknown DATA_SOURCE_TYPE: '{DATA_SOURCE_TYPE}'")
        print("    Valid options: csv | txt | pdf | url | raw")
        sys.exit(1)


# ============================================================
# SECTION 1: PREPROCESSING
# ============================================================

def preprocess_text(text, lemmatizer, stop_words):
    """Full NLP preprocessing for a single text."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)            # remove URLs
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)            # remove brackets
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)                        # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def run_preprocessing(df, text_col):
    """Apply preprocessing pipeline to entire DataFrame."""
    print("\n" + "="*60)
    print("SECTION 1: TEXT PREPROCESSING")
    print("="*60)

    lemmatizer = WordNetLemmatizer()
    stop_words  = set(stopwords.words('english'))

    # Project Gutenberg boilerplate removal (if applicable)
    if DATA_SOURCE_TYPE == "url" and "gutenberg" in DATA_SOURCE.lower():
        df[text_col] = df[text_col].apply(
            lambda x: re.sub(r'project gutenberg.*?(\.\s)', '', x, flags=re.IGNORECASE)
        )
        print("  → Gutenberg boilerplate stripped")

    print("  Processing text chunks...")
    df['cleaned_text'] = df[text_col].apply(
        lambda x: preprocess_text(x, lemmatizer, stop_words)
    )

    # Remove empty results
    df = df[df['cleaned_text'].str.strip() != ''].reset_index(drop=True)

    print(f"\n  Samples remaining after preprocessing: {len(df)}")
    print("\n--- BEFORE vs AFTER (first 3 samples) ---")
    for i in range(min(3, len(df))):
        print(f"\n  Sample {i+1}:")
        print(f"    BEFORE : {df[text_col].iloc[i][:120].strip()}")
        print(f"    AFTER  : {df['cleaned_text'].iloc[i][:120].strip()}")

    vocab = set(' '.join(df['cleaned_text']).split())
    avg_len = np.mean([len(t.split()) for t in df['cleaned_text']])
    print(f"\n  Vocabulary size : {len(vocab):,} unique tokens")
    print(f"  Avg tokens/chunk: {avg_len:.1f}")
    return df


# ============================================================
# SECTION 2: TF-IDF + COSINE SIMILARITY
# ============================================================

def run_tfidf_similarity(df):
    """Build TF-IDF matrix and compute cosine similarity between samples."""
    print("\n" + "="*60)
    print("SECTION 2: TF-IDF + COSINE SIMILARITY")
    print("="*60)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    print(f"\n  TF-IDF Matrix Shape : {tfidf_matrix.shape}")
    print(f"  Top 10 features     : {vectorizer.get_feature_names_out()[:10].tolist()}")

    # Similarity on first 3 samples
    n = min(3, len(df))
    sim_matrix = cosine_similarity(tfidf_matrix[:n])
    sim_df = pd.DataFrame(
        sim_matrix,
        index=[f"Chunk_{i+1}" for i in range(n)],
        columns=[f"Chunk_{i+1}" for i in range(n)]
    )
    print(f"\n  Cosine Similarity Matrix (first {n} chunks):")
    print(sim_df.round(4))

    print("\n  --- Interpretation ---")
    for i in range(n):
        for j in range(i+1, n):
            score = sim_matrix[i][j]
            level = "HIGH" if score > 0.5 else "MODERATE" if score > 0.2 else "LOW"
            print(f"  Chunk_{i+1} vs Chunk_{j+1} → {score:.4f} ({level} similarity)")

    return vectorizer, tfidf_matrix


# ============================================================
# SECTION 3A: CLASSIFICATION (CSV with labels)
# ============================================================

def run_classification(df, label_col, tfidf_matrix):
    """Train and evaluate ML classifiers on labeled data."""
    print("\n" + "="*60)
    print("SECTION 3: TRADITIONAL ML CLASSIFICATION")
    print("="*60)

    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].values)
    X  = tfidf_matrix
    print(f"  Classes : {le.classes_}")
    print(f"  Samples : {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train/Test split: {X_train.shape[0]} / {X_test.shape[0]}")

    results = {}

    for name, model in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("Naive Bayes",         MultinomialNB())
    ]:
        print(f"\n  --- {name} ---")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"  Accuracy         : {acc:.4f}")
        print(f"  Confusion Matrix :\n{confusion_matrix(y_test, preds)}")
        print(f"  Classification Report:\n"
              f"{classification_report(y_test, preds, target_names=le.classes_)}")

    print("\n  --- Summary ---")
    for m, a in results.items():
        print(f"  {m}: {a:.4f}")

    return results


# ============================================================
# SECTION 3B: CLUSTERING (books/articles — unsupervised)
# ============================================================

def run_clustering(df, tfidf_matrix):
    """Cluster chunks using KMeans when no labels are available."""
    print("\n" + "="*60)
    print("SECTION 3: UNSUPERVISED CLUSTERING (No Labels Mode)")
    print("="*60)

    n_clusters = min(N_CLUSTERS, len(df))
    print(f"  Running KMeans with {n_clusters} clusters on {tfidf_matrix.shape[0]} chunks...")

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['cluster'] = km.fit_predict(tfidf_matrix)

    print("\n  --- Cluster Distribution ---")
    print(df['cluster'].value_counts().sort_index().to_string())

    # Top terms per cluster
    vectorizer_temp = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    vectorizer_temp.fit(df['cleaned_text'])
    feature_names = vectorizer_temp.get_feature_names_out()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    print("\n  --- Top Terms per Cluster ---")
    for i in range(n_clusters):
        terms = [feature_names[ind] for ind in order_centroids[i, :8]]
        print(f"  Cluster {i}: {', '.join(terms)}")

    print("\n  --- Sample Chunks per Cluster ---")
    for i in range(n_clusters):
        sample = df[df['cluster'] == i]['cleaned_text'].iloc[0]
        print(f"\n  Cluster {i} sample: {sample[:120]}...")

    return df


# ============================================================
# SECTION 4: TRANSFORMER INFERENCE
# ============================================================

def run_transformer_inference(df, text_col):
    """Sentiment/Zero-shot inference using a pretrained transformer."""
    print("\n" + "="*60)
    print("SECTION 4: TRANSFORMER MODEL (DistilBERT)")
    print("="*60)

    print("  Loading pretrained sentiment pipeline...")
    print("  Model: distilbert-base-uncased-finetuned-sst-2-english")

    transformer_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

    n = min(5, len(df))
    samples = df[text_col].iloc[:n].tolist()
    print(f"\n  Running inference on {n} samples...\n")

    transformer_results = []
    for i, text in enumerate(samples):
        result = transformer_pipe(str(text)[:512])[0]
        transformer_results.append(result)
        print(f"  Sample {i+1}:")
        print(f"    Text  : {str(text)[:100].strip()}...")
        print(f"    Label : {result['label']}")
        print(f"    Score : {result['score']:.4f}\n")

    # Overall sentiment distribution
    labels = [r['label'] for r in transformer_results]
    avg_score = np.mean([r['score'] for r in transformer_results])
    print(f"  Label distribution (sample): { {l: labels.count(l) for l in set(labels)} }")
    print(f"  Average confidence          : {avg_score:.4f}")

    return transformer_results


# ============================================================
# SECTION 5: COMPARISON & INSIGHTS
# ============================================================

def print_comparison(ml_results, transformer_results, task_mode):
    """Final comparison between traditional ML and transformer approaches."""
    print("\n" + "="*60)
    print("SECTION 5: COMPARISON & INSIGHTS")
    print("="*60)

    print("""
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ Dimension           │ Traditional ML (TF-IDF)  │ Transformer (DistilBERT) │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Accuracy            │ Good on clean labeled    │ Better on nuanced text   │
│                     │ structured CSV data      │ (contextual embeddings)  │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Speed (Training)    │ Very fast (seconds)      │ Slow (GPU needed)        │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Speed (Inference)   │ Milliseconds             │ ~100ms–1s per sample     │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Interpretability    │ High (TF-IDF weights)    │ Low (black-box attention) │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Context Awareness   │ None (bag-of-words)      │ High (bidirectional)     │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Input Flexibility   │ Structured CSV only      │ Any raw text             │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Memory Usage        │ Very low                 │ High (~250MB+ model)     │
└─────────────────────┴──────────────────────────┴──────────────────────────┘
""")

    if task_mode == "classification" and ml_results:
        print("  Traditional ML Results:")
        for model, acc in ml_results.items():
            print(f"    {model}: {acc:.4f} ({acc*100:.2f}%)")

    elif task_mode == "unsupervised":
        print("  Traditional ML: KMeans Clustering (no accuracy — unsupervised)")
        print(f"  Clusters formed: {N_CLUSTERS}")

    avg_conf = np.mean([r['score'] for r in transformer_results])
    print(f"\n  Transformer avg confidence: {avg_conf:.4f}")

    print("""
  KEY HACKATHON INSIGHTS:
  1. TF-IDF + Logistic Regression → fast baseline; great for labeled CSV data.
  2. KMeans Clustering → useful when no labels exist (books, articles, blogs).
  3. Cosine Similarity → finds near-duplicate or topically similar chunks.
  4. DistilBERT → zero-shot sentiment on ANY text, no training needed.
  5. Chunking strategy matters: too small = noise, too large = loses granularity.
  6. For hackathon: run TF-IDF first for speed, then transformer for accuracy boost.
""")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "="*60)
    print("  NLP HACKATHON UNIVERSAL PIPELINE — STARTING")
    print("="*60)

    # ── STEP 0: Load Data (any format) ────────────────────
    df, text_col, label_col, detected_task = load_data()

    # Override task type from config if explicitly set
    task_mode = TASK_TYPE if TASK_TYPE else detected_task
    print(f"\n  Final Task Mode: {task_mode.upper()}")

    # ── STEP 1: Preprocess ────────────────────────────────
    df = run_preprocessing(df, text_col)

    # ── STEP 2: TF-IDF + Similarity ───────────────────────
    vectorizer, tfidf_matrix = run_tfidf_similarity(df)

    # ── STEP 3: Classification or Clustering ──────────────
    ml_results = {}
    if task_mode == "classification" and label_col and label_col in df.columns:
        ml_results = run_classification(df, label_col, tfidf_matrix)
    else:
        df = run_clustering(df, tfidf_matrix)
        task_mode = "unsupervised"

    # ── STEP 4: Transformer Inference ─────────────────────
    transformer_results = run_transformer_inference(df, text_col)

    # ── STEP 5: Comparison & Insights ─────────────────────
    print_comparison(ml_results, transformer_results, task_mode)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE ✓")
    print("="*60)


if __name__ == "__main__":
    main()