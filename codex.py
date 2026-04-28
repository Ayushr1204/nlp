import re
import string
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from transformers import pipeline


warnings.filterwarnings("ignore")


# ============================================================
# MANDATORY PLACEHOLDERS - CHANGE ONLY THESE FOR YOUR DATASET
# ============================================================

INPUT_TYPE = "csv"  # Options: "csv", "text", "url"

CSV_FILE_PATH = "your_dataset.csv"
TEXT_COLUMN = "TEXT_COLUMN"
LABEL_COLUMN = "LABEL_COLUMN"

TEXT_FILE_PATH = "your_text_file.txt"

URL = "https://example.com/article"


# ============================================================
# NLTK SETUP
# ============================================================

def setup_nltk():
    """Download required NLTK resources if missing."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[INFO] Downloading NLTK resource: {package}")
            nltk.download(package, quiet=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_data(input_type: str) -> pd.DataFrame:
    """
    Loads data from CSV, plain text file, or URL.

    Returns a common dataframe format:
        - text
        - label

    If labels are unavailable, dummy labels are created.
    Metadata:
        df.attrs["has_real_labels"] = True / False
    """

    print("\n==============================")
    print("STAGE 1: DATA LOADING")
    print("==============================")

    input_type = input_type.lower().strip()
    has_real_labels = False

    try:
        if input_type == "csv":
            print(f"[INFO] Loading CSV file: {CSV_FILE_PATH}")

            df_raw = pd.read_csv(CSV_FILE_PATH)

            if TEXT_COLUMN not in df_raw.columns:
                raise ValueError(
                    f"Text column '{TEXT_COLUMN}' not found. "
                    f"Available columns: {list(df_raw.columns)}"
                )

            df = pd.DataFrame()
            df["text"] = df_raw[TEXT_COLUMN].astype(str)

            if LABEL_COLUMN in df_raw.columns:
                df["label"] = df_raw[LABEL_COLUMN]
                has_real_labels = True
                print(f"[INFO] Label column found: {LABEL_COLUMN}")
            else:
                df["label"] = "unlabeled"
                print("[WARN] Label column not found. Created dummy labels.")

        elif input_type == "text":
            print(f"[INFO] Loading text file: {TEXT_FILE_PATH}")

            with open(TEXT_FILE_PATH, "r", encoding="utf-8") as file:
                content = file.read()

            samples = split_text_into_samples(content)

            df = pd.DataFrame({
                "text": samples,
                "label": ["unlabeled"] * len(samples)
            })

            print("[INFO] Plain text file loaded. Dummy labels created.")

        elif input_type == "url":
            print(f"[INFO] Scraping URL: {URL}")

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
                )
            }

            response = requests.get(URL, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator=" ")
            text = clean_whitespace(text)

            samples = split_text_into_samples(text)

            df = pd.DataFrame({
                "text": samples,
                "label": ["unlabeled"] * len(samples)
            })

            print("[INFO] URL content scraped. Dummy labels created.")

        else:
            raise ValueError("Invalid input_type. Use 'csv', 'text', or 'url'.")

    except Exception as error:
        print(f"[ERROR] Data loading failed: {error}")
        print("[INFO] Using fallback demo dataset so script remains runnable.")

        df = pd.DataFrame({
            "text": [
                "I love this product. It works really well.",
                "This was a terrible experience and I am disappointed.",
                "The service was okay, nothing special.",
                "Amazing quality and fast delivery.",
                "Poor design and very bad support."
            ],
            "label": [
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative"
            ]
        })

        has_real_labels = True

    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("unlabeled").astype(str)

    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    if df.empty:
        print("[WARN] No valid text found. Creating fallback samples.")
        df = pd.DataFrame({
            "text": [
                "Natural language processing is useful.",
                "Machine learning models can classify text.",
                "Transformers are powerful for language tasks."
            ],
            "label": ["unlabeled", "unlabeled", "unlabeled"]
        })
        has_real_labels = False

    df.attrs["has_real_labels"] = has_real_labels

    print(f"[INFO] Loaded samples: {len(df)}")
    print(f"[INFO] Real labels available: {has_real_labels}")
    print("\n[DATA PREVIEW]")
    print(df.head())

    return df


def split_text_into_samples(content: str) -> list:
    """Splits long text into sentence-like or paragraph-like samples."""

    content = clean_whitespace(content)

    paragraphs = [
        para.strip()
        for para in re.split(r"\n\s*\n", content)
        if len(para.strip()) > 30
    ]

    if len(paragraphs) >= 3:
        return paragraphs

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", content)
        if len(sentence.strip()) > 15
    ]

    return sentences if sentences else [content]


def clean_whitespace(text: str) -> str:
    """Normalizes whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_text(text: str) -> str:
    """
    Preprocessing steps:
        - lowercasing
        - punctuation removal
        - tokenization
        - stopword removal
        - lemmatization
    """

    try:
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", "", text)

        tokens = nltk.word_tokenize(text)

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        processed_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words and token.strip()
        ]

        return " ".join(processed_tokens)

    except Exception as error:
        print(f"[WARN] Preprocessing failed for one sample: {error}")
        return ""


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Applies preprocessing and prints before/after samples."""

    print("\n==============================")
    print("STAGE 2: PREPROCESSING")
    print("==============================")

    print("\n[BEFORE PREPROCESSING]")
    print(df["text"].head(3).to_string(index=False))

    df["processed_text"] = df["text"].apply(preprocess_text)

    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)

    print("\n[AFTER PREPROCESSING]")
    print(df["processed_text"].head(3).to_string(index=False))

    print(f"\n[INFO] Samples after preprocessing: {len(df)}")

    return df


# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

def vectorize_text(df: pd.DataFrame) -> Tuple[Optional[TfidfVectorizer], Optional[np.ndarray]]:
    """Converts processed text into TF-IDF vectors."""

    print("\n==============================")
    print("STAGE 3: TF-IDF VECTORIZATION")
    print("==============================")

    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

        print(f"[INFO] TF-IDF matrix shape: {tfidf_matrix.shape}")
        print("[INFO] Sample TF-IDF features:")
        print(vectorizer.get_feature_names_out()[:20])

        return vectorizer, tfidf_matrix

    except Exception as error:
        print(f"[ERROR] TF-IDF vectorization failed: {error}")
        return None, None


# ============================================================
# SIMILARITY
# ============================================================

def compute_similarity(tfidf_matrix):
    """Computes cosine similarity between the first 2-3 samples."""

    print("\n==============================")
    print("STAGE 4: COSINE SIMILARITY")
    print("==============================")

    try:
        if tfidf_matrix is None or tfidf_matrix.shape[0] < 2:
            print("[WARN] Need at least 2 samples for similarity.")
            return

        sample_count = min(3, tfidf_matrix.shape[0])
        similarity_matrix = cosine_similarity(tfidf_matrix[:sample_count])

        print(f"[INFO] Cosine similarity among first {sample_count} samples:")
        print(np.round(similarity_matrix, 3))

        # Interpretation:
        # Values closer to 1 mean texts are more similar.
        # Values closer to 0 mean texts are less similar.
        print(
            "[INTERPRETATION] Higher values indicate more similar text samples; "
            "lower values indicate more different samples."
        )

    except Exception as error:
        print(f"[ERROR] Similarity computation failed: {error}")


# ============================================================
# CLASSIFICATION MODEL
# ============================================================

def train_model(df: pd.DataFrame, tfidf_matrix):
    """
    Trains Logistic Regression if real labels are available.
    If labels are dummy/unavailable, training is skipped.
    """

    print("\n==============================")
    print("STAGE 5: CLASSIFICATION MODEL")
    print("==============================")

    try:
        has_real_labels = df.attrs.get("has_real_labels", False)

        if not has_real_labels:
            print("[INFO] No real labels available. Skipping supervised model training.")
            print("[INFO] Dummy labels exist only to preserve common dataframe format.")
            return None, None, None, None

        if tfidf_matrix is None:
            print("[WARN] TF-IDF matrix unavailable. Cannot train model.")
            return None, None, None, None

        unique_labels = df["label"].nunique()

        if unique_labels < 2:
            print("[WARN] Need at least 2 unique labels for classification.")
            return None, None, None, None

        label_counts = df["label"].value_counts()
        print("\n[LABEL DISTRIBUTION]")
        print(label_counts)

        stratify_labels = df["label"] if label_counts.min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_matrix,
            df["label"],
            test_size=0.25,
            random_state=42,
            stratify=stratify_labels
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        print("[INFO] Logistic Regression model trained successfully.")

        return model, X_test, y_test, df["label"].unique()

    except Exception as error:
        print(f"[ERROR] Model training failed: {error}")
        return None, None, None, None


def evaluate_model(model, X_test, y_test):
    """Evaluates classification model."""

    print("\n==============================")
    print("STAGE 6: MODEL EVALUATION")
    print("==============================")

    try:
        if model is None or X_test is None or y_test is None:
            print("[INFO] Evaluation skipped because model was not trained.")
            return

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)

        print(f"[ACCURACY] {accuracy:.4f}")

        print("\n[CONFUSION MATRIX]")
        print(cm)

        print("\n[CLASSIFICATION REPORT]")
        print(report)

    except Exception as error:
        print(f"[ERROR] Model evaluation failed: {error}")


# ============================================================
# TRANSFORMER MODEL
# ============================================================

def run_transformer(df: pd.DataFrame):
    """
    Runs pretrained Transformer sentiment-analysis pipeline.

    Traditional ML vs Transformer notes:
        - Traditional ML with TF-IDF is faster, lighter, and more interpretable.
        - Transformers usually understand context better but are slower and heavier.
        - TF-IDF models need task-specific labels for strong accuracy.
        - Transformers can provide useful zero/few-shot style insights even without labels.
    """

    print("\n==============================")
    print("STAGE 7: TRANSFORMER MODEL")
    print("==============================")

    try:
        print("[INFO] Loading pretrained sentiment-analysis pipeline...")

        sentiment_pipeline = pipeline("sentiment-analysis")

        samples = df["text"].head(5).tolist()
        samples = [sample[:512] for sample in samples]

        predictions = sentiment_pipeline(samples)

        print("\n[TRANSFORMER PREDICTIONS]")
        for idx, (sample, prediction) in enumerate(zip(samples, predictions), start=1):
            print(f"\nSample {idx}:")
            print(f"Text: {sample[:200]}...")
            print(f"Prediction: {prediction}")

    except Exception as error:
        print(f"[ERROR] Transformer pipeline failed: {error}")
        print(
            "[INFO] If running offline, download/cache the transformer model first "
            "or enable internet access."
        )


# ============================================================
# COMPARISON & INSIGHTS
# ============================================================

def print_comparison_insights():
    """Prints comparison between traditional ML and Transformer models."""

    print("\n==============================")
    print("STAGE 8: COMPARISON & INSIGHTS")
    print("==============================")

    print("""
Traditional ML Model:
- Uses TF-IDF features and Logistic Regression.
- Usually faster and easier to interpret.
- Works best when labeled training data is available.
- Accuracy depends heavily on dataset quality and feature representation.

Transformer Model:
- Uses pretrained contextual language understanding.
- Can work immediately on raw text for tasks like sentiment analysis.
- Often captures meaning, tone, and context better than TF-IDF.
- Usually slower and more computationally expensive.

Hackathon Strategy:
- Use TF-IDF + Logistic Regression when labels are provided.
- Use Transformer pipeline when labels are missing or fast semantic insight is needed.
- Report both results when possible for a stronger comparison.
""")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n==========================================")
    print("UNIVERSAL NLP HACKATHON PIPELINE STARTED")
    print("==========================================")

    setup_nltk()

    df = load_data(INPUT_TYPE)

    if df.empty:
        print("[ERROR] No data available. Exiting pipeline.")
        return

    df = apply_preprocessing(df)

    if df.empty:
        print("[ERROR] No valid text after preprocessing. Exiting pipeline.")
        return

    vectorizer, tfidf_matrix = vectorize_text(df)

    compute_similarity(tfidf_matrix)

    model, X_test, y_test, labels = train_model(df, tfidf_matrix)

    evaluate_model(model, X_test, y_test)

    run_transformer(df)

    print_comparison_insights()

    print("\n==========================================")
    print("UNIVERSAL NLP HACKATHON PIPELINE FINISHED")
    print("==========================================")


if __name__ == "__main__":
    main()
