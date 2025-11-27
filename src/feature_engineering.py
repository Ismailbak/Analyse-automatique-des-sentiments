"""Feature engineering: TF-IDF, embeddings, and helpers."""
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from typing import Any


def build_tfidf(corpus, max_features: int = 20000) -> TfidfVectorizer:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    vec.fit(corpus)
    return vec


def save_vectorizer(vec: Any, path: str) -> None:
    joblib.dump(vec, path)


def load_vectorizer(path: str) -> Any:
    return joblib.load(path)
