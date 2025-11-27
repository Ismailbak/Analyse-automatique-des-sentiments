"""Data loading utilities for Sentiment140 dataset."""
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_raw_sentiment140(path: str) -> pd.DataFrame:
    """Load the raw Sentiment140 CSV into a DataFrame.

    Expects a CSV with at least columns for text and label. Adjust as needed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(p, encoding='latin-1', header=None)
    # Typical Sentiment140 layout: [target, id, date, flag, user, text]
    if df.shape[1] >= 6:
        df = df.iloc[:, [0, 5]]
        df.columns = ['target', 'text']
    return df


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split as tts

    return tts(df, test_size=test_size, random_state=random_state, stratify=df['target'])
