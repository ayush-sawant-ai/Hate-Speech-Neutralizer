"""Tokenizer utilities for Keras sequence processing."""

import pickle
from pathlib import Path
from typing import Any, List, Union
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_classifier_tokenizer(tokenizer_path: Union[str, Path]) -> Any:
    """Load pickled Keras Tokenizer safely."""
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier tokenizer file not found at: {path}")
    
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def text_to_padded_sequence(
    texts: Union[str, List[str]],
    tokenizer: Any,
    max_length: int = 200,
    padding: str = "post",
    truncating: str = "post"
) -> np.ndarray:
    """Convert raw or cleaned texts to integer sequences padded to max_length."""
    if isinstance(texts, str):
        texts = [texts]
    
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Ensure sequences is not empty even if text produced no known tokens
    padded = pad_sequences(
        sequences,
        maxlen=max_length,
        padding=padding,
        truncating=truncating
    ).astype(np.int32)
    
    return padded

