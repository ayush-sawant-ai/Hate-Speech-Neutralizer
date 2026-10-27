"""Text preprocessing and cleaning utilities for classification and rewriting."""

import re
import html
import unicodedata
from typing import Tuple


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to NFKC form to unify accented and fullwidth chars."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return unicodedata.normalize("NFKC", text)


def clean_for_classifier(text: str) -> str:
    """Preprocess text for the RNN / GRU / LSTM classifiers.
    
    Steps:
    1. HTML unescape and unicode normalization.
    2. Lowercase conversion.
    3. Removal of URLs and @-mentions.
    4. Retention of letters and spaces, replacing punctuation with spaces.
    5. Stripping redundant whitespace.
    
    Guarantees returning a string (empty string for non-text or empty inputs).
    """
    if text is None:
        return ""
    
    text = str(text)
    text = html.unescape(text)
    text = normalize_unicode(text)
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Retain alphabetic characters and spaces
    # Note: replacing punctuation with a space prevents concatenation of adjacent words
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse multiple spaces and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_for_rewriter(text: str) -> str:
    """Preprocess text for T5 encoder-decoder neutralization.
    
    T5 is a pretrained transformer with subword byte-pair tokenization that relies
    on capitalization and punctuation to generate fluent, grammatical outputs.
    Therefore, we do NOT lowercase or strip punctuation, but normalize unicode,
    strip harmful control characters, and trim whitespace.
    """
    if text is None:
        return ""
    
    text = str(text)
    text = html.unescape(text)
    text = normalize_unicode(text)

    # Remove raw URLs to prevent hallucinated URLs in output
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Normalize whitespace without destroying sentence capitalization or punctuation
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()

    return text


def validate_input_text(text: str, max_chars: int = 1000) -> Tuple[bool, str]:
    """Validate user input before inference.
    
    Returns (is_valid, error_message).
    """
    if text is None or not text.strip():
        return False, "Input text cannot be empty."
    
    if len(text.strip()) > max_chars:
        return False, f"Input text exceeds maximum allowed length of {max_chars} characters."
    
    cleaned = clean_for_classifier(text)
    if not cleaned:
        return False, "Input contains no recognizable alphabetic characters or words."
    
    return True, ""

