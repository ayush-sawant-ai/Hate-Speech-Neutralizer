"""Hate speech and toxicity detector module wrapping Keras neural models."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import tensorflow as tf

from src.config import (
    CLASSIFIER_CHECKPOINTS,
    CLASSIFIER_MAX_LEN,
    DEFAULT_CLASSIFIER_KEY,
    DEFAULT_THRESHOLD,
    TOKENIZER_PATH,
)
from src.preprocessing.cleaner import clean_for_classifier, validate_input_text
from src.preprocessing.tokenizer_utils import (
    load_classifier_tokenizer,
    text_to_padded_sequence,
)


class ToxicityDetector:
    """Wrapper for Keras recurrent toxicity classifiers (BiGRU, BiLSTM, GRU, LSTM)."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        model_key: str = DEFAULT_CLASSIFIER_KEY,
    ):
        self.model_key = model_key
        if model_path is None:
            checkpoint = CLASSIFIER_CHECKPOINTS.get(model_key)
            if checkpoint:
                self.model_path = checkpoint["path"]
            else:
                self.model_path = CLASSIFIER_CHECKPOINTS["bigru"]["path"]
        else:
            self.model_path = Path(model_path)

        self.tokenizer_path = Path(tokenizer_path or TOKENIZER_PATH)
        self.threshold = threshold
        self.model = None
        self.tokenizer = None
        self.max_len = CLASSIFIER_MAX_LEN

    def load(self) -> None:
        """Load model and tokenizer weights into memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}. "
                "Ensure Git LFS pull has been executed."
            )
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found at: {self.tokenizer_path}"
            )

        self.tokenizer = load_classifier_tokenizer(self.tokenizer_path)
        self.model = tf.keras.models.load_model(str(self.model_path), compile=False)

        # Dynamically infer max length from model input layer if available
        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if len(input_shape) > 1 and input_shape[1] is not None:
            self.max_len = int(input_shape[1])

    def predict(
        self,
        text: str,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Analyze text and predict toxicity probability and classification label.
        
        Args:
            text: Raw input string.
            threshold: Optional threshold override. If None, uses self.threshold.
            
        Returns:
            Dict containing label, probability, threshold, cleaned text, model metadata.
        """
        if self.model is None or self.tokenizer is None:
            self.load()

        effective_threshold = threshold if threshold is not None else self.threshold

        is_valid, err_msg = validate_input_text(text)
        if not is_valid:
            # Handle empty or invalid input gracefully
            return {
                "label": "SAFE",
                "probability": 0.0,
                "is_toxic": False,
                "threshold": effective_threshold,
                "model_key": self.model_key,
                "cleaned_text": "",
                "note": f"Input validation warning: {err_msg}",
                "is_valid": False,
            }

        cleaned = clean_for_classifier(text)
        padded = text_to_padded_sequence(
            cleaned,
            self.tokenizer,
            max_length=self.max_len,
            padding="post",
            truncating="post"
        )

        raw_pred = self.model.predict(padded, verbose=0)
        prob = float(raw_pred[0][0])
        is_toxic = prob >= effective_threshold
        label = "TOXIC" if is_toxic else "SAFE"
        model_name = CLASSIFIER_CHECKPOINTS.get(self.model_key, {}).get("name", self.model_key.upper())

        return {
            "label": label,
            "probability": prob,
            "is_toxic": is_toxic,
            "threshold": effective_threshold,
            "model_key": self.model_key,
            "model_name": model_name,
            "cleaned_text": cleaned,
            "is_valid": True,
            "note": "Binary toxicity target derived from Jigsaw dataset."
        }

