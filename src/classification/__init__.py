"""Classification module for toxicity detection."""

from src.classification.detector import ToxicityDetector
from src.classification.evaluator import (
    evaluate_classifier_predictions,
    find_optimal_threshold,
)

__all__ = [
    "ToxicityDetector",
    "evaluate_classifier_predictions",
    "find_optimal_threshold",
]

