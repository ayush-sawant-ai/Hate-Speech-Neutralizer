"""Unit tests for toxicity classifier models and tokenizer."""

import pytest
from src.classification.detector import ToxicityDetector
from src.config import CLASSIFIER_CHECKPOINTS


def test_bigru_detector_loading_and_prediction():
    detector = ToxicityDetector(model_key="bigru", threshold=0.40)
    detector.load()
    assert detector.model is not None
    assert detector.tokenizer is not None

    # Test obvious toxic text
    res = detector.predict("I hate you so much, you are disgusting and ugly.")
    assert "probability" in res
    assert "label" in res
    assert res["probability"] > 0.40
    assert res["label"] == "TOXIC"

    # Test obvious safe text
    safe_res = detector.predict("The weather in the city today is wonderful and sunny.")
    assert safe_res["probability"] < 0.40
    assert safe_res["label"] == "SAFE"


def test_detector_threshold_override():
    detector = ToxicityDetector(model_key="bigru", threshold=0.40)
    detector.load()

    # Probability around ~0.35 - 0.45 should flip with threshold changes
    res_low_th = detector.predict("Your work is completely terrible.", threshold=0.10)
    res_high_th = detector.predict("Your work is completely terrible.", threshold=0.90)

    assert res_low_th["threshold"] == 0.10
    assert res_high_th["threshold"] == 0.90
    assert res_low_th["probability"] == res_high_th["probability"]


def test_all_checkpoints_exist_and_load():
    for key, info in CLASSIFIER_CHECKPOINTS.items():
        assert info["path"].exists(), f"Model file for {key} missing at {info['path']}"
        detector = ToxicityDetector(model_key=key)
        detector.load()
        assert detector.model is not None


def test_identical_text_scoring_reproducibility():
    """Verify that identical text scored under identical settings produces matching scores within numerical tolerance."""
    for key in CLASSIFIER_CHECKPOINTS.keys():
        detector = ToxicityDetector(model_key=key, threshold=0.40)
        detector.load()
        sample_texts = [
            "You are a complete idiot and nobody cares what you think.",
            "Please focus on improving the quality of your work.",
            "I disagree with your proposal because latency is too high.",
        ]
        for t in sample_texts:
            score1 = detector.predict(t)["probability"]
            score2 = detector.predict(t)["probability"]
            assert abs(score1 - score2) < 1e-6, f"Score mismatch for model {key}: {score1} vs {score2}"


