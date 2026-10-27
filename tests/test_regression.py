"""Regression test executing the 11-category edge case evaluation."""

import pytest
from src.classification.detector import ToxicityDetector
from src.evaluation.regression_test import run_regression_suite


def test_regression_suite_execution():
    detector = ToxicityDetector(model_key="bigru", threshold=0.40)
    detector.load()

    suite_res = run_regression_suite(detector, threshold=0.40)
    assert suite_res["total_cases"] == 22
    assert suite_res["total_passed"] >= 12
    assert len(suite_res["category_summary"]) == 11
    # Verify critical categories succeed
    assert suite_res["category_summary"]["direct_insults_and_harassment"]["passed"] == 2
    assert suite_res["category_summary"]["strong_legitimate_criticism"]["passed"] == 2
    assert suite_res["category_summary"]["negative_sentiment_without_toxicity"]["passed"] == 2
