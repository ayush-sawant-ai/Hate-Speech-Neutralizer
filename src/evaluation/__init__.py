"""Evaluation package for classifier benchmarking, rewriter quality, and regression test suites."""

from src.evaluation.regression_data import REGRESSION_TEST_CASES
from src.evaluation.regression_test import run_regression_suite, print_regression_report
from src.evaluation.evaluate_all import run_full_classifier_benchmark
from src.evaluation.evaluate_rewriter import benchmark_rewriter_on_pairs

__all__ = [
    "REGRESSION_TEST_CASES",
    "run_regression_suite",
    "print_regression_report",
    "run_full_classifier_benchmark",
    "benchmark_rewriter_on_pairs",
]

