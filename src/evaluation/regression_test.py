"""Regression testing script evaluating the 11 edge-case categories."""

from typing import Any, Dict, List
from src.classification.detector import ToxicityDetector
from src.evaluation.regression_data import REGRESSION_TEST_CASES


def run_regression_suite(
    detector: ToxicityDetector,
    threshold: float = 0.40
) -> Dict[str, Any]:
    """Execute the regression suite and return detailed category-by-category breakdown."""
    results = []
    category_summary = {}

    for item in REGRESSION_TEST_CASES:
        cat = item["category"]
        if cat not in category_summary:
            category_summary[cat] = {"total": 0, "passed": 0, "failed": 0}

        pred = detector.predict(item["text"], threshold=threshold)
        passed = (pred["label"] == item["expected_label"])

        category_summary[cat]["total"] += 1
        if passed:
            category_summary[cat]["passed"] += 1
        else:
            category_summary[cat]["failed"] += 1

        results.append({
            "category": cat,
            "text": item["text"],
            "expected_label": item["expected_label"],
            "predicted_label": pred["label"],
            "probability": pred["probability"],
            "passed": passed,
            "rationale": item["rationale"],
        })

    total_cases = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    overall_accuracy = total_passed / total_cases if total_cases > 0 else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "total_cases": total_cases,
        "total_passed": total_passed,
        "threshold": threshold,
        "category_summary": category_summary,
        "detailed_results": results,
    }


def print_regression_report(suite_result: Dict[str, Any]) -> None:
    """Print a clean terminal report of regression results."""
    print("=" * 70)
    print("REGRESSION SUITE EVALUATION REPORT (11 Context & Edge Categories)")
    print("=" * 70)
    print(f"Overall Accuracy: {suite_result['overall_accuracy'] * 100:.1f}% ({suite_result['total_passed']}/{suite_result['total_cases']})")
    print(f"Threshold: {suite_result['threshold']:.2f}")
    print("\nCategory Summary:")
    print(f"{'Category':<38} | {'Passed':<7} | {'Total':<6} | {'Pass Rate':<10}")
    print("-" * 70)
    for cat, stats in suite_result["category_summary"].items():
        rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        print(f"{cat:<38} | {stats['passed']:<7} | {stats['total']:<6} | {rate:>6.1f}%")
    print("-" * 70)
    print("\nFailures / Discrepancies:")
    failures = [r for r in suite_result["detailed_results"] if not r["passed"]]
    if not failures:
        print("None! All regression cases matched expected labels.")
    else:
        for f in failures:
            print(f"- [{f['category']}] Expected: {f['expected_label']} | Pred: {f['predicted_label']} (prob: {f['probability']:.4f})")
            print(f"  Text: \"{f['text']}\"")
            print(f"  Rationale: {f['rationale']}\n")

