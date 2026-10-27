"""Evaluation script comparing all available classifier checkpoints on the test set."""

import json
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from src.config import CLASSIFIER_CHECKPOINTS, TEST_DATA_PKL
from src.classification.evaluator import evaluate_classifier_predictions, find_optimal_threshold


def run_full_classifier_benchmark(
    test_data_path: Path = TEST_DATA_PKL,
    output_json_path: Path = None,
    threshold: float = 0.40
) -> dict:
    """Benchmark all available classifier checkpoints on the held-out test set."""
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found at: {test_data_path}")

    with open(test_data_path, "rb") as f:
        X_test, y_test = pickle.load(f)

    y_test = np.asarray(y_test, dtype=int)
    results = {}

    for key, info in CLASSIFIER_CHECKPOINTS.items():
        model_path = info["path"]
        if not model_path.exists():
            print(f"Skipping {key}: file not found at {model_path}")
            continue

        print(f"Evaluating {info['name']} ({key})...")
        model = tf.keras.models.load_model(str(model_path), compile=False)
        y_prob = model.predict(X_test, batch_size=256, verbose=0).ravel()

        metrics_at_th = evaluate_classifier_predictions(y_test, y_prob, threshold=threshold)
        metrics_at_50 = evaluate_classifier_predictions(y_test, y_prob, threshold=0.50)
        threshold_sweep = find_optimal_threshold(y_test, y_prob)

        results[key] = {
            "name": info["name"],
            "description": info["description"],
            "metrics_at_custom_threshold": metrics_at_th,
            "metrics_at_0_50": metrics_at_50,
            "threshold_sweep": threshold_sweep,
        }

    if output_json_path:
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results saved to: {output_json_path}")

    return results


if __name__ == "__main__":
    from src.config import BASE_DIR
    out_file = BASE_DIR / "models" / "benchmark_results.json"
    run_full_classifier_benchmark(output_json_path=out_file)

