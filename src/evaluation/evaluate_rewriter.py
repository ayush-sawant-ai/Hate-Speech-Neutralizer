"""Evaluation script for the fine-tuned T5 neutralizer."""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from tqdm import tqdm

from src.classification.detector import ToxicityDetector
from src.config import REWRITING_PAIRS_CSV
from src.rewriting.neutralizer import TextNeutralizer


def benchmark_rewriter_on_pairs(
    num_samples: int = 50,
    csv_path: Path = REWRITING_PAIRS_CSV,
    threshold: float = 0.40
) -> Dict[str, Any]:
    """Evaluate T5 neutralizer on held-out parallel toxic-neutral pairs."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Rewriting pairs not found at: {csv_path}")

    df = pd.read_csv(csv_path).dropna().drop_duplicates()
    # Sample from the end (held-out portion if 80/20 split was used)
    test_slice = df.tail(num_samples)

    detector = ToxicityDetector(threshold=threshold)
    detector.load()

    neutralizer = TextNeutralizer()
    neutralizer.load()

    eval_records = []
    total_orig_tox = 0.0
    total_post_tox = 0.0
    successful_reductions = 0
    acceptable_count = 0

    print(f"Benchmarking T5 neutralizer on {len(test_slice)} sentence pairs...")
    for _, row in test_slice.iterrows():
        toxic_text = str(row["toxic"])
        ground_truth_neutral = str(row["neutral"])

        res = neutralizer.rewrite(
            toxic_text,
            detector_instance=detector,
            threshold=threshold
        )

        orig_tox = res["original_toxicity"] or 0.0
        post_tox = res["post_toxicity"] or 0.0
        total_orig_tox += orig_tox
        total_post_tox += post_tox

        if orig_tox > post_tox:
            successful_reductions += 1
        if res["is_acceptable"]:
            acceptable_count += 1

        eval_records.append({
            "toxic_input": toxic_text,
            "ground_truth_neutral": ground_truth_neutral,
            "model_rewrite": res["rewritten_text"],
            "status": res["status"],
            "is_acceptable": res["is_acceptable"],
            "orig_tox_prob": round(orig_tox, 4),
            "post_tox_prob": round(post_tox, 4),
            "tox_reduction": round(orig_tox - post_tox, 4)
        })

    n = len(eval_records)
    avg_orig = total_orig_tox / n if n > 0 else 0.0
    avg_post = total_post_tox / n if n > 0 else 0.0
    avg_reduction = avg_orig - avg_post

    summary = {
        "num_evaluated": n,
        "average_input_toxicity_prob": round(avg_orig, 4),
        "average_rewritten_toxicity_prob": round(avg_post, 4),
        "average_toxicity_reduction": round(avg_reduction, 4),
        "fraction_reduced": round(successful_reductions / n, 4) if n > 0 else 0.0,
        "fraction_acceptable": round(acceptable_count / n, 4) if n > 0 else 0.0,
        "sample_records": eval_records[:10]
    }

    return summary


if __name__ == "__main__":
    benchmark = benchmark_rewriter_on_pairs(num_samples=25)
    print("=" * 60)
    print("REWRITER BENCHMARK SUMMARY:")
    print(f"Evaluated samples: {benchmark['num_evaluated']}")
    print(f"Avg input toxicity: {benchmark['average_input_toxicity_prob']:.4f}")
    print(f"Avg post toxicity:  {benchmark['average_rewritten_toxicity_prob']:.4f}")
    print(f"Avg toxicity reduction: {benchmark['average_toxicity_reduction']:.4f}")
    print(f"Acceptable rewrites rate: {benchmark['fraction_acceptable'] * 100:.1f}%")
    print("=" * 60)

