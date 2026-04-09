"""
SYNERGIA-LM — Evaluation Metrics
Hits@1, Hits@3, Hits@10, MRR (Mean Reciprocal Rank)
Standard KG completion evaluation.
"""

import numpy as np


def hits_at_k(ranked_list: list, ground_truth: str, k: int) -> int:
    """Returns 1 if ground_truth is in top-k, else 0."""
    top_k = [item[0] if isinstance(item, (list, tuple)) else item
             for item in ranked_list[:k]]
    return 1 if ground_truth in top_k else 0


def reciprocal_rank(ranked_list: list, ground_truth: str) -> float:
    """Returns 1/rank of ground_truth, or 0 if not found."""
    for i, item in enumerate(ranked_list):
        entity = item[0] if isinstance(item, (list, tuple)) else item
        if entity == ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(all_rr: list) -> float:
    """Mean of all reciprocal ranks."""
    return np.mean(all_rr) if all_rr else 0.0


def evaluate_predictions(all_predictions: list, all_ground_truths: list,
                         ks: list = [1, 3, 10]) -> dict:
    """
    Evaluate KG completion predictions.
    all_predictions: list of ranked lists [(entity, score), ...]
    all_ground_truths: list of ground truth entity strings
    """
    n = len(all_predictions)
    results = {f"Hits@{k}": 0.0 for k in ks}
    results["MRR"] = 0.0
    all_rr = []

    for preds, gt in zip(all_predictions, all_ground_truths):
        for k in ks:
            results[f"Hits@{k}"] += hits_at_k(preds, gt, k)
        rr = reciprocal_rank(preds, gt)
        all_rr.append(rr)
        results["MRR"] += rr

    for k in ks:
        results[f"Hits@{k}"] /= n
    results["MRR"] /= n

    return results


def print_evaluation_results(results: dict, model_name: str = "SYNERGIA-LM"):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 50}")
    for metric, value in results.items():
        bar_len = int(value * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {metric:<12} {value:.4f}  {bar}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    # Demo evaluation
    predictions = [
        [("/m/a", 0.9), ("/m/b", 0.7), ("/m/c", 0.5), ("/m/d", 0.3)],
        [("/m/x", 0.8), ("/m/y", 0.6), ("/m/z", 0.4)],
        [("/m/correct", 0.95), ("/m/wrong", 0.3)],
        [("/m/wrong1", 0.9), ("/m/wrong2", 0.8), ("/m/correct2", 0.1)],
    ]
    ground_truths = ["/m/a", "/m/y", "/m/correct", "/m/correct2"]

    results = evaluate_predictions(predictions, ground_truths)
    print_evaluation_results(results)