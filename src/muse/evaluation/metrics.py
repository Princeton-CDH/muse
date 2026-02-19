"""
Metrics for evaluating machine translation quality.

This module provides functions for computing various MT evaluation metrics
including ChrF, and potentially COMET, BLEU, and others in the future.
"""

import evaluate


def compute_chrf(
    tr_text: str,
    ref_text: str,
) -> float:
    """
    Compute ChrF score for a translation against a reference translation using
    HuggingFace's evaluate library.

    Returns a float in the range [0, 100], where 0 indicates no match and 100
    indicates a perfect match.
    """
    chrf_metric = evaluate.load("chrf")
    result = chrf_metric.compute(
        predictions=[tr_text],
        references=[ref_text],
    )
    score = result["score"]

    return score
