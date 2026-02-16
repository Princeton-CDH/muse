"""
Compute ChrF (Character n-gram F-score) metric for machine translation evaluation.

ChrF is an n-gram based MT metric that measures translation quality by comparing
character n-grams between the translation and reference text. It is particularly
effective for morphologically rich languages and does not require tokenization.
"""

import evaluate


def compute_chrf(
    tr_text: str,
    ref_text: str,
) -> float:
    """
    This function uses HuggingFace's evaluate library to compute ChrF score
    for a translation against a reference translation.

    Args:
        tr_text: The translation text to be scored
        ref_text: The reference translation

    Returns:
        ChrF score as a float in the range [0, 100], where 0 indicates no match
        and 100 indicates a perfect match.
    """
    if not tr_text or not tr_text.strip():
        raise ValueError("Translation text cannot be empty")
    if not ref_text or not ref_text.strip():
        raise ValueError("Reference text cannot be empty")

    chrf_metric = evaluate.load("chrf")
    result = chrf_metric.compute(
        predictions=[tr_text],
        references=[ref_text],
    )
    score = result["score"]

    return score
