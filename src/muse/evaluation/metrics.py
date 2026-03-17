"""
Metrics for evaluating machine translation quality.

This module provides functions for computing various MT evaluation metrics
including ChrF, COMET, and potentially BLEU and others in the future.
"""

import contextlib
import io
import logging
import os
import warnings

import evaluate
import torch

# Suppress verbose HuggingFace and PyTorch Lightning logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PL_DISABLE_FORK"] = "1"

# Suppress all PyTorch Lightning loggers
for logger_name in [
    "pytorch_lightning",
    "lightning.pytorch",
    "lightning.pytorch.utilities.rank_zero",
    "lightning.pytorch.accelerators.mps",
    "lightning.pytorch.core",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

warnings.filterwarnings("ignore")

# Cache for loaded metrics to avoid reloading models
_LOADED_METRICS = {
    "chrf": None,
    "comet": None,
}


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
    # Load metric once and cache it
    if _LOADED_METRICS["chrf"] is None:
        _LOADED_METRICS["chrf"] = evaluate.load("chrf")

    chrf_metric = _LOADED_METRICS["chrf"]
    result = chrf_metric.compute(
        predictions=[tr_text],
        references=[ref_text],
    )
    score = result["score"]

    return score


def compute_comet(
    tr_text: str,
    src_text: str,
    ref_text: str,
) -> float:
    """
    Compute COMET score for a translation using HuggingFace's evaluate library.

    COMET (Crosslingual Optimized Metric for Evaluation of Translation) is an
    LLM-based metric that evaluates machine translations by considering the
    source text, reference translation, and machine translation.

    Returns a float in the range [0, 1], where 0 indicates a poor translation
    and 1 indicates a perfect translation.
    """
    # Load metric once and cache it
    if _LOADED_METRICS["comet"] is None:
        # Suppress stdout/stderr during model loading
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            _LOADED_METRICS["comet"] = evaluate.load("comet")

    comet_metric = _LOADED_METRICS["comet"]
    gpus = 1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0

    # Suppress stdout/stderr during computation
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        result = comet_metric.compute(
            predictions=[tr_text],
            references=[ref_text],
            sources=[src_text],
            gpus=gpus,
        )
    score = result["mean_score"]

    return score
