"""
Metrics for evaluating machine translation quality.

This module provides functions for computing various MT evaluation metrics
including ChrF, COMET, and potentially BLEU and others in the future.
"""

import contextlib
import io
import logging
import os
import sys
import warnings

import evaluate
import torch

# Environment variable configuration for PyTorch and HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Disable tokenizers parallelism to avoid deadlocks
)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Enable fallback for unsupported MPS operations
)

# Suppress PyTorch Lightning logging by disabling its loggers
for logger_name in ["pytorch_lightning", "lightning", "lightning.pytorch"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True

warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

# Cache for loaded metrics to avoid reloading models
# Note: Caching COMET model requires ~2GB RAM for the wmt22-comet-da model
LOADED_METRICS = {
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

    Returns a float in the range [0, 1], where 0 indicates no match and 1
    indicates a perfect match.
    """
    # Load metric once and cache it
    if LOADED_METRICS["chrf"] is None:
        LOADED_METRICS["chrf"] = evaluate.load("chrf")

    chrf_metric = LOADED_METRICS["chrf"]
    result = chrf_metric.compute(
        predictions=[tr_text],
        references=[ref_text],
    )
    # Normalize score from 0-100 range to 0-1 range
    score = result["score"] / 100

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
    if LOADED_METRICS["comet"] is None:
        # Suppress stdout/stderr during model loading
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            LOADED_METRICS["comet"] = evaluate.load("comet")

    comet_metric = LOADED_METRICS["comet"]
    gpus = 1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0

    # Suppress PyTorch Lightning INFO messages by redirecting stderr at file descriptor level
    # This is necessary because PyTorch Lightning bypasses Python's logging system
    _stderr_fd = sys.stderr.fileno()
    _original_stderr_fd = os.dup(_stderr_fd)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, _stderr_fd)

    try:
        result = comet_metric.compute(
            predictions=[tr_text],
            references=[ref_text],
            sources=[src_text],
            gpus=gpus,
        )
    finally:
        # Restore original stderr
        os.dup2(_original_stderr_fd, _stderr_fd)
        os.close(_original_stderr_fd)
        os.close(_devnull_fd)

    score = result["mean_score"]

    return score
