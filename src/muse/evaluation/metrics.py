"""
Metrics for evaluating machine translation quality.

This module provides functions for computing various MT evaluation metrics
including ChrF, COMET, and potentially BLEU and others in the future.
"""

import contextlib
import io
import logging
import os
from typing import Any

import evaluate
import torch
from comet import download_model, load_from_checkpoint

# Environment variable configuration for PyTorch and HuggingFace libraries
os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Disable tokenizers parallelism to avoid deadlocks
)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Enable fallback for unsupported MPS operations
)

# Suppress PyTorch Lightning INFO messages
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.migration").setLevel(logging.WARNING)

# Cache for loaded metrics to avoid reloading models
# Note: Caching COMET model requires ~2GB RAM for the wmt22-comet-da model
LOADED_METRICS: dict[str, Any] = {
    "chrf": None,
    "comet": None,
    "cometkiwi": None,
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

    result = comet_metric.compute(
        predictions=[tr_text],
        references=[ref_text],
        sources=[src_text],
        gpus=gpus,
    )
    score = result["mean_score"]

    return score


def compute_cometkiwi(
    tr_text: str,
    src_text: str,
) -> float:
    """
    Compute CometKiwi score for a translation using the comet package.

    CometKiwi is a reference-free quality estimation metric that combines COMET
    with OpenKiwi. Unlike COMET, it does not require a reference translation and
    evaluates translation quality based only on the source text and machine
    translation.

    Returns a float in the range [0, 1], where 0 indicates a poor translation
    and 1 indicates a perfect translation.
    """
    # Load model once and cache it
    if LOADED_METRICS["cometkiwi"] is None:
        try:
            model_path = download_model("Unbabel/wmt22-cometkiwi-da")
            LOADED_METRICS["cometkiwi"] = load_from_checkpoint(model_path)
        except KeyError as e:
            msg = (
                "Authentication required for CometKiwi model. "
                "Please:\n"
                "1. Visit https://huggingface.co/Unbabel/wmt22-cometkiwi-da and accept the license\n"
                "2. Run: hf auth login\n"
                "3. Enter your HuggingFace token when prompted"
            )
            raise RuntimeError(msg) from e

    model = LOADED_METRICS["cometkiwi"]
    gpus = 1 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 0

    # Prepare data in the format expected by CometKiwi
    data = [{"src": src_text, "mt": tr_text}]

    # Predict returns a Prediction object; access the first score
    model_output = model.predict(data, batch_size=1, gpus=gpus)
    # The Prediction object can be indexed to get individual scores
    score = model_output[0]

    return score
