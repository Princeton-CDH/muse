"""
Metrics for evaluating machine translation quality.

This module provides functions for computing various MT evaluation metrics
including ChrF, and potentially COMET, BLEU, and others in the future.
"""

from muse.metrics.chrf import compute_chrf

__all__ = ["compute_chrf"]
