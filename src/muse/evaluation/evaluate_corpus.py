"""
Generate CSV containing MT metric scores for machine translation corpus.

This script processes a machine translation corpus (JSONL format) and computes
evaluation metrics (ChrF and COMET) for each translation. The output is a CSV
file with columns: tr_id, chrf, comet.

Usage:
    evaluate_corpus.py INPUT OUTPUT [--verbose]
"""

import argparse
import csv
import logging
import pathlib
import sys

import orjsonl
from tqdm import tqdm

from muse.evaluation.metrics import compute_chrf, compute_comet

# Required fields in input machine translation corpus records
REQUIRED_FIELDS = ["tr_id", "src_text", "ref_text", "tr_text"]

logger = logging.getLogger(__name__)


def evaluate_corpus(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    verbose: bool = False,
) -> None:
    """
    Compute evaluation metrics for machine translation corpus and save to CSV.

    Reads machine translation records from input JSONL file, computes ChrF and
    COMET scores for each translation, and writes results to output CSV file
    with columns: tr_id, chrf, comet.
    """
    # Count total records for progress bar
    total_records = sum(1 for _ in orjsonl.stream(input_path))
    logger.info(f"Found {total_records} translations to evaluate")

    # Open output CSV file
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["tr_id", "chrf", "comet"])
        writer.writeheader()

        # Process each translation record
        progress = tqdm(
            orjsonl.stream(input_path),
            total=total_records,
            desc="Evaluating translations",
        )

        for record in progress:
            # Compute metrics
            chrf_score = compute_chrf(
                tr_text=record["tr_text"],
                ref_text=record["ref_text"],
            )
            comet_score = compute_comet(
                tr_text=record["tr_text"],
                src_text=record["src_text"],
                ref_text=record["ref_text"],
            )

            # Write to CSV
            writer.writerow(
                {
                    "tr_id": record["tr_id"],
                    "chrf": chrf_score,
                    "comet": comet_score,
                }
            )

    logger.info(f"Evaluation complete. Results written to: {output_path}")


def main():
    """Main entry point for script."""
    args = argparse.ArgumentParser(
        description="Generate CSV containing MT metric scores for machine translation corpus"
    )
    args.add_argument(
        "input",
        type=pathlib.Path,
        help="Input machine translation corpus JSONL file",
    )
    args.add_argument(
        "output",
        type=pathlib.Path,
        help="Output CSV file with evaluation metrics",
    )
    args.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parsed = args.parse_args()

    # Setup logging
    log_level = logging.DEBUG if parsed.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # Validate input
    if not parsed.input.is_file():
        logger.error(f"{parsed.input} does not exist")
        sys.exit(1)
    if parsed.output.is_file():
        logger.error(f"{parsed.output} exists. Not overwriting.")
        sys.exit(1)

    # Evaluate corpus
    evaluate_corpus(parsed.input, parsed.output, verbose=parsed.verbose)


if __name__ == "__main__":
    main()
