#!/usr/bin/env python3
"""
Generate machine translation corpus from parallel text corpus.

This script processes a parallel text corpus (JSONL format) and generates
machine translations using HuggingFace models. For each input record, it
produces two translations: original→English and English→original.
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from muse.translation.translate import translate

# Constants
REQUIRED_FIELDS = ["id", "lang", "text", "en_tr"]
PROGRESS_INTERVAL = 10  # Log progress every N records

logger = logging.getLogger(__name__)


def validate_input_file(input_path: Path) -> int:
    """
    Validate input parallel corpus file.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    record_count = 0
    with input_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

            # Check required fields
            for field in REQUIRED_FIELDS:
                if field not in record:
                    raise ValueError(
                        f"Missing required field '{field}' at line {line_num}"
                    )
                if not record[field] and field != "id":
                    raise ValueError(f"Empty field '{field}' at line {line_num}")

            record_count += 1

    if record_count == 0:
        raise ValueError("Input file is empty")

    return record_count


def generate_translation_record(
    pair_id: int,
    model: str,
    src_lang: str,
    tgt_lang: str,
    src_text: str,
    ref_text: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Generate a machine translation record.

    Args:
        pair_id: ID from parallel corpus record
        model: HuggingFace model identifier
        src_lang: Source language ISO 639-1 code
        tgt_lang: Target language ISO 639-1 code
        src_text: Text to translate
        ref_text: Reference translation
        verbose: Enable verbose translation output

    Returns:
        Translation record dict with all required fields
    """
    # Generate unique ID
    tr_id = str(uuid.uuid4())

    # Perform translation
    tr_text = translate(
        model=model,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        text=src_text,
        verbose=verbose,
    )

    # Build record
    return {
        "tr_id": tr_id,
        "pair_id": pair_id,
        "model": model,
        "src_lang": src_lang,
        "tr_lang": tgt_lang,
        "src_text": src_text,
        "ref_text": ref_text,
        "tr_text": tr_text,
    }


def process_parallel_corpus(
    input_path: Path,
    output_path: Path,
    model: str,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """
    Process parallel corpus and generate machine translations.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        model: HuggingFace model identifier
        verbose: Enable verbose output

    Returns:
        Tuple of (total_records, success_count, error_count)
    """
    total_records = 0
    success_count = 0
    error_count = 0

    with (
        input_path.open(encoding="utf-8") as infile,
        output_path.open("w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total_records += 1

            # Translation 1: original language → English
            try:
                tr1 = generate_translation_record(
                    pair_id=record["id"],
                    model=model,
                    src_lang=record["lang"],
                    tgt_lang="en",
                    src_text=record["text"],
                    ref_text=record["en_tr"],
                    verbose=verbose,
                )
                outfile.write(json.dumps(tr1, ensure_ascii=False) + "\n")
                outfile.flush()
                success_count += 1
            except Exception as e:
                logger.warning(
                    f"Translation failed for record {record['id']} "
                    f"({record['lang']}→en): {e}"
                )
                error_count += 1

            # Translation 2: English → original language
            try:
                tr2 = generate_translation_record(
                    pair_id=record["id"],
                    model=model,
                    src_lang="en",
                    tgt_lang=record["lang"],
                    src_text=record["en_tr"],
                    ref_text=record["text"],
                    verbose=verbose,
                )
                outfile.write(json.dumps(tr2, ensure_ascii=False) + "\n")
                outfile.flush()
                success_count += 1
            except Exception as e:
                logger.warning(
                    f"Translation failed for record {record['id']} "
                    f"(en→{record['lang']}): {e}"
                )
                error_count += 1

            # Log progress
            if total_records % PROGRESS_INTERVAL == 0:
                logger.info(f"Processed {total_records} records")

    return (total_records, success_count, error_count)


def main():
    """Main entry point for script."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate machine translation corpus from parallel text corpus"
    )
    parser.add_argument(
        "model",
        help="HuggingFace model identifier (e.g., tencent/HY-MT1.5-7B)",
    )
    parser.add_argument("input", help="Input parallel corpus JSONL file")
    parser.add_argument("output", help="Output machine translation corpus JSONL file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    # Validate input and prepare paths
    input_path = Path(args.input)
    record_count = validate_input_file(input_path)
    logger.info(f"Found {record_count} records in input file")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process corpus
    logger.info(f"Starting translation with model: {args.model}")
    try:
        total, success, errors = process_parallel_corpus(
            input_path, output_path, args.model, args.verbose
        )
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(1)

    # Log summary
    logger.info("Processing complete")
    logger.info(f"Total records: {total}")
    logger.info(f"Successful translations: {success}/{total * 2}")
    if errors > 0:
        logger.warning(f"Failed translations: {errors}")
    logger.info(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
