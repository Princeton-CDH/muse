#!/usr/bin/env python3
"""
Generate machine translation corpus from parallel text corpus.

This script processes a parallel text corpus (JSONL format) and generates
machine translations using HuggingFace models. For each input record, it
produces two translations: original→English and English→original.

Usage:
    translate_corpus.py MODEL INPUT OUTPUT [--verbose]
"""

import argparse
import logging
import pathlib
import sys
import uuid

import orjsonl

from muse.translation.translate import translate

# Required fields in input parallel corpus records
REQUIRED_FIELDS = ["id", "lang", "text", "en_tr"]
# Log progress every N records
PROGRESS_INTERVAL = 10

logger = logging.getLogger(__name__)


def validate_input_file(input_path: pathlib.Path) -> int:
    """
    Validate input parallel corpus file.

    Returns the number of valid records found.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    record_count = 0
    for line_num, record in enumerate(orjsonl.stream(input_path), 1):
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in record:
                raise ValueError(f"Missing required field '{field}' at line {line_num}")
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
) -> dict[str, str]:
    """
    Generate a machine translation record.

    Returns a translation record dict with all required fields.
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
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    model: str,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """
    Process parallel corpus and generate machine translations.

    Returns a tuple of (total_records, success_count, error_count).
    """
    total_records = 0
    success_count = 0
    error_count = 0

    translations = []
    for record in orjsonl.stream(input_path):
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
            translations.append(tr1)
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
            translations.append(tr2)
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

    # Write all translations to output file
    orjsonl.save(output_path, translations)

    return (total_records, success_count, error_count)


def main():
    """Main entry point for script."""
    args = argparse.ArgumentParser(
        description="Generate machine translation corpus from parallel text corpus"
    )
    args.add_argument(
        "model",
        help="HuggingFace model identifier (e.g., tencent/HY-MT1.5-7B)",
    )
    args.add_argument(
        "input", type=pathlib.Path, help="Input parallel corpus JSONL file"
    )
    args.add_argument(
        "output", type=pathlib.Path, help="Output machine translation corpus JSONL file"
    )
    args.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parsed = args.parse_args()

    # Setup logging
    log_level = logging.DEBUG if parsed.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    # Validate input
    if not parsed.input.is_file():
        print(f"ERROR: {parsed.input} does not exist")
        sys.exit(1)

    if parsed.output.is_file():
        print(f"ERROR: {parsed.output} exists. Not overwriting")
        sys.exit(1)

    # Validate input file format
    record_count = validate_input_file(parsed.input)
    logger.info(f"Found {record_count} records in input file")

    # Ensure output directory exists
    parsed.output.parent.mkdir(parents=True, exist_ok=True)

    # Process corpus
    logger.info(f"Starting translation with model: {parsed.model}")
    try:
        total, success, errors = process_parallel_corpus(
            parsed.input, parsed.output, parsed.model, parsed.verbose
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
    logger.info(f"Output written to: {parsed.output}")


if __name__ == "__main__":
    main()
