"""
Generate machine translation corpus from parallel text corpus.

This script processes a parallel text corpus (JSONL format) and generates
machine translations using supported translation models. For each input record, it
produces two translations: original→English and English→original.

Each output record represents a single translation with fields: tr_id, pair_id,
model, src_lang, tr_lang, src_text, ref_text, tr_text.

Usage:
    translate_corpus.py MODEL INPUT OUTPUT [--verbose]
"""

import argparse
import logging
import pathlib
import sys
import uuid
from collections.abc import Iterator

import orjsonl
from tqdm import tqdm

from muse.translation.translate import SUPPORTED_MODELS, translate

# Required fields in input parallel corpus records
REQUIRED_FIELDS = ["id", "lang", "text", "en_tr"]

logger = logging.getLogger(__name__)


def validate_model(model: str) -> None:
    """
    Validate that the specified model is supported.

    Args:
        model: Model identifier

    Raises:
        ValueError: If model is not supported
    """
    if model not in SUPPORTED_MODELS:
        supported = list(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Unsupported model: {model}. Supported models: {', '.join(supported)}"
        )


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

    Args:
        pair_id: ID of the source parallel text pair
        model: Model identifier
        src_lang: Source language ISO 639-1 code
        tgt_lang: Target language ISO 639-1 code
        src_text: Source text to translate
        ref_text: Reference translation text
        verbose: If True, print timing and token information during translation

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


def generate_translations(
    input_path: pathlib.Path,
    model: str,
    verbose: bool = False,
) -> Iterator[dict[str, str]]:
    """
    Generate translation records from parallel corpus.

    Yields translation records one at a time for memory efficiency.
    For each input record, generates two translations:
    1. Original language → English
    2. English → Original language

    Args:
        input_path: Path to input parallel corpus JSONL file
        model: Model identifier
        verbose: If True, print timing and token information during translation

    Yields:
        Translation record dicts with fields: tr_id, pair_id, model,
        src_lang, tr_lang, src_text, ref_text, tr_text
    """
    for record in orjsonl.stream(input_path):
        # Validate required fields at record level
        missing_fields = [field for field in REQUIRED_FIELDS if field not in record]
        if missing_fields:
            logger.warning(
                f"Skipping record {record.get('id', 'unknown')}: "
                f"missing fields {missing_fields}"
            )
            continue

        # Translation 1: original language → English
        try:
            src_to_en = generate_translation_record(
                pair_id=record["id"],
                model=model,
                src_lang=record["lang"],
                tgt_lang="en",
                src_text=record["text"],
                ref_text=record["en_tr"],
                verbose=verbose,
            )
            yield src_to_en
        except Exception as e:
            logger.warning(
                f"Translation failed for record {record['id']} "
                f"({record['lang']}→en): {e}"
            )

        # Translation 2: English → original language
        try:
            en_to_src = generate_translation_record(
                pair_id=record["id"],
                model=model,
                src_lang="en",
                tgt_lang=record["lang"],
                src_text=record["en_tr"],
                ref_text=record["text"],
                verbose=verbose,
            )
            yield en_to_src
        except Exception as e:
            logger.warning(
                f"Translation failed for record {record['id']} "
                f"(en→{record['lang']}): {e}"
            )


def save_translated_corpus(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    model: str,
    verbose: bool = False,
) -> None:
    """
    Generate machine translations from parallel corpus and save to JSONL file.

    Reads parallel text records from input file, generates bidirectional
    translations (original→English and English→original), and writes
    translation records to output file. Each output record represents a
    single translation with fields: tr_id, pair_id, model, src_lang, tr_lang,
    src_text, ref_text, tr_text.

    Uses streaming to handle large corpora efficiently without loading
    all translations into memory.

    Args:
        input_path: Path to input parallel corpus JSONL file
        output_path: Path to output machine translation corpus JSONL file
        model: Model identifier
        verbose: If True, print timing and token information during translation
    """
    # Count total records for progress bar
    total_records = sum(1 for _ in orjsonl.stream(input_path))

    logger.info(f"Found {total_records} records in input file")
    logger.info(f"Starting translation with model: {model}")

    # Generate translations with progress bar
    # Each input record produces 2 output records (bidirectional)
    translations_generator = generate_translations(input_path, model, verbose)

    try:
        with tqdm(
            total=total_records * 2, desc="Translating", unit="translation"
        ) as pbar:

            def progress_wrapper():
                for translation in translations_generator:
                    pbar.update(1)
                    yield translation

            orjsonl.save(output_path, progress_wrapper())
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        raise

    logger.info(f"Processing complete. Output written to: {output_path}")


def main():
    """Main entry point for script."""
    args = argparse.ArgumentParser(
        description="Generate machine translation corpus from parallel text corpus"
    )
    args.add_argument(
        "model",
        help="Model identifier (e.g., tencent/HY-MT1.5-7B, google/translation-llm)",
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

    # Validate model early (fail fast)
    try:
        validate_model(parsed.model)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Validate input
    if not parsed.input.is_file():
        logger.error(f"{parsed.input} does not exist")
        sys.exit(1)

    if parsed.output.is_file():
        logger.error(f"{parsed.output} exists. Not overwriting")
        sys.exit(1)

    # Process corpus
    try:
        save_translated_corpus(
            parsed.input, parsed.output, parsed.model, parsed.verbose
        )
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
