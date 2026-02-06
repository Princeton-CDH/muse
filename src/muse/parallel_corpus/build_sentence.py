#!/usr/bin/env python3
"""Build a parallel sentence JSONL corpus from a Notion export file.
Supported language pairs: Chinese, Japanese, Spanish paired with English.

Usage:
    python3 src/muse/parallel_corpus/build.py --input /path/to/notion_terms.jsonl --output /path/to/parallel.jsonl
"""

from __future__ import annotations

import argparse
import pathlib
import re

import ftfy
import orjsonl

# LANGUAGE CONFIGURATION
# To add/remove supported languages, update SUPPORTED_LANGUAGES below
# Format: "Human-readable name": "ISO 639-1 code"
SUPPORTED_LANGUAGES = {
    "Chinese": "zh",
    "English": "en",
    "Japanese": "ja",
    "Spanish": "es",
}

# Regex pattern for matching language labels in text blocks
# Extract language names from dictionary keys
LANGUAGE_LABELS = "|".join(sorted(SUPPORTED_LANGUAGES.keys()))

# Regex pattern for letter suffixes used in label blocks: A, B, C, D
LETTER_SUFFIX_PATTERN = r"[A-D]"

# Match labels like "English A:", "Japanese B:", "Chinese A:", "Spanish B:"
# Format: "[Language] [A-D]:" - required space between name and letter suffix
LABEL_RE = re.compile(
    rf"^(?P<label>{LANGUAGE_LABELS}) (?P<letter>{LETTER_SUFFIX_PATTERN}):\s*(?P<rest>.*)$"
)


# CITATION PATTERNS CONFIGURATION
# Citation indicators that appear at the start of citation text after the main content
CITATION_PATTERNS = {
    "p.",  # page reference: p. 268
    "pp.",  # pages reference: pp. 1070-1071
    "[",  # bracket reference: [1.17]
    "(",  # parenthetical reference: (Footnote...), (See also...)
    "Footnote",  # footnote reference: Footnote 28
    "Refer to",  # cross-reference: Refer to Section 1
    "See also",  # cross-reference: See also [2.7]
}

# Regex pattern to validate extracted citation content
# Checks if text starts with a known citation pattern
# Escape special regex characters in patterns before joining
CITATION_PATTERN = "|".join(re.escape(p) for p in sorted(CITATION_PATTERNS))


def extract_text_and_citation(rest: str) -> tuple[str, str]:
    """Extract quoted text and citation from a paragraph rest.

    Assumes double quotes (") are used to mark quoted text in the input.
    Handles nested quotes by extracting content between first and last quote.

    Args:
        rest: Text content after the language label (e.g., after "English A: ")

    Returns:
        A tuple of (text, citation). Returns ("", "") if no valid text is found.
    """
    rest = rest.strip()
    first_quote = rest.find('"')
    last_quote = rest.rfind('"')

    if first_quote != -1 and last_quote > first_quote:
        # Extract quote content between first and last quote
        text = rest[first_quote + 1 : last_quote].strip()
        after = rest[last_quote + 1 :].strip()

        return text, after

    # To handle entries without quotes,
    # split on the LAST whitespace to separate text from citation
    # rsplit(maxsplit=1) splits from right, so we get [text, citation]
    parts = rest.rsplit(None, 1)  # Split on any whitespace, max 1 split

    if len(parts) == 2:
        potential_text, potential_cite = parts
        # Validate citation by checking it starts with a known citation pattern
        if potential_cite and re.match(f"^({CITATION_PATTERN})", potential_cite, re.I):
            text = potential_text.strip()
            cite = potential_cite.strip()
        else:
            # Not a valid citation, treat entire text as content
            return rest, ""
    else:
        return rest, ""

    # Fallback for entries with no quotes and no dashes
    # Use the entire rest as text and leave citation empty
    return text, cite


def parse_labelled_paragraphs(entries: list[str]) -> dict[str, dict[str, dict]]:
    """
    Parse labeled paragraphs into a nested structure grouped by letter and language.
    The letter suffix (A, B, C, D) aligns parallel translations across languages.
    Format: "[Language] [A-D]:"
    """
    blocks: dict[str, dict[str, dict]] = {}
    for entry in entries:
        # Normalize text encoding and fix mojibake using ftfy library
        entry = ftfy.fix_text(entry).strip()
        # Match language label and letter suffix (A, B, C, D)
        label_match = LABEL_RE.match(entry)
        if not label_match:
            continue

        # Extract language label (English, Japanese, Chinese, Spanish)
        label_word = label_match.group("label")

        # Extract letter suffix (A, B, C, or D) using named group
        letter = label_match.group("letter")

        # Extract rest of the paragraph after language label and letter suffix
        rest = label_match.group("rest").strip()

        # Extract quoted text and citation
        text, cite = extract_text_and_citation(rest)

        # Store in nested dictionary structure by letter and language
        if letter not in blocks:
            blocks[letter] = {}
        blocks[letter][label_word] = {"text": text, "cite": cite}

    return blocks


def pair_blocks(blocks: dict[str, dict[str, dict]]) -> list[dict]:
    """Pair source language blocks with English by letter suffix.

    Format: "[Language] [A-D]:"
    Only processes entries with letter suffixes (A, B, C, D).
    """
    pairs: list[dict] = []

    # Pair entries by letter suffix (A, B, C, D...)
    letters = sorted(blocks.keys())

    for L in letters:
        entry = blocks[L]
        if "English" not in entry:
            continue
        en = entry["English"]
        for lang_name in SUPPORTED_LANGUAGES:
            if lang_name == "English":
                continue
            if lang_name in entry:
                src = entry[lang_name]
                pairs.append(
                    {
                        "lang_code": SUPPORTED_LANGUAGES[lang_name],
                        "src_text": src["text"],
                        "src_cite": src.get("cite", ""),
                        "en_text": en["text"],
                        "en_cite": en.get("cite", ""),
                    }
                )

    return pairs


def build_sentence_parallel_corpus(input_path: pathlib.Path, output_path: pathlib.Path):
    """Build parallel corpus from Notion export to JSONL."""
    entries = []
    count = 0

    for rec in orjsonl.load(input_path):
        texts = rec.get("texts", [])
        term = rec.get("term", "")
        # Group labeled paragraphs by letter suffix (A, B, C...) to align parallel translations
        blocks = parse_labelled_paragraphs(texts)
        # Pair source language blocks with English by letter
        pairs = pair_blocks(blocks)

        for pair in pairs:
            src_text = pair["src_text"]
            en_text = pair["en_text"]
            if not src_text or not en_text:
                continue
            count += 1
            obj = {
                "id": count,
                "lang": pair["lang_code"],
                "text": src_text,
                "en_tr": en_text,
                "cite": pair["src_cite"] or "",
                "en_cite": pair["en_cite"] or "",
                "term": term,
            }
            entries.append(obj)

    # Use orjsonl.save() to overwrite the output file with all entries
    orjsonl.save(output_path, entries)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("input", type=pathlib.Path)
    args.add_argument("output", type=pathlib.Path
    parsed = args.parse_args()

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    build_sentence_parallel_corpus(parsed.input, parsed.output)


if __name__ == "__main__":
    main()
