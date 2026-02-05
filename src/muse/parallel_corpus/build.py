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
SUPPORTED_LANGUAGES = {"English", "Japanese", "Chinese", "Spanish"}

# Regex pattern for matching language labels in text blocks
LANGUAGE_LABELS = "|".join(sorted(SUPPORTED_LANGUAGES))

# Regex pattern for letter suffixes used in label blocks: A, B, C, D
LETTER_SUFFIX_PATTERN = r"[A-D]"

# Match labels like "English A:", "Japanese B:", "Chinese:", "English:"
# Labels must have a colon, may or may not have letter suffixes [A-D]
LABEL_RE = re.compile(
    rf"^(?P<label>{LANGUAGE_LABELS})\s*(?:{LETTER_SUFFIX_PATTERN})?:\s*(?P<rest>.*)$"
)


# DASH CHARACTERS CONFIGURATION
DASH_CHARS = {
    "\u2014",  # — EM DASH (long dash)
    "\u2013",  # - EN DASH (shorter dash)
    "\u2e3a",  # ⸺ TWO-EM DASH
    "\u2e3b",  # ⸻ THREE-EM DASH
    "\u2015",  # ― HORIZONTAL BAR
    "-",  # - HYPHEN-MINUS
}

# This dash regex exist as the fallback for non-quoted entries)
# Most entries follow format: "quoted text" — citation
#   → We handle this with STRING SLICING (find first/last quote)
# Some entries have NO quotes at all: text — citation
#   → For these, we FALLBACK to this regex to find dash separators
DASH_SPLIT_RE = re.compile(f"[{''.join(sorted(DASH_CHARS))}]+")


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


def extract_text_and_citation(rest: str) -> tuple[str | None, str | None]:
    """Extract quoted text and citation from a paragraph rest."""
    # To preserve nested quotes, we find FIRST and LAST quote positions,
    # and extract everything between them as the main text.
    # Nested quotes example: "诗歌...韵是强化意义、"穿连"诗歌的韵律手段..."
    first_quote = rest.find('"')
    last_quote = rest.rfind('"')

    if first_quote != -1 and last_quote > first_quote:
        # Extract quote content between first and last quote
        text = rest[first_quote + 1 : last_quote].strip()
        after = rest[last_quote + 1 :].strip()

        if after:
            return text, after
        return text, None

    # To handle entried withpit quotes,
    # we look for em dash separators as fallback for text/citation split
    match = DASH_SPLIT_RE.search(rest)
    if match:
        # Split at the first emdash occurrence
        idx = match.start()
        text = rest[:idx].strip()  # text before first em dash
        cite = rest[match.end() :].strip()  # text after first em dash

        # Validate extracted content as a citation by checking it starts
        # with a known citation pattern. If not, treat entire text as content.
        if cite and not re.match(f"^({CITATION_PATTERN})", cite, re.I):
            text = rest.strip()
            cite = None
        return text, cite

    # Fallback for entries with no quotes and no em dashes
    # Use the entire rest as text and leave citation empty
    # Example: "English: Charm (see 乡韵 xiāngyùn )" → text="Charm...", cite=None
    text = rest.strip()
    return text if text else None, None


def parse_labelled_paragraphs(paragraphs: list[str]) -> dict[str, dict[str, dict]]:
    """
    Parse labeled paragraphs into a nested structure grouped by letter and language.
    The letter suffix (A, B, C, D) aligns parallel translations across languages.
    """
    blocks: dict[str, dict[str, dict]] = {}
    i = 0
    while i < len(paragraphs):
        # Normalize text encoding and fix mojibake using ftfy library
        paragraph = ftfy.fix_text(paragraphs[i]).strip()
        # Match language label and letter suffix (A, B, C, D)
        label_match = LABEL_RE.match(paragraph)
        if not label_match:
            i += 1
            continue

        # Extract language label (English, Japanese, Chinese, Spanish)
        label_word = label_match.group("label").strip()

        # Extract letter suffix (A, B, C, or D) from the label area only
        # The letter suffix appears right after the label name, before the colon
        # Example: "English A:" - letter 'A' is after "English" but before ":"
        # We search for [letter + colon] pattern to avoid matching letters
        # in the label name itself (e.g., 'C' in "Chinese")
        label_area = paragraph[: label_match.end() - len(label_match.group("rest")) - 1]
        letter_match = re.search(f"({LETTER_SUFFIX_PATTERN})\\s*:", label_area)
        letter = letter_match.group(1) if letter_match else None

        # Extract rest of the paragraph after language label and letter suffix
        rest = label_match.group("rest").strip()

        # Extract quoted text and citation
        text, cite = extract_text_and_citation(rest)

        # Store in nested dictionary structure by letter and language
        if letter not in blocks:
            blocks[letter] = {}
        blocks[letter][label_word] = {"text": text or "", "cite": cite or ""}
        i += 1

    return blocks


def pair_blocks(blocks: dict[str, dict[str, dict]]) -> list[tuple]:
    """Pair source language blocks with English by letter.

    Strategy:
    1. First, pair entries WITH letter suffixes (A, B, C...) - matched by letter
    2. Then, pair entries WITHOUT letter suffixes (None) - matched by position
    """
    pairs = []

    # PART 1: Pair entries WITH letter suffixes (A, B, C, D...)
    letters = sorted(k for k in blocks if k is not None)

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
                    (
                        lang_name,
                        src["text"],
                        src.get("cite", ""),
                        en["text"],
                        en.get("cite", ""),
                    )
                )

    # PART 2: Pair entries WITHOUT letter suffixes (None) by position
    # Some entries have "Chinese:" without letter suffix, paired with "English:" by position
    if None in blocks:
        entry = blocks[None]
        if "English" in entry:
            en = entry["English"]
            for lang_name in SUPPORTED_LANGUAGES:
                if lang_name == "English":
                    continue
                if lang_name in entry:
                    src = entry[lang_name]
                    # Check if not already added by letter matching
                    if not any(
                        p[0] == lang_name and p[1] == src["text"] for p in pairs
                    ):
                        pairs.append(
                            (
                                lang_name,
                                src["text"],
                                src.get("cite", ""),
                                en["text"],
                                en.get("cite", ""),
                            )
                        )

    return pairs


def build_sentence_parallel_corpus(input_path: str, output_path: str):
    """Build parallel corpus from Notion export to JSONL."""
    count = 0

    for rec in orjsonl.load(input_path):
        texts = rec.get("texts", [])
        term = rec.get("term", "")
        # Group labeled paragraphs by letter suffix (A, B, C...) to align parallel translations
        blocks = parse_labelled_paragraphs(texts)
        # Pair source language blocks with English by letter
        pairs = pair_blocks(blocks)

        for lang_name, src_text, src_cite, en_text, en_cite in pairs:
            if not src_text or not en_text:
                continue
            count += 1
            obj = {
                "id": count,
                "lang": lang_name,
                "text": src_text,
                "en_tr": en_text,
                "cite": src_cite or "",
                "en_cite": en_cite or "",
                "term": term,
            }
            # Use orjsonl to append JSONL entry, preserving original formatting
            orjsonl.append(output_path, obj)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--input", type=pathlib.Path, required=True)
    args.add_argument("--output", type=pathlib.Path, required=True)
    parsed = args.parse_args()

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    build_sentence_parallel_corpus(str(parsed.input), str(parsed.output))


if __name__ == "__main__":
    main()
