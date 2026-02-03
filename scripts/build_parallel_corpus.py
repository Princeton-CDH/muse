#!/usr/bin/env python3
"""Build a parallel sentence JSONL corpus from a Notion export file.
Supported languages: Chinese (zh), Japanese (ja), Spanish (es) paired with English (en):
    Chinese (zh) --> English (en)
    Japanese (ja) --> English (en)
    Spanish (es) --> English (en)

Usage:
    python3 scripts/build_parallel_corpus.py --input /path/to/notion_terms.jsonl --output /path/to/parallel.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Language label to code mapping
LABEL_TO_CODE = {
    "chinese": "zh",
    "japanese": "ja",
    "spanish": "es",
    "english": "en",
}

# Match labels like "English A:", "Japanese B:"
LABEL_RE = re.compile(
    r"^(?P<label>English|Japanese|Chinese|Spanish)\s*([A-D]):\s*(?P<rest>.*)$", re.I
)
# Match quoted text (handles straight quotes "...", left curly "...", right curly "...")
QUOTE_RE = re.compile(r'[""](?P<quote>.+?)[""]')
# Match emdash-like separators (em dash, en dash, hyphen) for text/citation extraction fallback
EMDASH_SPLIT_RE = re.compile(r"\s+[\u2014\u2013-]\s+")


def parse_labelled_paragraphs(paragraphs: list[str]) -> dict[str, dict[str, dict]]:
    """Parse paragraphs and return letter -> lang_code -> {text, cite} mapping."""
    blocks: dict[str, dict[str, dict]] = {}
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i].strip()
        m = LABEL_RE.match(p)
        if not m:
            i += 1
            continue

        label_word = m.group("label").strip().lower()
        letter_match = re.search(r"([A-D])", p, re.I)
        letter = letter_match.group(1).upper() if letter_match else "A"
        rest = m.group("rest").strip()

        # Some quotes span multiple paragraphs (e.g., long translations broken up).
        # Join paragraphs until we find the closing quote.
        if rest.count('"') % 2 == 1:
            parts = [rest]
            j = i + 1
            while j < len(paragraphs):
                parts.append(paragraphs[j].strip())
                if paragraphs[j].count('"') > 0:
                    break
                j += 1
            rest = " ".join(parts)
            i = j

        # Extract quoted text and citation
        text = None
        cite = None
        q = QUOTE_RE.search(rest)
        if q:
            text = q.group("quote").strip()
            after = rest[q.end() :].strip()
            if after:
                cite = after.strip()
        else:
            # Fallback: some entries use emdash separators (em dash, en dash, hyphen) instead of quotes.
            # Split on emdash to extract text and citation. This recovers ~200 pairs
            # that would be missed if we only accepted quoted text.
            parts = EMDASH_SPLIT_RE.split(rest, maxsplit=1)
            if parts:
                text = parts[0].strip()
                if len(parts) == 2:
                    cite = parts[1].strip()

        lang_code = LABEL_TO_CODE.get(label_word, label_word)
        if letter not in blocks:
            blocks[letter] = {}
        blocks[letter][lang_code] = {"text": text or "", "cite": cite or ""}
        i += 1

    return blocks


def pair_blocks(
    blocks: dict[str, dict[str, dict]], source_langs: list[str]
) -> list[tuple]:
    """Pair source language blocks with English by letter, fall back to order-based pairing."""
    pairs = []
    letters = sorted(blocks.keys())

    # Collect all source and English blocks in order
    src_blocks = []  # (letter, lang_code, text, cite)
    en_blocks = []  # (letter, text, cite)

    src_blocks.extend(
        (L, s, blocks[L][s]["text"], blocks[L][s].get("cite", ""))
        for L in letters
        for s in source_langs
        if s in blocks[L]
    )
    en_blocks.extend(
        (L, blocks[L]["en"]["text"], blocks[L]["en"].get("cite", ""))
        for L in letters
        if "en" in blocks[L]
    )

    # Letter-matched pairs
    for L in letters:
        entry = blocks[L]
        if "en" not in entry:
            continue
        en = entry["en"]
        for s in source_langs:
            if s in entry:
                src = entry[s]
                pairs.append(
                    (
                        s,
                        src["text"],
                        src.get("cite", ""),
                        en["text"],
                        en.get("cite", ""),
                    )
                )

    # Fallback: pair by position if letter matching is incomplete.
    # Some entries have source blocks without matching English letters (or vice versa).
    # Order-based pairing recovers these by matching the nth source with the nth English block.
    n = min(len(src_blocks), len(en_blocks))
    for idx in range(n):
        _, s_code, s_text, s_cite = src_blocks[idx]
        _, en_text, en_cite = en_blocks[idx]
        # Skip if already added via letter matching
        if not any(
            p[0] == s_code and p[1] == s_text and p[3] == en_text for p in pairs
        ):
            pairs.append((s_code, s_text, s_cite, en_text, en_cite))

    return pairs


def build_corpus(input_path: str, output_path: str, langs: list[str]):
    """Build parallel corpus from Notion export to JSONL."""
    count = 0

    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)

    with (
        input_path_obj.open(encoding="utf-8") as inf,
        output_path_obj.open("w", encoding="utf-8") as outf,
    ):
        for line in inf:
            rec = json.loads(line)
            texts = rec.get("texts", [])
            term = rec.get("term", "")
            blocks = parse_labelled_paragraphs(texts)
            pairs = pair_blocks(blocks, langs)

            for lang_code, src_text, src_cite, en_text, en_cite in pairs:
                if not src_text or not en_text:
                    continue
                count += 1
                obj = {
                    "id": count,
                    "lang": lang_code,
                    "text": src_text.replace("\n", " ").replace("\\n", " ").strip(),
                    "en_tr": en_text.replace("\n", " ").replace("\\n", " ").strip(),
                    "cite": src_cite or "",
                    "en_cite": en_cite or "",
                    "term": term,
                }
                # Remove inter-character spaces from Japanese text.
                # Source data contains unnormalized spaces between CJK characters (e.g., "日 本 の 音 楽").
                if obj["lang"] == "ja":
                    obj["text"] = obj["text"].replace(" ", "")
                json.dump(obj, outf, ensure_ascii=False)
                outf.write("\n")


def main():
    args = argparse.ArgumentParser(
        description="Build parallel sentence JSONL from Notion export."
    )
    args.add_argument("--input", required=True, help="Path to notion_terms.jsonl input")
    args.add_argument(
        "--output", required=True, help="Path to write parallel JSONL output"
    )
    # Configurable to support language pairs beyond zh, ja, es → en (e.g., --langs ko en)
    args.add_argument(
        "--langs",
        nargs="+",
        default=["zh", "ja", "es"],
        help="Source language codes (default: zh ja es)",
    )
    parsed = args.parse_args()

    Path(parsed.output).parent.mkdir(parents=True, exist_ok=True)
    build_corpus(parsed.input, parsed.output, parsed.langs)


if __name__ == "__main__":
    main()
