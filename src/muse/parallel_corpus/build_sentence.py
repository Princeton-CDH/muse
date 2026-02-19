"""
Build a parallel sentence corpus (JSONL) from MuSE music-theoretical concept data
from Notion. Each parallel sentence will be a language pair of the source language
of the associated musical theoretical concept and English.

Supported source languages: Chinese, Japanese, Portuguese, Spanish

Usage:
    build_sentence.py notion_terms.jsonl parallel_sentences.jsonl
"""

import argparse
import pathlib
import re
import sys

import ftfy
import orjsonl

# Index of supported languages.
## Currently maps language names to their ISO 639-1 code.
SUPPORTED_LANGUAGES = {
    "Chinese": "zh",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Spanish": "es",
}


def split_on_outer_quotes(text: str) -> tuple[str, str]:
    """
    For an input text beginning with a quoted passage (with double quotes), extract
    and return the quoted passage from the text that follows it. When the input
    does not contain a quote, this returns the input text and an empty string.

    Example:
        - '"Quote" - text after quote.' --> ("Quote", " - text after quote.")
        - 'There is no quote' --> ("There is no quote", "")

    Returns a tuple of the quoted text and the text after the quote.
    """

    if text.startswith('"') and '"' in text[1:]:
        return text[1:].rsplit('"', maxsplit=1)

    return text, ""


def extract_text_data(text_record: dict[str, str | list[str]]) -> tuple[str, str]:
    """
    Extract the text and its citation from a Notion "text" record for a parallel text.
    Input text records are assumed to have the following fields:
      - text: The plaintext of the Notion comment block. As a parallel text it will
              have the following prefix: [Language Name] [A-D]:
      - links: List of strings corresponding to the visibile text for each external
               (i.e., non-term) link in the Notion comment block.

    Returns a tuple of the text and its citation.
    """
    # Get normalized text entry
    body = text_record["text"].split(":", maxsplit=1)[1]
    text = ftfy.fix_text(body).strip()
    cite = ""

    # If there are links, assume the first link corresponds to the citation
    if text_record["links"]:
        link_text = ftfy.fix_text(text_record["links"][0])

        # Split on final instance of citation
        text, post_cite = text.rsplit(link_text, maxsplit=1)
        cite = link_text + post_cite

    # Check for quotation marks to demarcate the quoted text from the citation
    text, post_quote = split_on_outer_quotes(text)
    text = text.strip()
    cite = (post_quote + cite).strip()

    # Check for multiple quoted passages (assume quotations use double quotes)
    ## Quotes and citations will be split by two new lines
    if text_record["links"] and link_text in text:
        # Parentheses let us also capture the citation text
        inner_cite_re = re.compile(rf'"([^"]+{re.escape(link_text)}[^"]+)"')

        new_text = ""
        for i, split in enumerate(inner_cite_re.split(text)):
            split = split.strip()
            if i == 0:
                # First quote
                new_text = split
            elif i % 2 == 0:
                # Later quotes
                new_text += "\n\n" + split
            else:
                # Citation
                cite = f"{split}\n\n{cite}"
        text = new_text

    return text, cite


def get_parallel_texts(term_record) -> list[dict[str, str]]:
    """
    Extracts parallel texts from a term record.

    Returns a list of parallel text records.
    """
    src_re = re.compile(rf"{term_record['lang']} (?P<label>[A-D]):")
    eng_re = re.compile("English (?P<label>[A-D]):")

    src_texts = {}
    eng_texts = {}
    for comment in term_record["comments"]:
        # Check for (potential) source parallel text block
        match = src_re.match(comment["text"])
        if match:
            src_texts[match.group("label")] = comment
            continue
        # Check for (potential) English parallel text block
        match = eng_re.match(comment["text"])
        if match:
            eng_texts[match.group("label")] = comment
            continue

    # Build parallel text records
    parallel_texts = []
    for label in src_texts:
        if label not in eng_texts:
            continue
        # Extract texts and citations
        src_text, src_cite = extract_text_data(src_texts[label])
        eng_text, eng_cite = extract_text_data(eng_texts[label])
        entry = {
            "text": src_text,
            "cite": src_cite,
            "en_tr": eng_text,
            "en_cite": eng_cite,
        }
        parallel_texts.append(entry)
    return parallel_texts


def build_sentence_parallel_corpus(in_jsonl: pathlib.Path, out_jsonl: pathlib.Path):
    """
    Build parallel sentence corpus (JSONL) from a musical-theoretical concept data
    from Notion (JSONL).
    """
    entries = []
    count = 0

    for rec in orjsonl.load(in_jsonl):
        lang_name = rec["lang"]
        # Skip unsupported languages
        if lang_name not in SUPPORTED_LANGUAGES:
            continue
        lang_code = SUPPORTED_LANGUAGES[lang_name]

        for parallel_text in get_parallel_texts(rec):
            count += 1
            entry = (
                {"id": count, "lang": lang_code} | parallel_text | {"term": rec["term"]}
            )
            entries.append(entry)

    orjsonl.save(out_jsonl, entries)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("input", type=pathlib.Path)
    args.add_argument("output", type=pathlib.Path)
    parsed = args.parse_args()

    if not parsed.input.is_file():
        print(f"ERROR: {parsed.input} does not exist")
        sys.exit(1)

    if parsed.output.is_file():
        print(f"ERROR: {parsed.output} exists. Not overwriting")
        sys.exit(1)

    build_sentence_parallel_corpus(parsed.input, parsed.output)


if __name__ == "__main__":
    main()
