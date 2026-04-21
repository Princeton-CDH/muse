# Copyright Center for Digital Humanities, Princeton University 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""
Build a parallel paragraph corpus (JSONL) from side-by-side translation data (CSVs).
These CSVs are assumed to be within a single directory and have names corresponding
to their corresponding MTO article id (e.g., 30.4.12.csv)

Usage:
    build_paragraph.py input_dir parallel_paragraphs.jsonl
"""

import argparse
import csv
import pathlib
import re
import sys
from collections.abc import Iterator

import ftfy
import orjsonl

# Hard-coded mapping of article MTO ID to source language
MTO_ID2LANG = {
    "30.4.12": "ja",
    "30.4.16": "zh",
    "30.4.18": "ja",
    "30.4.22": "zh",
    "30.4.24": "pt",
    "30.4.28": "es",
    "30.4.30": "pt",
}


def prepare_paragraph_text(text: str, par_id: str) -> str:
    """
    Curate paragraph texts to form parallel paragraph record fields.
    """
    # Clean & normalize text with ftfy
    result = ftfy.fix_text(text)
    # Remove bracketed paragraph id prefix
    if f"[{par_id}]" in result:
        result = result.split(f"[{par_id}]", maxsplit=1)[1]
    # Strip leading/trailing whitespace
    return result.strip()


def extract_parallel_paragraphs(
    in_csv: pathlib.Path, doc_id: str = ""
) -> Iterator[dict[str, str]]:
    """
    Extracts parallel paragraphs from a side-by-side translation CSV.

    Yields partial parallel paragraph records (only text, en_tr, par_id fields)
    """
    # Input CSVs do not have headers
    csv_fieldnames = ["label", "src_text", "en_tr"]
    # Labels (1st column) for paragraphs will have the following form: [par_id]
    par_id_re = re.compile(r"\[(?P<par_id>\d+(\.\d+)?)\]")

    with in_csv.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=csv_fieldnames)
        for item in reader:
            match = par_id_re.match(item["label"])
            # Only extract rows corresponding to paragraphs.
            if match:
                par_id = match.group("par_id")
                text = prepare_paragraph_text(item["src_text"], par_id)
                en_tr = prepare_paragraph_text(item["en_tr"], par_id)
                # Texts must be non-empty
                if text and en_tr:
                    yield {"text": text, "en_tr": en_tr, "par_id": par_id}
                else:
                    # Using doc_id for debugging
                    if doc_id:
                        print(
                            f"WARNING: Skipping {par_id} in {doc_id} due to empty paragraph",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"WARNING: Skipping {par_id} due to empty paragraph",
                            file=sys.stderr,
                        )


def get_parallel_paragraphs(in_dir: pathlib.Path) -> Iterator[dict[str, int | str]]:
    """
    Constructs parallel paragraph records from the side-by-side translation CSVs within
    a given input directory.

    Yields parallel paragraph records
    """
    count = 0
    for csvfile in in_dir.glob("*.csv"):
        # Only examine CSV files with an expected MTO_ID name
        doc_id = csvfile.stem
        if doc_id in MTO_ID2LANG:
            lang = MTO_ID2LANG[doc_id]
            for record in extract_parallel_paragraphs(csvfile, doc_id=doc_id):
                yield {
                    "id": count,
                    "lang": lang,
                    "text": record["text"],
                    "en_tr": record["en_tr"],
                    "doc_id": doc_id,
                    "par_id": record["par_id"],
                }
                count += 1
        else:
            print(f"WARNING: Unexpected CSV {csvfile}", file=sys.stderr)


def save_parallel_paragraph_corpus(
    in_dir: pathlib.Path, out_jsonl: pathlib.Path
) -> None:
    """
    Build parallel paragraph corpus (JSONL) from the side-by-side translation CSVs within
    the input directory.
    """
    orjsonl.save(out_jsonl, get_parallel_paragraphs(in_dir))


def main():
    args = argparse.ArgumentParser()
    args.add_argument("input_dir", type=pathlib.Path)
    args.add_argument("output", type=pathlib.Path)
    parsed = args.parse_args()

    # Validate args
    if not parsed.input_dir.is_dir():
        print(f"ERROR: directory {parsed.input_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    if parsed.output.is_file():
        print(f"ERROR: {parsed.output} exists. Not overwriting.", file=sys.stderr)
        sys.exit(1)

    save_parallel_paragraph_corpus(parsed.input_dir, parsed.output)


if __name__ == "__main__":
    main()
