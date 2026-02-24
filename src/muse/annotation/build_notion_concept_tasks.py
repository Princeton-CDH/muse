"""
This script is used to prepare the input for the Notion concept annotation task
with Prodigy. This corpus is built using the Notion parallel sentence corpus
and some number of Notion sentence translation corpora.

Example Usage:

    python out.jsonl notion-parallel-sents.jsonl --mt-corpus mt_corpus.jsonl
    python out.jsonl notion-parallel-sents.jsonl --mt-corpus mt1.jsonl mt2.jsonl
"""

import argparse
import pathlib
import sys

import polars as pl


def build_tasks(
    parallel_corpus: pathlib.Path, mt_corpora: list[pathlib.Path], output: pathlib.Path
) -> None:
    # Load parallel sentences
    terms_df = (
        pl.read_ndjson(parallel_corpus)
        # Select terms of interest, namely the record id and term
        .select(["id", "term"])
        # Rename id to pair_id for join
        .rename({"id": "pair_id"})
    )
    # Load machine translations
    mt_df = (
        pl.concat([pl.read_ndjson(corpus) for corpus in mt_corpora])
        # Ignore back translations
        .filter(pl.col("src_lang") != "en")
        # Rename translation text to text so for span annotations in prodigy
        .rename({"tr_text": "text"})
    )

    # Join dataframes on pair_id
    result_df = mt_df.join(terms_df, "pair_id")

    # Write output
    result_df.write_ndjson(output)


def main():
    parser = argparse.ArgumentParser(
        description="Builds prodigy annotation tasks from Notion sentence translations"
    )
    parser.add_argument("output", type=pathlib.Path, help="Output prodigy task JSONL")
    parser.add_argument(
        "parallel_corpus", type=pathlib.Path, help="Parallel notion sentence corpus"
    )
    parser.add_argument(
        "--mt-corpus",
        nargs="+",
        type=pathlib.Path,
        required=True,
        help="One or more machine translation corpora",
    )

    args = parser.parse_args()

    if not args.parallel_corpus:
        print(f"Error: {args.parallel_corpus} does not exist", sys.stderr)
        sys.exit(1)
    for f in args.mt_corpus:
        if not f.is_file():
            print(f"Error: {f} does not exist", sys.stderr)
            sys.exit(1)
    if args.output.is_file():
        print(f"Error: {args.output} exist. Not overwriting.")
        sys.exit(1)

    build_tasks(
        args.parallel_corpus,
        args.mt_corpus,
        args.output,
    )


if __name__ == "__main__":
    main()
