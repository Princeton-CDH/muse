# Copyright Center for Digital Humanities, Princeton University 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""
This script is used to convert Prodigy exported annotation data (JSONL) into
a human-readable spreadsheet (CSV).

Currently supports the following annotation recipes: concept-eval

Example Usage:

    build_results.py concept-eval muse-concepts.jsonl concept-annotations.csv
"""

import argparse
import csv
import pathlib
import sys

import orjsonl

from muse.annotation.constants import CONCEPT_EVAL_TYPOLOGY


def get_span_texts(
    spans: list[dict[str, int | str]],
    ref_text: str,
    trim_ws: bool = True,
) -> list[str]:
    """
    For a list of span annotations, extract their corresponding text from the
    reference text. Removes leading and trailing whitespace from span texts
    by default.
    """
    span_texts = []
    for span in spans:
        span_text = ref_text[span["start"] : span["end"]]
        if trim_ws:
            span_text = span_text.strip()
        span_texts.append(span_text)
    return span_texts


def save_concept_eval_results(dataset: pathlib.Path, out_csv) -> None:
    """
    Build the annotation results (CSV) for exported annotations (JSONL)
    for the concept-eval Prodigy annotation recipe.
    """
    fieldnames = [
        "tr_id",
        "pair_id",
        "model",
        "language",
        "term",
        "src_text",
        "ref_tr",
        "mt_text",
        "q1",
        "q2",
        "q3",
        "annotator",
    ]
    with out_csv.open(mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for anno in orjsonl.stream(dataset):
            # Q1's answers are in spans field
            q1_spans = get_span_texts(anno.get("spans", []), anno["text"])
            # Q2's answer is in accept field which contains a list with
            # a single label id
            q2_type = CONCEPT_EVAL_TYPOLOGY[anno["accept"][0]]
            # Q3's answer if present is in the user_input field
            q3_text = anno.get("user_input", "").strip()
            # session id = "concept-eval-[lang]_[annotator]"
            annotator = anno["_session_id"].rsplit("_", maxsplit=1)[-1]
            res_row = {
                "tr_id": anno["tr_id"],
                "pair_id": anno["pair_id"],
                "model": anno["model"],
                "language": anno["lang_name"],
                "term": anno["term"],
                "src_text": anno["src_text"],
                "ref_tr": anno["ref_text"],
                "mt_text": anno["text"],
                "q1": "\n".join(q1_spans),
                "q2": q2_type,
                "q3": q3_text,
                "annotator": annotator,
            }
            writer.writerow(res_row)


def save_annotation_results(
    task: str,
    dataset: pathlib.Path,
    out_csv: pathlib.Path,
) -> None:
    """
    Build annotation results (CSV) for a given Prodigy task and
    exported annotations (JSONL).
    """
    if task == "concept-eval":
        save_concept_eval_results(dataset, out_csv)
    else:
        raise ValueError(f"Unknown task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Builds results csv for a given Prodigy annotation recipe and datset export"
    )
    parser.add_argument("task", type=str, help="Name of the Prodigy annotation recipe")
    parser.add_argument(
        "dataset",
        type=pathlib.Path,
        help="Prodigy dataset export (JSONL) of annotations",
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="Output annotation results (CSV)",
    )
    parsed = parser.parse_args()

    # Validate args
    if not parsed.dataset.is_file():
        print(f"ERROR: {parsed.dataset} does not exist", file=sys.stderr)
        sys.exit(1)
    if parsed.output.is_file():
        print(f"ERROR: {parsed.output} exists. Not overwriting.", file=sys.stderr)
        sys.exit(1)

    save_annotation_results(
        parsed.task,
        parsed.dataset,
        parsed.output,
    )


if __name__ == "__main__":
    main()
