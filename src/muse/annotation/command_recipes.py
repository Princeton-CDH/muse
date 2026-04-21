# Copyright Center for Digital Humanities, Princeton University 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""
This module contains custom command recipes for Prodigy.

Recipes:

    - muse-task-progress: Report the current progress for a MuSE annotation
      task at the language and annotator level.
"""

from collections import defaultdict

from prodigy.components.db import connect
from prodigy.components.loaders import JSONL
from prodigy.core import Arg, recipe
from prodigy.errors import RecipeError
from prodigy.types import SourceType
from prodigy.util import msg


@recipe(
    "muse-task-progress",
    dataset=Arg(help="Prodigy dataset ID"),
    source=Arg(
        "--source",
        "-s",
        help="Optional source data JSONL to use for progress calculation",
    ),
)
def muse_task_progress(dataset: str, source: SourceType | None = None) -> None:
    # Load examples from database
    DB = connect()
    if dataset not in DB:
        raise RecipeError(f"Can't find dataset '{dataset}' in database {DB.db_name}")
    annotations = DB.get_dataset_examples(dataset)
    msg.good(f"Loaded {len(annotations)} annotations from dataset '{dataset}'")

    # Get language-specific translation counts if the source data is provided
    if source:
        lang_tr_counts = defaultdict(int)
        for tr in JSONL(source):
            lang_tr_counts[tr["src_lang"]] += 1

    # Organize data for tables
    ## Infer languages if source data is not provided
    langs = set() if not source else lang_tr_counts.keys()
    annotations_by_session = defaultdict(list)
    for a in annotations:
        if "answer" not in a:
            # Skip entries without answers (hopefully won't encounter)
            continue
        if not source:
            langs.add(a["src_lang"])
        tr_id = a["tr_id"]
        session_name = a["_session_id"].split(f"{dataset}-")[-1]
        annotations_by_session[session_name].append(tr_id)

    # Build language table
    ## Set header and column alignments
    header = ["Language", "# Annotations"]
    aligns = ["l", "r"]
    if source:
        header.append("Progress*")
        aligns.append("r")
    ## Build row data
    rows = []
    for lang in sorted(langs):
        # Determine the (existing) annotators for the language
        lang_annotators = [
            s for s in annotations_by_session if s.startswith(f"{lang}_")
        ]
        # For each annotator, only count one annotation per translation
        # NOTE: Generally, annotators should not annotate the same translation
        #       multiple times, but it can happen.
        n_annotations = sum(
            len(set(annotations_by_session[la])) for la in lang_annotators
        )
        row = [lang, n_annotations]
        if source:
            n_annots = len(lang_annotators)
            # If no annotators set total_expect to 1 to avoid divide by zero error
            total_expected = lang_tr_counts[lang] * n_annots if n_annots else 1
            row.append(f"{n_annotations / total_expected * 100:.1f}%")
        rows.append(row)
    msg.table(
        rows,
        title="Overall Progress by Language",
        header=header,
        aligns=aligns,
        divider=True,
    )

    # Build session table
    ## Set header and column alignments
    header = ["Annotator", "Count", "Unique"]
    aligns = ["l", "r", "r"]
    if source:
        header.append("Progress")
        aligns.append("r")
    ## Build row data
    rows = []
    for session, trs in sorted(annotations_by_session.items()):
        count = len(trs)
        unique = len(set(trs))
        row = [session, count, unique]
        if source:
            lang = session.split("_", maxsplit=1)[0]
            row.append(f"{unique / lang_tr_counts[lang] * 100:.1f}%")
        rows.append(row)
    msg.table(
        rows,
        title="Annotator Progress",
        header=header,
        aligns=aligns,
        divider=True,
    )
