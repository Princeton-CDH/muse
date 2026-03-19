"""
This module contains custom command recipes for Prodigy.

Recipes:

    - muse-task-progress: Report the current progress for a MuSE annotation
      task at the language and annotator level.
"""

from collections import Counter, defaultdict

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
        "--source", "-s", help="Optional source data to use for progress calculation"
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
        lang_tr_counts = Counter()
        for tr in JSONL(source):
            lang_tr_counts[tr["src_lang"]] += 1

    # Organize data for tables
    lang_anno_counts = Counter()
    annotations_by_session = defaultdict(list)
    for a in annotations:
        if "answer" not in a:
            # Skip entries without answers (hopefully won't encounter)
            continue
        lang_anno_counts[a["src_lang"]] += 1
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
    for lang, n_annotations in sorted(lang_anno_counts.items()):
        row = [lang, n_annotations]
        if source:
            # Number of annotators for the language (only counts those who've submitted an annotation)
            n_annotators = sum(
                1 for a in annotations_by_session if a.startswith(f"{lang}_")
            )
            total_expected = lang_tr_counts[lang] * n_annotators
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
