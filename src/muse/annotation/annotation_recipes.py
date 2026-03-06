"""
This module provides custom recipes for Prodigy annotation.

Recipes:
    * ``concept-eval``: Notion concept evaluation recipe.

Example Usage:

    prodigy concept-eval muse_concepts notion-concept-tasks.jsonl -F annotation_recipes.py
"""

import pathlib
from collections.abc import Iterator

import spacy
from prodigy import log, set_hashes
from prodigy.components.preprocess import tokenize_example
from prodigy.components.stream import get_stream
from prodigy.core import Arg, Controller, recipe
from prodigy.types import RecipeSettingsType, StreamType
from prodigy.util import TASK_HASH_ATTR

# Mapping of languages to number of required annotations
LANG2N_ANNOT = {
    "es": 1,
    "ja": 2,
    "pt": 1,
    "zh": 2,
}


def get_total_examples_target(in_jsonl: pathlib.Path) -> int:
    """
    Determines the total number of annotations expected for the given JSONL
    """
    n_annot = 0
    for ex in get_stream(in_jsonl):
        n_annot += LANG2N_ANNOT[ex["src_lang"]]
    return n_annot


def add_tokens(stream: StreamType) -> Iterator[StreamType]:
    """
    Workaround to add tokens using appropriate tokenizer.
    """
    tokenizers = {}

    for ex in stream:
        lang = ex["tr_lang"]
        if lang not in tokenizers:
            tokenizers[lang] = spacy.blank(lang)
        nlp = tokenizers[lang]
        yield tokenize_example(ex, nlp(ex["text"]))


def add_questions(questions, stream: StreamType) -> Iterator[StreamType]:
    """
    Add questions to items in stream
    """
    for ex in stream:
        yield ex | {"questions": questions}


def get_lang_annotators(ctrl: Controller, lang: str) -> list[str]:
    """
    Gathers the annotators for a specific language.
    """
    # Get all known session names
    annotators = ctrl.session_ids
    return [a for a in annotators if f"-{lang}_" in a]


def lang_task_router(ctrl: Controller, session_id: str, item: dict) -> list[str]:
    """
    Route tasks based on language and selecting 2 random annotators when possible
    """
    # Get source language of item
    src_lang = item["src_lang"]
    # Required number of annotations
    n_annot = LANG2N_ANNOT[src_lang]
    # Get pool of annotators
    pool = get_lang_annotators(ctrl, src_lang)
    # Use hash to select n_annot random, but deterministic annotators
    task_hash = item[TASK_HASH_ATTR]
    selected_annotators = []
    while len(selected_annotators) < n_annot:
        # If the pool is empty, just return the annotators so far
        if len(pool) == 0:
            return selected_annotators
        i = task_hash % len(pool)
        selected_annotators.append(pool.pop(i))
    return selected_annotators


@recipe(
    "concept-eval",
    dataset=Arg(help="Dataset to save answers to"),
    source=Arg(help="The source data as a JSONL file"),
)
def concept_eval_recipe(
    dataset: str,
    source: str,
) -> RecipeSettingsType:
    # TODO: Consider adding an instruction page. See https://prodi.gy/docs/api-web-app#instructions
    log("RECIPE: Starting recipe concept-eval", locals())

    # Task elements most likely to be modified
    ## Question prompts for task
    questions = [
        "Q1. For the following translation, highlight the translation of the concept",
        "Q2. Evaluate the machine translation of the concept",
        "Q3. Notes / observations",
    ]

    q2_labels = [
        {"id": "correct", "text": "Correct"},
        {"id": "partial", "text": "Partially correct"},
        {"id": "wrong", "text": "Incorrect"},
        {"id": "verbatim", "text": "Copied verbatim"},
        {"id": "missing", "text": "Missing / Omitted"},
    ]

    def validate_answer(eg) -> None:
        q1_spans = eg.get("spans", [])
        q2_selected = eg.get("accept", [])

        # Validate Q1 answer
        if len(q1_spans) == 0 and "missing" not in q2_selected:
            raise ValueError(
                "Must select the translation of the concept if it wasn't omitted entirely"
            )
        # Validate Q2 answer
        if len(q2_selected) == 0:
            raise ValueError("Missing answer for Q2")
        elif "missing" in q2_selected and len(q1_spans) > 0:
            raise ValueError(
                "If the concept was omitted in the translation, no selections should be made for Q1"
            )

    # Recipe organization
    ## Initial html template for starting text
    init_html_tmpl = "\n".join(
        [
            "<h2>Concept: {{term}}</h2>",
            "<p><b>Source Text</b>",
            "{{src_text}}",
            "<details>",
            "\t<summary><b>Professional English Translation</b></summary>{{ref_text}}",
            "</details>",
            f"<hr><b>{questions[0]}</b>",
        ]
    )
    ## Arrangement of combined interfaces
    blocks = [
        {"view_id": "html", "html_template": init_html_tmpl},
        {"view_id": "ner_manual", "labels": ["CONCEPT"]},
        {"view_id": "html", "html": f"<hr><b>{questions[1]}</b>"},
        {"view_id": "choice", "text": None, "options": q2_labels},
        {"view_id": "html", "html": f"<hr><b>{questions[2]}</b>"},
        {"view_id": "text_input", "field_rows": 3},
    ]
    ## Configuration
    config = {
        "buttons": ["accept", "reject", "undo"],  # remove ignore button
        "show_flag": True,  # show flag button to mark weird machine translations
        "honor_token_whitespace": True,  # reflect whitespace accurately (e.g. in case of leading/trailing spaces)
        "blocks": blocks,
        "ner_manual_highlight_chars": True,
        "custom_theme": {"cardMaxWidth": "70%"},
        "allow_work_stealing": False,
        "total_examples_target": get_total_examples_target(source),
    }

    # Create stream
    stream = get_stream(source)
    stream.apply(add_tokens, stream)

    # set hashes
    def set_stream_hashes(stream: StreamType) -> Iterator[StreamType]:
        for ex in stream:
            yield set_hashes(
                ex, input_keys=("tr_id"), task_keys=("questions", "spans", "options")
            )

    stream.apply(set_stream_hashes, stream)

    components = {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": config,
        "validate_answer": validate_answer,
        "task_router": lang_task_router,
    }

    return components
