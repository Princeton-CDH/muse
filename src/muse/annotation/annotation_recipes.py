"""
This module provides custom recipes for Prodigy annotation.

Recipes:
    * ``concept-eval``: Notion concept evaluation recipe.

Example Usage:

    prodigy concept-eval muse_concepts notion-concept-tasks.jsonl -F annotation_recipes.py
"""

from collections.abc import Iterator

import spacy
from prodigy import log, set_hashes
from prodigy.components.preprocess import tokenize_example
from prodigy.components.stream import get_stream
from prodigy.core import Arg, recipe
from prodigy.types import RecipeSettingsType, StreamType


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
    for ex in stream:
        yield ex | {"questions": questions}


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

    # Question prompts for task
    questions = [
        "Q1. For the folloing translation, highlight the translation of the concept",
        "Q2. Evaluate the machine translation of the concept",
        "Q3. Notes / observations",
    ]

    # Initial html template for starting text
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

    options = [
        {"id": "correct", "text": "Correct"},
        {"id": "partial", "text": "Partially correct"},
        {"id": "wrong", "text": "Incorrect"},
        {"id": "verbatim", "text": "Copied verbatim"},
        {"id": "missing", "text": "Missing / Omitted"},
    ]

    blocks = [
        {"view_id": "html", "html_template": init_html_tmpl},
        {"view_id": "ner_manual", "labels": ["CONCEPT"]},
        {"view_id": "html", "html": f"<hr><b>{questions[1]}</b>"},
        {"view_id": "choice", "text": None, "options": options},
        {"view_id": "html", "html": f"<hr><b>{questions[2]}</b>"},
        {"view_id": "text_input", "field_rows": 3},
    ]

    # Setup config
    config = {
        "buttons": ["accept", "undo"],  # remove reject and ignore buttons
        "show_flag": True,  # show flag button to mark weird machine translations
        "honor_token_whitespace": True,  # reflect whitespace accurately (e.g. in case of leading/trailing spaces)
        "blocks": blocks,
        "ner_manual_highlight_chars": True,
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
    }

    return components
