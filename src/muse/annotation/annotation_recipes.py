"""
This module provides custom recipes for Prodigy annotation.

Recipes:
    * ``concept-eval``: Notion concept evaluation recipe.

Example Usage:

    prodigy concept-eval muse_concepts notion-concept-tasks.jsonl -F annotation_recipes.py
"""

import pathlib
from collections import Counter
from collections.abc import Iterator

import spacy
from prodigy import log, set_hashes
from prodigy.components.preprocess import tokenize_example
from prodigy.components.routers import log_router
from prodigy.components.session import Session
from prodigy.components.stream import get_stream
from prodigy.core import Arg, Controller, recipe
from prodigy.structured_types import get_input_hash, get_task_hash
from prodigy.types import PathInputType, RecipeSettingsType, StreamType
from prodigy.util import INPUT_HASH_ATTR, TASK_HASH_ATTR


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


def get_lang_name(lang_code: str):
    """
    Returns language name for given language code
    """
    if lang_code == "zh":
        return "Chinese"
    if lang_code == "ja":
        return "Japanese"
    if lang_code == "pt":
        return "Portuguese"
    if lang_code == "es":
        return "Spanish"
    raise ValueError(f"Unknown language code '{lang_code}'")


def add_lang_name(stream: StreamType) -> Iterator[StreamType]:
    """
    Adds language name to items in stream
    """
    for ex in stream:
        yield ex | {"lang_name": get_lang_name(ex["src_lang"])}


def add_history_text(stream: StreamType) -> Iterator[StreamType]:
    """
    Adds history text to items in stream with the history text corresponding
    to the item's term field. Note that the history_text field sets the text
    displayed within the history panel of the web app's dashboard.
    """
    for ex in stream:
        yield ex | {"history_text": ex["term"]}


def get_lang_annotators(ctrl: Controller, lang: str) -> list[str]:
    """
    Gathers the annotators for a specific language.
    """
    # Language-specific session prefix
    session_pfx = ctrl.get_session_name(f"{lang}_")
    # Get all known session names
    annotators = ctrl.session_ids
    return [a for a in annotators if a.startswith(session_pfx)]


def lang_task_router(ctrl: Controller, session_id: str, item: dict) -> list[str]:
    """
    Language-specific task routing. Currently, routes an example to all
    available annotators associated with its source language.
    """
    # TODO: May update to reduce annotator load when there are 3+ annotators
    # Get hash data for logging (should default to input)
    exclude_by_task = ctrl.exclude_by == "task"
    hash_attr = TASK_HASH_ATTR if exclude_by_task else INPUT_HASH_ATTR
    item_hash = get_task_hash(item) if exclude_by_task else get_input_hash(item)
    # Get source language of item
    src_lang = item["src_lang"]
    # Get annotator pool
    annotators = get_lang_annotators(ctrl, src_lang)
    log_router(hash_attr, item_hash, annotators)
    return annotators


@recipe(
    "concept-eval",
    dataset=Arg(help="Dataset to save answers to"),
    source=Arg(help="The source data as a JSONL file"),
    instruct=Arg(
        "--instruct",
        "-I",
        help="Optional intructions text or HTML file",
    ),
)
def concept_eval_recipe(
    dataset: str,
    source: str,
    instruct: PathInputType | None = None,
) -> RecipeSettingsType:
    # TODO: Consider adding an instruction page. See https://prodi.gy/docs/api-web-app#instructions
    log("RECIPE: Starting recipe concept-eval", locals())

    # Task elements most likely to be modified
    ## Question prompts for task
    questions = [
        "Q1. For the following translation, identify the translation(s) of the concept.\n"
        "Highlight all instances.",
        "Q2. Evaluate the machine translation of the concept.",
        "Q3. Notes / observations",
    ]

    q2_labels = [
        {"id": "correct", "text": "Correct translation"},
        {"id": "translated", "text": "Should not translate"},
        {"id": "missing", "text": "Omitted or missing"},
        {"id": "ils", "text": "Incorrect lexical selection"},
        {"id": "dit", "text": "Disambiguation issue in target"},
        {"id": "untranslated", "text": "Incorrectly left untranslated"},
        {"id": "other", "text": "Other error"},
    ]

    # Recipe organization
    ## Initial html template for starting text
    init_html_tmpl = "\n".join(
        [
            "<h2>Concept: {{term}}</h2>",
            "<p><b>Original {{lang_name}} Text</b>",
            "{{src_text}}",
            "<details>",
            "<summary><b>Professional English Translation</b></summary>{{ref_text}}",
            "</details>",
            f"<hr><b>{questions[0]}</b>",
        ]
    )

    q2_key = "\n".join(
        [
            "<details>",
            "<summary><b>Evaluation Typology</b></summary>"
            # Keep on same line as previous to reduce vertical whitespace
            "<p><b>Correct translation.</b> The MT model correctly translated the"
            " source term, either matching or approximating the professional"
            " translation.",
            "<p><b>Should not translate.</b> Though the MT model provided a"
            " translation of the source term, the source term should be left in its"
            " original language because of its untranslatability and the richness of"
            " its affordances.",
            "<p><b>Omitted or missing.</b> The source term is missing from the MT"
            " model translation.",
            "<p><b>Incorrect lexical selection.</b> The translation of the source"
            " term is an incorrect lexical choice (Haque et al. 2019).",
            "<p><b>Disambiguation error in target.</b> Although the MT system makes"
            " a potentially correct lexical choice for the source term, its"
            " translation-equivalent does not carry the meaning of the source term"
            " (Haque et al. 2019).",
            "<p><b>Incorrectly left untranslated.</b> The MT model did not translate"
            " the source term when it should have.",
            "<p><b>Other error.</b> Cases that do not fit into the above typology;"
            " provide context in the notes section below.",
            "</details>",
        ]
    )

    ## Arrangement of combined interfaces
    blocks = [
        {"view_id": "html", "html_template": init_html_tmpl},
        {"view_id": "ner_manual", "labels": ["CONCEPT"]},
        {"view_id": "html", "html": f"<hr><b>{questions[1]}</b>\n<p>{q2_key}"},
        {"view_id": "choice", "text": None, "options": q2_labels},
        {"view_id": "html", "html": f"<hr><b>{questions[2]}</b>"},
        {"view_id": "text_input", "field_rows": 3},
    ]
    ## Configuration
    config = {
        "buttons": ["accept", "undo"],  # remove ignore and reject buttons
        "show_flag": True,  # show flag button to mark weird machine translations
        "honor_token_whitespace": True,  # reflect whitespace accurately (e.g. in case of leading/trailing spaces)
        "blocks": blocks,
        "ner_manual_highlight_chars": True,
        "custom_theme": {"cardMaxWidth": "70%"},
        "allow_work_stealing": False,
        "show_stats": False,  # removing since accept is the only option
    }
    ### Add instructions if provided
    if instruct and pathlib.Path(instruct).is_file():
        config |= {"instructions": instruct}

    # Define custom recipe methods
    ## Get language example totals
    lang_example_counts = Counter(ex["src_lang"] for ex in get_stream(source))

    def progress(
        ctrl: Controller,
        session: Session,
        answers: list[dict],
        update_return_value: float | int | None,
    ) -> float:
        """
        Returns session-specific progress.
        """
        session_id = session.id

        # Determine the session language
        annot = session_id.split(ctrl.get_session_name(""), maxsplit=1)[1]
        session_lang = annot.split("_", maxsplit=1)[0]

        return session.total_annotated / lang_example_counts[session_lang]

    def set_stream_hashes(stream: StreamType) -> Iterator[StreamType]:
        """
        Set hashes for stream
        """
        for ex in stream:
            yield set_hashes(
                ex, input_keys=("tr_id"), task_keys=("questions", "spans", "options")
            )

    def validate_answer(eg) -> None:
        """
        Validates answers submitted in the UI. May raise validation errors.
        """
        q1_spans = eg.get("spans", [])
        q2_selected = eg.get("accept", [])
        q3_response = eg.get("user_input", "").strip()
        flagged = eg.get("flagged", False)

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
        # Validate Q3 answer
        if "other" in q2_selected and not q3_response:
            raise ValueError(
                "If the concept translation error is marked as other,"
                " then the nature of the error must be documented in Q3"
            )
        if flagged and not q3_response:
            raise ValueError(
                "If an annotation is flagged, then the issue(s) must be documented in Q3"
            )

    # Setup stream
    stream = get_stream(source)
    stream.apply(add_lang_name, stream)
    stream.apply(add_history_text, stream)
    stream.apply(add_tokens, stream)
    stream.apply(set_stream_hashes, stream)

    components = {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": config,
        "validate_answer": validate_answer,
        "task_router": lang_task_router,
        "progress": progress,
    }
    return components
