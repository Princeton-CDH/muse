# Copyright Center for Digital Humanities, Princeton University 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Phase 1 Machine Translation Browser

    Browse Phase 1 machine translations side-by-side with source, reference, and evaluation
    scores (chrF, COMET, CometKiwi). Translations and backtranslations are shown in separate
    sections.
    """)
    return


@app.cell
def _():
    import pathlib

    import polars as pl

    return pathlib, pl


@app.cell
def _(mo):
    mo.md("""
    ## Configuration

    This notebook uses the phase-1 data directory as it is organized on TigerData.
    """)
    return


@app.cell
def _(pathlib):
    # Set this to your local copy of the project's TigerData directory
    tigerdata_filepath = "../data/tigerdata"
    # Set to phase-1 directory
    DATA_DIR = pathlib.Path(tigerdata_filepath) / "phase-1"
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR, pl):
    # Load full sentence translations and join with eval scores.
    # Uses the complete notion-sents corpus (all Notion concepts), not the annotation subset.
    _sents_meta = pl.concat(
        [
            pl.read_ndjson(DATA_DIR / f"notion-sents/mt-sents-{m}.jsonl")
            for m in ["google_tllm", "hymt", "gemma"]
        ]
    )
    _sents_scores = pl.concat(
        [
            pl.read_csv(DATA_DIR / f"notion-sents/eval-sents-{m}.csv")
            for m in ["google_tllm", "hymt", "gemma"]
        ]
    )
    sents_df = _sents_meta.join(
        _sents_scores, on="tr_id", how="left"
    ).with_columns(
        pl.col("chrf").round(3),
        pl.col("comet").round(3),
        pl.col("cometkiwi").round(3),
    )
    return (sents_df,)


@app.cell
def _(DATA_DIR, pl):
    # Load paragraph translations and join with eval scores.
    _pars_meta = pl.concat(
        [
            pl.read_ndjson(DATA_DIR / f"mto-pars/mt-pars-{m}.jsonl")
            for m in ["google_tllm", "hymt", "gemma"]
        ]
    )
    _pars_scores = pl.concat(
        [
            pl.read_csv(DATA_DIR / f"mto-pars/eval-pars-{m}.csv")
            for m in ["google_tllm", "hymt", "gemma"]
        ]
    )
    pars_df = _pars_meta.join(_pars_scores, on="tr_id", how="left").with_columns(
        pl.col("chrf").round(3),
        pl.col("comet").round(3),
        pl.col("cometkiwi").round(3),
    )
    return (pars_df,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Sentence translations
    """)
    return


@app.cell
def _(pl, sents_df):
    sents_df.filter(pl.col("src_lang") != "en").select(
        [
            "pair_id",
            "model",
            "src_lang",
            "tr_lang",
            "chrf",
            "comet",
            "cometkiwi",
            "src_text",
            "ref_text",
            "tr_text",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Sentence backtranslations
    """)
    return


@app.cell
def _(pl, sents_df):
    sents_df.filter(pl.col("src_lang") == "en").select(
        [
            "pair_id",
            "model",
            "src_lang",
            "tr_lang",
            "chrf",
            "comet",
            "cometkiwi",
            "src_text",
            "ref_text",
            "tr_text",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Paragraph translations
    """)
    return


@app.cell
def _(pars_df, pl):
    pars_df.filter(pl.col("src_lang") != "en").select(
        [
            "pair_id",
            "model",
            "src_lang",
            "tr_lang",
            "chrf",
            "comet",
            "cometkiwi",
            "src_text",
            "ref_text",
            "tr_text",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Paragraph backtranslations
    """)
    return


@app.cell
def _(pars_df, pl):
    pars_df.filter(pl.col("src_lang") == "en").select(
        [
            "pair_id",
            "model",
            "src_lang",
            "tr_lang",
            "chrf",
            "comet",
            "cometkiwi",
            "src_text",
            "ref_text",
            "tr_text",
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (c)2026 Trustees of Princeton University. Permission granted for non-commercial distribution online under the [Apache 2.0 License](https://github.com/Princeton-CDH/muse/blob/main/LICENSE).
    """)
    return


if __name__ == "__main__":
    app.run()
