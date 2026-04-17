import marimo

__generated_with = "0.22.0"
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

    Set `DATA_DIR` to the phase-1 data directory (e.g. the project drive or TigerData mount).
    All data files are resolved relative to this path.
    """)
    return


@app.cell
def _(pathlib):
    # Set this to the phase-1 data directory on the project drive / TigerData mount.
    # Defaults to the local data directory for local development.
    DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "Phase 1"
    return (DATA_DIR,)


@app.cell
def _(DATA_DIR, pl):
    # Load full sentence translations and join with eval scores.
    # Uses the complete notion-sents corpus (all Notion concepts), not the annotation subset.
    # Note: madlad does not have eval scores.
    _sents_meta = pl.concat(
        [
            pl.read_ndjson(DATA_DIR / f"notion-sents/mt-sents-{m}.jsonl")
            for m in ["google_tllm", "hymt", "gemma", "madlad"]
        ]
    )
    _sents_scores = pl.concat(
        [
            pl.read_csv(DATA_DIR / f"notion-sents/eval-sents-{m}.csv")
            for m in ["google_tllm", "hymt", "gemma"]
        ]
    )
    sents_df = _sents_meta.join(_sents_scores, on="tr_id", how="left").with_columns(
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
    mo.md("## 1. Sentence translations")
    return


@app.cell
def _(pl, sents_df):
    sents_df.filter(pl.col("src_lang") != "en").select(
        ["pair_id", "model", "src_lang", "tr_lang", "chrf", "comet", "cometkiwi", "src_text", "ref_text", "tr_text"]
    )


@app.cell
def _(mo):
    mo.md("## 2. Sentence backtranslations")
    return


@app.cell
def _(pl, sents_df):
    sents_df.filter(pl.col("src_lang") == "en").select(
        ["pair_id", "model", "src_lang", "tr_lang", "chrf", "comet", "cometkiwi", "src_text", "ref_text", "tr_text"]
    )


@app.cell
def _(mo):
    mo.md("## 3. Paragraph translations")
    return


@app.cell
def _(pars_df, pl):
    pars_df.filter(pl.col("src_lang") != "en").select(
        ["pair_id", "model", "src_lang", "tr_lang", "chrf", "comet", "cometkiwi", "src_text", "ref_text", "tr_text"]
    )


@app.cell
def _(mo):
    mo.md("## 4. Paragraph backtranslations")
    return


@app.cell
def _(pars_df, pl):
    pars_df.filter(pl.col("src_lang") == "en").select(
        ["pair_id", "model", "src_lang", "tr_lang", "chrf", "comet", "cometkiwi", "src_text", "ref_text", "tr_text"]
    )


if __name__ == "__main__":
    app.run()
