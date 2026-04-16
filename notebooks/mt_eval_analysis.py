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

    **Metrics:** chrF measures character n-gram overlap with the reference (unreliable for
    cross-script directions such as `en→zh`). COMET and CometKiwi are neural metrics trained
    on human judgements; CometKiwi is reference-free and does not penalise paraphrastic output.
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
    # Load sentence eval CSVs and join with metadata from notion-concept-tasks.jsonl
    sents_meta = pl.read_ndjson(
        DATA_DIR / "prodigy/notion-concept/notion-concept-tasks.jsonl"
    ).select(["tr_id", "model", "src_lang", "tr_lang", "pair_id"])

    def load_sents_csv(model: str) -> pl.DataFrame:
        return pl.read_csv(DATA_DIR / f"notion-sents/eval-sents-{model}.csv").join(
            sents_meta.filter(pl.col("model") == model), on="tr_id"
        )

    sents_df = pl.concat(
        [
            load_sents_csv("google_tllm"),
            load_sents_csv("hymt"),
            load_sents_csv("gemma"),
        ]
    )
    return (sents_df,)


@app.cell
def _(DATA_DIR, pl):
    # Load paragraph eval CSVs and join with metadata from mt-pars-*.jsonl
    def load_pars_csv(model: str) -> pl.DataFrame:
        meta = pl.read_ndjson(DATA_DIR / f"mto-pars/mt-pars-{model}.jsonl").select(
            ["tr_id", "model", "src_lang", "tr_lang", "pair_id"]
        )
        return pl.read_csv(DATA_DIR / f"mto-pars/eval-pars-{model}.csv").join(
            meta, on="tr_id"
        )

    pars_df = pl.concat(
        [
            load_pars_csv("google_tllm"),
            load_pars_csv("hymt"),
            load_pars_csv("gemma"),
        ]
    )
    return (pars_df,)


@app.cell
def _(DATA_DIR, pars_df, pl, sents_df):
    # Load full text corpora and join with scores
    _sents_full = (
        pl.read_ndjson(
            DATA_DIR / "prodigy/notion-concept/notion-concept-tasks.jsonl"
        )
        .rename({"text": "tr_text"})
        .select(
            [
                "tr_id",
                "pair_id",
                "model",
                "src_lang",
                "tr_lang",
                "src_text",
                "ref_text",
                "tr_text",
                "term",
            ]
        )
        .with_columns(pl.lit("sentences").alias("corpus"))
    )
    _pars_full = pl.concat(
        [
            pl.read_ndjson(DATA_DIR / f"mto-pars/mt-pars-{m}.jsonl").select(
                [
                    "tr_id",
                    "pair_id",
                    "model",
                    "src_lang",
                    "tr_lang",
                    "src_text",
                    "ref_text",
                    "tr_text",
                ]
            )
            for m in ["google_tllm", "hymt", "gemma"]
        ]
    ).with_columns(
        pl.lit(None).cast(pl.String).alias("term"),
        pl.lit("paragraphs").alias("corpus"),
    )

    _scores = pl.concat(
        [
            sents_df.select(["tr_id", "chrf", "comet", "cometkiwi"]),
            pars_df.select(["tr_id", "chrf", "comet", "cometkiwi"]),
        ]
    )

    browser_data = (
        pl.concat([_sents_full, _pars_full])
        .join(_scores, on="tr_id", how="left")
    )
    return (browser_data,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Translations
    """)
    return


@app.cell
def _(browser_data, mo, pl):
    # Sentences are all forward translations (src != en).
    # Paragraphs: forward translations have src_lang != en.
    _fwd = browser_data.filter(pl.col("src_lang") != "en")
    _corpora = sorted(_fwd["corpus"].unique().to_list())
    corpus_sel_fwd = mo.ui.dropdown(_corpora, value="sentences", label="Corpus")

    _models = sorted(_fwd["model"].unique().to_list())
    model_sel_fwd = mo.ui.dropdown(["all", *_models], value="all", label="Model")

    _all_dirs = sorted(
        _fwd.with_columns((pl.col("src_lang") + "→" + pl.col("tr_lang")).alias("dir"))["dir"]
        .unique().to_list()
    )
    dir_sel_fwd = mo.ui.dropdown(["all", *_all_dirs], value="all", label="Direction")

    _sort_opts = ["cometkiwi ↓", "cometkiwi ↑", "comet ↓", "comet ↑", "chrf ↓", "chrf ↑", "pair_id"]
    sort_sel_fwd = mo.ui.dropdown(_sort_opts, value="cometkiwi ↓", label="Sort by")
    n_sel_fwd = mo.ui.slider(5, 50, step=5, value=10, label="Show N")

    mo.hstack([corpus_sel_fwd, model_sel_fwd, dir_sel_fwd, sort_sel_fwd, n_sel_fwd], justify="start")
    return corpus_sel_fwd, dir_sel_fwd, model_sel_fwd, n_sel_fwd, sort_sel_fwd


@app.cell
def _(browser_data, corpus_sel_fwd, dir_sel_fwd, model_sel_fwd, mo, n_sel_fwd, pl, sort_sel_fwd):
    _df = browser_data.filter(
        (pl.col("src_lang") != "en") & (pl.col("corpus") == corpus_sel_fwd.value)
    )
    if model_sel_fwd.value != "all":
        _df = _df.filter(pl.col("model") == model_sel_fwd.value)
    if dir_sel_fwd.value != "all":
        _s, _t = dir_sel_fwd.value.split("→")
        _df = _df.filter((pl.col("src_lang") == _s) & (pl.col("tr_lang") == _t))
    _sc, _sd = (
        sort_sel_fwd.value.rsplit(" ", 1)
        if " " in sort_sel_fwd.value
        else (sort_sel_fwd.value, "↑")
    )
    _df = _df.sort(_sc, descending=(_sd == "↓"))
    mo.plain(
        _df.head(n_sel_fwd.value)
        .select(["pair_id", "model", "src_lang", "tr_lang", "chrf", "comet", "cometkiwi", "src_text", "ref_text", "tr_text"])
        .with_columns(pl.col("chrf").round(3), pl.col("comet").round(3), pl.col("cometkiwi").round(3))
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Backtranslations
    """)
    return


@app.cell
def _(browser_data, mo, pl):
    # Backtranslations: src_lang == en (paragraphs only — sentences have no backtranslations).
    _back = browser_data.filter(pl.col("src_lang") == "en")

    _models_b = sorted(_back["model"].unique().to_list())
    model_sel_back = mo.ui.dropdown(["all", *_models_b], value="all", label="Model")

    _all_dirs_b = sorted(
        _back.with_columns((pl.col("src_lang") + "→" + pl.col("tr_lang")).alias("dir"))["dir"]
        .unique().to_list()
    )
    dir_sel_back = mo.ui.dropdown(["all", *_all_dirs_b], value="all", label="Direction")

    _sort_opts_b = ["cometkiwi ↓", "cometkiwi ↑", "comet ↓", "comet ↑", "chrf ↓", "chrf ↑", "pair_id"]
    sort_sel_back = mo.ui.dropdown(_sort_opts_b, value="cometkiwi ↓", label="Sort by")
    n_sel_back = mo.ui.slider(5, 50, step=5, value=10, label="Show N")

    mo.hstack([model_sel_back, dir_sel_back, sort_sel_back, n_sel_back], justify="start")
    return dir_sel_back, model_sel_back, n_sel_back, sort_sel_back


@app.cell
def _(browser_data, dir_sel_back, model_sel_back, mo, n_sel_back, pl, sort_sel_back):
    _df = browser_data.filter(pl.col("src_lang") == "en")
    if model_sel_back.value != "all":
        _df = _df.filter(pl.col("model") == model_sel_back.value)
    if dir_sel_back.value != "all":
        _s, _t = dir_sel_back.value.split("→")
        _df = _df.filter((pl.col("src_lang") == _s) & (pl.col("tr_lang") == _t))
    _sc, _sd = (
        sort_sel_back.value.rsplit(" ", 1)
        if " " in sort_sel_back.value
        else (sort_sel_back.value, "↑")
    )
    _df = _df.sort(_sc, descending=(_sd == "↓"))
    mo.plain(
        _df.head(n_sel_back.value)
        .select(["pair_id", "model", "src_lang", "tr_lang", "chrf", "comet", "cometkiwi", "src_text", "ref_text", "tr_text"])
        .with_columns(pl.col("chrf").round(3), pl.col("comet").round(3), pl.col("cometkiwi").round(3))
    )
    return


if __name__ == "__main__":
    app.run()
