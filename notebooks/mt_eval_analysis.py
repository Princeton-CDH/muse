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
    # MT Evaluation Score Analysis

    chrF, COMET, CometKiwi across three models (google_tllm, hymt, gemma)
    and two corpora (sentences: 427 pairs, paragraphs: 526 pairs).
    """)
    return


@app.cell
def _():
    import pathlib

    import altair as alt
    import pandas as pd
    import polars as pl
    from scipy import stats

    return alt, pathlib, pd, pl, stats


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

    sents_df.head(3)
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

    pars_df.head(3)
    return (pars_df,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Score distributions by model
    """)
    return


@app.cell
def _(alt, mo, pars_df, sents_df):
    def score_dist_chart(df, title: str):
        melted = (
            df.select(["model", "chrf", "comet", "cometkiwi"])
            .unpivot(index="model", variable_name="metric", value_name="score")
            .to_pandas()
        )
        return (
            alt.Chart(melted, title=title)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("score:Q", title="Score", scale=alt.Scale(zero=False)),
                color="model:N",
                column=alt.Column("metric:N", title="Metric"),
            )
            .properties(width=160, height=220)
        )


    mo.vstack(
        [
            mo.md("**Sentences**"),
            mo.ui.altair_chart(score_dist_chart(sents_df, "Sentences")),
            mo.md("**Paragraphs**"),
            mo.ui.altair_chart(score_dist_chart(pars_df, "Paragraphs")),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Model ranking
    """)
    return


@app.cell
def _(alt, mo, pars_df, pl, sents_df):
    def model_ranking_chart(df, title: str):
        melted = (
            df.group_by("model")
            .agg(
                pl.col("chrf").mean().alias("chrf"),
                pl.col("comet").mean().alias("comet"),
                pl.col("cometkiwi").mean().alias("cometkiwi"),
            )
            .unpivot(index="model", variable_name="metric", value_name="score")
            .to_pandas()
        )
        return (
            alt.Chart(melted, title=title)
            .mark_bar()
            .encode(
                x=alt.X(
                    "score:Q", scale=alt.Scale(zero=False), title="Mean score"
                ),
                y=alt.Y("model:N", sort="-x", title=None),
                color="model:N",
                row=alt.Row("metric:N", title="Metric"),
            )
            .properties(width=300, height=70)
        )


    mo.vstack(
        [
            mo.md("**Sentences**"),
            mo.ui.altair_chart(model_ranking_chart(sents_df, "Sentences")),
            mo.md("**Paragraphs**"),
            mo.ui.altair_chart(model_ranking_chart(pars_df, "Paragraphs")),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Metric agreement

    Key finding: chrF vs CometKiwi correlation collapses on paragraphs (r ≈ −0.05), suggesting reference translations are paraphrastic rather than literal. Low paragraph COMET/CometKiwi correlation (r ≈ 0.28) means they diverge on longer texts — CometKiwi is arguably more honest since it doesn't penalise paraphrastic output.
    """)
    return


@app.cell
def _(mo, pars_df, sents_df, stats):
    def correlation_table(df, label: str) -> str:
        pairs = [("chrf", "comet"), ("chrf", "cometkiwi"), ("comet", "cometkiwi")]
        rows = []
        for a, b in pairs:
            x = df[a].to_numpy()
            y = df[b].to_numpy()
            r, p = stats.pearsonr(x, y)
            rho, p_rho = stats.spearmanr(x, y)
            rows.append(
                f"| {a} vs {b} | {r:.3f} | {p:.2e} | {rho:.3f} | {p_rho:.2e} |"
            )
        header = (
            f"**{label}**\n\n"
            "| Pair | Pearson r | p | Spearman ρ | p |\n"
            "|------|-----------|---|------------|---|"
        )
        return header + "\n" + "\n".join(rows)


    mo.vstack(
        [
            mo.md(correlation_table(sents_df, "Sentences")),
            mo.md(correlation_table(pars_df, "Paragraphs")),
        ]
    )
    return


@app.cell
def _(alt, mo, pars_df, sents_df):
    def scatter_matrix(df, title: str):
        pdf = df.select(["model", "chrf", "comet", "cometkiwi"]).to_pandas()
        return (
            alt.Chart(pdf)
            .mark_point(opacity=0.4, size=20)
            .encode(
                alt.X(
                    alt.repeat("column"),
                    type="quantitative",
                    scale=alt.Scale(zero=False),
                ),
                alt.Y(
                    alt.repeat("row"),
                    type="quantitative",
                    scale=alt.Scale(zero=False),
                ),
                color="model:N",
            )
            .repeat(
                row=["chrf", "comet", "cometkiwi"],
                column=["chrf", "comet", "cometkiwi"],
            )
        )


    mo.vstack(
        [
            mo.md("**Sentences**"),
            mo.ui.altair_chart(scatter_matrix(sents_df, "Sentences")),
            mo.md("**Paragraphs**"),
            mo.ui.altair_chart(scatter_matrix(pars_df, "Paragraphs")),
        ]
    )
    return


@app.cell
def _(alt, mo, pars_df, sents_df):
    def kiwi_vs_comet(df, title: str):
        pdf = df.select(["model", "comet", "cometkiwi"]).to_pandas()
        base = alt.Chart(pdf, title=title).properties(width=300, height=260)
        points = base.mark_point(opacity=0.5, size=25).encode(
            x=alt.X("comet:Q", scale=alt.Scale(zero=False), title="COMET"),
            y=alt.Y("cometkiwi:Q", scale=alt.Scale(zero=False), title="CometKiwi"),
            color="model:N",
        )
        regression = (
            base.transform_regression("comet", "cometkiwi")
            .mark_line(color="gray", strokeDash=[4, 2])
            .encode(
                x=alt.X("comet:Q"),
                y=alt.Y("cometkiwi:Q"),
            )
        )
        return alt.layer(points, regression)


    mo.vstack(
        [
            mo.md("**Sentences — COMET vs CometKiwi**"),
            mo.ui.altair_chart(kiwi_vs_comet(sents_df, "Sentences")),
            mo.md("**Paragraphs — COMET vs CometKiwi**"),
            mo.ui.altair_chart(kiwi_vs_comet(pars_df, "Paragraphs")),
        ]
    )
    return


@app.cell
def _(alt, mo, pars_df, pd, pl, sents_df, stats):
    def agreement_heatmap(df, title: str):
        pairs = [("chrf", "comet"), ("chrf", "cometkiwi"), ("comet", "cometkiwi")]
        rows = [
            {
                "model": model,
                "pair": f"{a} vs {b}",
                "rho": round(
                    stats.spearmanr(
                        df.filter(pl.col("model") == model)[a].to_numpy(),
                        df.filter(pl.col("model") == model)[b].to_numpy(),
                    )[0],
                    3,
                ),
            }
            for model in sorted(df["model"].unique().to_list())
            for a, b in pairs
        ]
        pdf = pd.DataFrame(rows)
        return (
            alt.Chart(pdf, title=title)
            .mark_rect()
            .encode(
                x=alt.X("pair:N", title=None),
                y=alt.Y("model:N", title=None),
                color=alt.Color(
                    "rho:Q",
                    scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                    title="Spearman ρ",
                ),
                tooltip=["model", "pair", "rho"],
            )
            .properties(width=280, height=100)
        )


    mo.vstack(
        [
            mo.md("**Sentences — instance-level metric agreement (Spearman ρ)**"),
            mo.ui.altair_chart(agreement_heatmap(sents_df, "Sentences")),
            mo.md("**Paragraphs — instance-level metric agreement (Spearman ρ)**"),
            mo.ui.altair_chart(agreement_heatmap(pars_df, "Paragraphs")),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Language breakdown

    If only chrF drops for CJK languages, it's a script artefact. If COMET/CometKiwi also drop, it's a real quality signal.
    """)
    return


@app.cell
def _(alt, mo, pars_df, pl, sents_df):
    def lang_dir_chart(df, title: str):
        melted = (
            df.with_columns(
                (pl.col("src_lang") + "→" + pl.col("tr_lang")).alias("direction")
            )
            .select(["direction", "chrf", "comet", "cometkiwi"])
            .unpivot(index="direction", variable_name="metric", value_name="score")
            .to_pandas()
        )
        return (
            alt.Chart(melted, title=title)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("direction:N", title="Direction"),
                y=alt.Y("score:Q", scale=alt.Scale(zero=False)),
                color="direction:N",
                column=alt.Column("metric:N"),
            )
            .properties(width=160, height=220)
        )


    mo.vstack(
        [
            mo.md("**Sentences — score by direction**"),
            mo.ui.altair_chart(lang_dir_chart(sents_df, "Sentences")),
            mo.md("**Paragraphs — score by direction**"),
            mo.ui.altair_chart(lang_dir_chart(pars_df, "Paragraphs")),
        ]
    )
    return


@app.cell
def _(alt, mo, pars_df, pl, sents_df):
    def lang_breakdown_chart(df, title: str) -> alt.Chart:
        melted = (
            df.with_columns(
                (pl.col("src_lang") + "→" + pl.col("tr_lang")).alias("direction")
            )
            .group_by(["direction", "model"])
            .agg(
                pl.col("chrf").mean().round(4).alias("chrf"),
                pl.col("comet").mean().round(4).alias("comet"),
                pl.col("cometkiwi").mean().round(4).alias("cometkiwi"),
            )
            .unpivot(
                index=["direction", "model"],
                variable_name="metric",
                value_name="score",
            )
            .to_pandas()
        )
        return (
            alt.Chart(melted, title=title)
            .mark_bar()
            .encode(
                x=alt.X("model:N", title=None),
                y=alt.Y("score:Q", scale=alt.Scale(zero=False)),
                color="model:N",
                column=alt.Column("direction:N", title="Direction"),
                row=alt.Row("metric:N", title="Metric"),
            )
            .properties(width=80, height=100)
        )


    mo.vstack(
        [
            mo.md("**Sentences — mean score by direction and model**"),
            mo.ui.altair_chart(lang_breakdown_chart(sents_df, "Sentences")),
            mo.md("**Paragraphs — mean score by direction and model**"),
            mo.ui.altair_chart(lang_breakdown_chart(pars_df, "Paragraphs")),
        ]
    )
    return


@app.cell
def _(alt, mo, pars_df, pd, pl, sents_df):
    def direction_chart(df, title: str):
        rows = [
            {
                "direction": direction,
                "metric": metric,
                "score": group[metric].mean(),
            }
            for direction, group in [
                ("src→en", df.filter(pl.col("tr_lang") == "en")),
                ("en→src", df.filter(pl.col("src_lang") == "en")),
            ]
            for metric in ["chrf", "comet", "cometkiwi"]
        ]
        pdf = pd.DataFrame(rows)
        return (
            alt.Chart(pdf, title=title)
            .mark_bar()
            .encode(
                x=alt.X("direction:N", title=None),
                y=alt.Y(
                    "score:Q", scale=alt.Scale(zero=False), title="Mean score"
                ),
                color="direction:N",
                column=alt.Column("metric:N", title="Metric"),
            )
            .properties(width=120, height=180)
        )


    mo.vstack(
        [
            mo.md("**Sentences — src→en vs en→src**"),
            mo.ui.altair_chart(direction_chart(sents_df, "Sentences")),
            mo.md("**Paragraphs — src→en vs en→src**"),
            mo.ui.altair_chart(direction_chart(pars_df, "Paragraphs")),
        ]
    )
    return


@app.cell
def _(DATA_DIR, pars_df, pl, sents_df):
    # Load full text corpora and join with scores for qualitative browser
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
        .with_columns(
            (pl.col("cometkiwi") - pl.col("chrf")).alias("kiwi_minus_chrf")
        )
    )
    return (browser_data,)


@app.cell
def _(browser_data, mo, pl):
    _corpora = ["sentences", "paragraphs"]
    _models = sorted(browser_data["model"].unique().to_list())
    _directions = sorted(
        browser_data.with_columns(
            (pl.col("src_lang") + "→" + pl.col("tr_lang")).alias("dir")
        )["dir"]
        .unique()
        .to_list()
    )
    _terms = ["all", *sorted(browser_data["term"].drop_nulls().unique().to_list())]
    _sort_opts = [
        "kiwi_minus_chrf ↓",  # high CometKiwi but low chrF — paraphrase cases
        "kiwi_minus_chrf ↑",
        "cometkiwi ↓",
        "cometkiwi ↑",
        "comet ↓",
        "comet ↑",
        "chrf ↓",
        "chrf ↑",
        "pair_id",
    ]

    corpus_sel = mo.ui.dropdown(_corpora, value="sentences", label="Corpus")
    model_sel = mo.ui.dropdown(["all", *_models], value="all", label="Model")
    dir_sel = mo.ui.dropdown(["all", *_directions], value="all", label="Direction")
    term_sel = mo.ui.dropdown(_terms, value="all", label="Term")
    sort_sel = mo.ui.dropdown(
        _sort_opts, value="kiwi_minus_chrf ↓", label="Sort by"
    )
    n_sel = mo.ui.slider(5, 50, step=5, value=10, label="Show N")

    mo.vstack(
        [
            mo.md(
                "## 5. Qualitative translation browser\n\nRead src / ref / translation side-by-side with scores. Sort by `kiwi_minus_chrf ↓` to surface cases where CometKiwi rates the translation highly but chrF doesn't — these are the paraphrase cases."
            ),
            mo.hstack(
                [corpus_sel, model_sel, dir_sel, term_sel, sort_sel, n_sel],
                justify="start",
            ),
        ]
    )
    return corpus_sel, dir_sel, model_sel, n_sel, sort_sel, term_sel


@app.cell
def _(
    browser_data,
    corpus_sel,
    dir_sel,
    mo,
    model_sel,
    n_sel,
    pl,
    sort_sel,
    term_sel,
):
    _df = browser_data.filter(pl.col("corpus") == corpus_sel.value)

    if model_sel.value != "all":
        _df = _df.filter(pl.col("model") == model_sel.value)

    if dir_sel.value != "all":
        _src, _tgt = dir_sel.value.split("→")
        _df = _df.filter(
            (pl.col("src_lang") == _src) & (pl.col("tr_lang") == _tgt)
        )

    if term_sel.value != "all":
        _df = _df.filter(pl.col("term") == term_sel.value)

    _sort_col, _sort_dir = (
        sort_sel.value.rsplit(" ", 1)
        if " " in sort_sel.value
        else (sort_sel.value, "↑")
    )
    _df = _df.sort(_sort_col, descending=(_sort_dir == "↓"))

    _display = (
        _df.head(n_sel.value)
        .select(
            [
                "pair_id",
                "model",
                "src_lang",
                "tr_lang",
                "chrf",
                "comet",
                "cometkiwi",
                "kiwi_minus_chrf",
                "term",
                "src_text",
                "ref_text",
                "tr_text",
            ]
        )
        .with_columns(
            [
                pl.col("chrf").round(3),
                pl.col("comet").round(3),
                pl.col("cometkiwi").round(3),
                pl.col("kiwi_minus_chrf").round(3),
            ]
        )
    )

    mo.plain(_display)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    **Models**
    google_tllm is the clear winner on all metrics across both corpora, with statistically
    significant margins. gemma and hymt are statistically indistinguishable from each other.

    **Metrics**
    - **COMET** — It's trained on human judgements, robust to paraphrase,
      and shows consistent model separation. The compressed range (0.77–0.81) is normal.
    - **CometKiwi** — worth reporting alongside COMET, especially for paragraphs where
      it diverges from COMET (r ≈ 0.28). Being reference-free makes it more honest when
      the reference translations are paraphrastic, which is likely here.
    - **chrF** — use with caution. It's the most discriminating metric numerically (high CV),
      but for non-literal, domain-specific translations it mostly measures surface similarity
      to the reference rather than actual quality. The near-zero correlation with CometKiwi
      on paragraphs (r ≈ −0.05) is a red flag.
    """)
    return


if __name__ == "__main__":
    app.run()
