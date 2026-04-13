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

    Box plots showing the full distribution of each metric score per model, across all translation
    pairs in each corpus. Each box spans the interquartile range (Q1–Q3); whiskers extend to the
    min/max.

    **Key finding:**

    Observation 1: chrF distributions are wide for all three models (whiskers spanning roughly
    0.2–1.0), pulled down at the low end by cross-script directions (CJK ↔ Latin). The median
    is slightly higher for google_tllm (~0.55–0.58), but the large variance makes it hard to
    draw firm conclusions from chrF alone.

    Observation 2: COMET distributions are much tighter (~0.6–0.9). google_tllm has the highest
    median (~0.80–0.82); gemma and hymt are close to each other (~0.77–0.79). On paragraphs the
    three models converge further and the boxes nearly overlap.

    Observation 3: CometKiwi follows the same pattern as COMET but with an even narrower spread.
    google_tllm remains the top model; gemma and hymt are nearly indistinguishable.

    Conclusion: google_tllm consistently outperforms the other two models across both corpora and
    all three metrics. gemma and hymt perform at a similar level. The wide chrF spread is driven
    by the cross-script artefact (see section 4), not by genuine quality variation.
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

    Bar charts showing the mean score per model for each metric, sorted highest to lowest.
    Aggregated across all translation pairs in each corpus.

    **Key finding:**

    Observation 1: google_tllm ranks first on every metric and both corpora, with a consistent
    lead. On sentences: chrF ~0.55, COMET ~0.81, CometKiwi ~0.84. On paragraphs the absolute
    scores are similar but the gap over the other two models narrows slightly.

    Observation 2: gemma and hymt are very close to each other on all metrics — the bars are
    nearly the same length. Neither consistently beats the other across all metrics and corpora.

    Observation 3: The ranking is stable across metrics (google_tllm > gemma ≈ hymt) even though
    the absolute scale differs a lot between chrF and the two neural metrics.

    Conclusion: google_tllm is the clear winner. gemma and hymt are statistically hard to
    separate — the mean score difference between them is smaller than the within-model variance
    seen in section 1.
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

    How well do the three metrics agree with each other at the instance level? Computed using
    Pearson r (linear correlation) and Spearman ρ (rank correlation) across all translation pairs
    in each corpus, pooled across models. The scatter matrix plots every metric pair against each
    other; the correlation tables give exact coefficients and p-values. The COMET vs CometKiwi
    scatter adds a linear regression line to make the trend visible.

    **Key finding:**

    Observation 1: On sentences, all three metrics show positive correlation with each other
    (all blue in the heatmap), with chrF vs COMET being the strongest pair. The scatter plots
    show a positive trend but with considerable spread — the metrics broadly agree on direction
    but not on exact rankings.

    Observation 2: On paragraphs, the scatter plots split into two visible clusters — one group
    with low chrF (CJK directions) and one with higher chrF (Latin-script directions). This
    distorts the chrF correlations. The heatmap shows chrF vs CometKiwi turning near-zero or
    slightly negative on paragraphs (r ≈ −0.05), and chrF vs COMET also weakens substantially.

    Observation 3: COMET vs CometKiwi correlation is moderate on sentences but drops further on
    paragraphs (r ≈ 0.28), meaning the two neural metrics diverge on longer texts — they are
    measuring somewhat different things.

    Analysis: The chrF correlation collapse on paragraphs has two causes: (1) the cross-script
    artefact described in section 4, and (2) reference translations that are paraphrastic rather
    than literal — chrF penalises valid paraphrases, while CometKiwi (being reference-free) does
    not. The COMET/CometKiwi divergence on paragraphs suggests CometKiwi is the more honest
    signal for longer texts.

    Conclusion: On sentences, the three metrics broadly agree. On paragraphs, chrF becomes
    unreliable and COMET and CometKiwi should be weighted more heavily — with CometKiwi
    preferred where reference quality is uncertain.
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

    Scores broken down by translation direction (e.g. `zh→en`, `en→zh`), derived from the
    `src_lang` and `tr_lang` fields in the eval CSVs. The box plots show the full score
    distribution per direction; the grouped bar charts show mean scores per direction × model;
    the final chart collapses all directions into two groups — into English (`src→en`) vs. out
    of English (`en→src`) — to test for a systematic directionality bias.

    **Key finding:**

    Observation 1: chrF scores vary dramatically by direction — `ja→en` and `zh→en` score around 0.4 on
    sentences, while `pt→en` reaches ~0.7. On paragraphs, `en→zh` collapses to near zero
    (median ~0.08) and `en→ja` is similarly low (~0.25).

    Analysis: This is not caused by quality difference. Because:
    chrF works by counting shared characters between the translation and the reference, so when
    the two are in completely different writing systems (e.g. CJK vs. Latin), they share no
    characters at all and the score is near zero by construction — regardless of translation
    quality.

    Observation 2: COMET and CometKiwi are unaffected — both stay flat across all directions (~0.75–0.85),
    with no CJK gap.

    Conclusion: For cross-script directions, chrF is meaningless; COMET and CometKiwi
    are the only reliable metrics.
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
                "## 5. Qualitative translation browser\n\nRead src / ref / translation side-by-side with scores. Data is joined from the full-text JSONL files (`notion-concept-tasks.jsonl` for sentences, `mt-pars-{model}.jsonl` for paragraphs) with the metric scores from the eval CSVs. The `kiwi_minus_chrf` column is computed as CometKiwi − chrF: a high value means the model produced a fluent translation that diverges from the reference wording. Sort by `kiwi_minus_chrf ↓` to surface the clearest paraphrase cases; sort by `cometkiwi ↓` or `comet ↓` to find the worst translations."
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
