"""Regressions and tables: H1 (explanatory), H2 (existence proof), H3 (capability),
H4 (counter-cost).

Everything is written to ``results/tables/`` as both CSV (for the paper's pipeline)
and Markdown (for reading in the terminal / pasting into the draft).  Every model's
full statsmodels summary also goes to the log file, so the numbers in the paper are
traceable to a run id.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .data_share import share_features
from .languages import BY_CODE, resolve
from .logging_utils import get_logger
from .state import read_json, read_jsonl

log = get_logger("analysis")

BYTE_TOKENIZERS = {"byt5", "utf8"}


# ---------------------------------------------------------------------------
# locating stage outputs across runs
# ---------------------------------------------------------------------------
def find_stage_dir(stage: str, run_id: str | None = None) -> Path | None:
    """Directory of ``stage`` in ``run_id``, else the most recent run that has it."""
    if run_id:
        d = paths.RUNS_DIR / run_id / stage
        if (d / "records.jsonl").is_file():
            return d
        log.warning("run %s has no %s records", run_id, stage)
    cands = sorted(
        (p.parent for p in paths.RUNS_DIR.glob(f"*/{stage}/records.jsonl")),
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        return None
    chosen = cands[-1]
    log.info("using %s from run %s", stage, chosen.parent.name)
    return chosen


def load_part_c_pooled(run_id: str | None = None,
                       pool: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Part C results across every run that has them.

    H3 is the weakest-powered analysis in the paper: within one model, fertility and
    boundary-F1 vary only across languages, so a single run gives ~15 independent
    points for two collinear predictors.  Evaluating a second model doubles the cells
    and adds tokenizer-level variation, so results are pooled across runs by default
    and the model enters the regression as a fixed effect.
    """
    dirs: list[Path] = []
    if run_id:
        d = paths.RUNS_DIR / run_id / "part_c"
        if (d / "records.jsonl").is_file():
            dirs.append(d)
    if pool or not dirs:
        for p in sorted(paths.RUNS_DIR.glob("*/part_c/records.jsonl"),
                        key=lambda p: p.stat().st_mtime):
            if p.parent not in dirs:
                dirs.append(p.parent)
    frames, used = [], []
    for d in dirs:
        recs = read_jsonl(d / "records.jsonl")
        if not recs:
            continue
        cfg = read_json(d / "config.json", {}) or {}
        f = pd.DataFrame(recs).drop(columns=["_unit", "_ts"], errors="ignore")
        f["run_id"] = d.parent.name
        if "model" not in f.columns:
            f["model"] = cfg.get("model", "unknown")
        f["model_tokenizer_key"] = f.get(
            "model_tokenizer_key", pd.Series([cfg.get("model_tokenizer_key")] * len(f)))
        frames.append(f)
        used.append(f"{d.parent.name}:{cfg.get('model', '?')}")
    if not frames:
        return pd.DataFrame(), []
    df = pd.concat(frames, ignore_index=True)
    df = df[df.get("n", 0) > 0]
    df = df.drop_duplicates(subset=["model", "task", "language", "condition"],
                            keep="last")
    log.info("Part C pooled from %d run(s): %s  -> %d cells, %d model(s)",
             len(used), "; ".join(used), len(df), df.model.nunique())
    return df, used


def load_stage(stage: str, run_id: str | None = None) -> pd.DataFrame:
    d = find_stage_dir(stage, run_id)
    if d is None:
        log.warning("no %s results found", stage)
        return pd.DataFrame()
    recs = read_jsonl(d / "records.jsonl")
    df = pd.DataFrame(recs).drop(columns=["_unit", "_ts"], errors="ignore")
    # Part A's aggregated CSV carries the English-relative premium columns, which are
    # computed from the per-sentence parquets rather than stored per record.
    audit_csv = d / "indic_audit.csv"
    if stage == "part_a" and audit_csv.is_file() and not df.empty:
        audit = pd.read_csv(audit_csv)
        extra = [c for c in audit.columns
                 if c not in df.columns or c in ("tokenizer", "language")]
        df = df.merge(audit[extra], on=["tokenizer", "language"], how="left")
    keys = {"part_a": ["tokenizer", "language"], "part_b": ["tokenizer", "language"],
            "part_c": ["task", "language", "condition"]}[stage]
    if not df.empty:
        df = df.drop_duplicates(subset=keys, keep="last")
    df.attrs["run_id"] = d.parent.name
    return df


# ---------------------------------------------------------------------------
# merged dataset
# ---------------------------------------------------------------------------
def build_merged(run_id: str | None = None) -> pd.DataFrame:
    a = load_stage("part_a", run_id)
    b = load_stage("part_b", run_id)
    if a.empty:
        raise SystemExit("Part A results are required for analysis — run `morphtok part-a`")
    keep_b = [c for c in b.columns if c not in a.columns or c in ("tokenizer", "language")]
    merged = a.merge(b[keep_b], on=["tokenizer", "language"], how="left") if not b.empty else a
    langs = resolve(list(merged.language.unique()), [])
    share = share_features(langs)
    share_df = pd.DataFrame(share).T.rename_axis("language").reset_index()
    merged = merged.merge(share_df, on="language", how="left")
    merged["is_specialized"] = (merged.group == "specialized").astype(int)
    merged["is_byte"] = merged.tokenizer.isin(BYTE_TOKENIZERS).astype(int)
    merged["agglutinative"] = merged.language.map(
        lambda c: BY_CODE[c].agglutinative if c in BY_CODE else np.nan
    )
    merged["log_fertility"] = np.log(merged.fertility.clip(lower=1e-6))
    merged["is_indic"] = merged.family.isin(["Indo-Aryan", "Dravidian"]).astype(int)
    if "script_vocab_count" in merged.columns:
        merged["log10_script_vocab"] = np.log10(
            merged.script_vocab_count.fillna(0).clip(lower=1))
    merged["log10_vocab"] = np.log10(merged.vocab_size.clip(lower=1))
    out = paths.TABLES_DIR / "merged_tokenizer_language.csv"
    merged.to_csv(out, index=False)
    log.info("merged dataset -> %s (%d rows, %d tokenizers, %d languages)",
             paths.relpath(out), len(merged), merged.tokenizer.nunique(),
             merged.language.nunique())
    return merged


# ---------------------------------------------------------------------------
# regression helpers
# ---------------------------------------------------------------------------
def _fit(formula: str, data: pd.DataFrame, cluster: str | None = "language"):
    import statsmodels.formula.api as smf

    data = data.dropna(subset=_formula_vars(formula, data))
    if len(data) < 8:
        log.warning("skipping model (n=%d too small): %s", len(data), formula)
        return None
    model = smf.ols(formula, data=data)
    if cluster and data[cluster].nunique() > 2:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})
    else:
        res = model.fit(cov_type="HC3")
    log.info("MODEL %s   (n=%d, %s)", formula, len(data),
             "cluster by " + cluster if cluster else "HC3")
    for line in str(res.summary()).splitlines():
        log.debug(line)
    return res


def _formula_vars(formula: str, data: pd.DataFrame) -> list[str]:
    import re as _re

    names = set(_re.findall(r"[A-Za-z_][A-Za-z_0-9]*", formula))
    return [c for c in data.columns if c in names]


def _coef_table(res, label: str) -> pd.DataFrame:
    if res is None:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "model": label,
            "term": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "t": res.tvalues.values,
            "p_value": res.pvalues.values,
            "ci_low": res.conf_int()[0].values,
            "ci_high": res.conf_int()[1].values,
        }
    )
    df["n"] = int(res.nobs)
    df["r2"] = res.rsquared
    df["r2_adj"] = res.rsquared_adj
    return df


def _write_table(df: pd.DataFrame, name: str, float_fmt: str = "%.4f") -> None:
    if df is None or df.empty:
        log.warning("table %s is empty — not written", name)
        return
    csv = paths.TABLES_DIR / f"{name}.csv"
    md = paths.TABLES_DIR / f"{name}.md"
    df.to_csv(csv, index=False, float_format=float_fmt)
    try:
        md.write_text(df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        log.warning("markdown export failed for %s: %s", name, exc)
    log.info("table -> %s", paths.relpath(csv))


# ---------------------------------------------------------------------------
# H1 — boundary misalignment explains fertility, with typological structure
# ---------------------------------------------------------------------------
def _h1_models(metric: str, budget_control: bool, source_control: bool = False,
               ) -> dict[str, str]:
    """The H1 model ladder.

    The vocabulary-budget control (``log10_script_vocab``) is deliberately *not*
    folded into the headline model: M4 is the claim as the audit literature would
    state it, and M7 is the same model after holding constant how much vocabulary the
    tokenizer spends on that script.  Reporting both is the point — the difference
    between them is itself a finding.
    """
    source = " + C(gold_source)" if source_control else ""
    return {
        "M1 fertility~F1": f"fertility ~ {metric}",
        "M2 +resource": f"fertility ~ {metric} + log10_wiki_articles",
        "M3 +family": f"fertility ~ {metric} + log10_wiki_articles + C(family)",
        "M4 interaction (headline)":
            f"fertility ~ {metric} * agglutinative + log10_wiki_articles{source}",
        "M5 +tokenizer FE":
            f"fertility ~ {metric} * agglutinative + log10_wiki_articles"
            f"{source} + C(tokenizer)",
        "M6 integrity":
            "fertility ~ morpheme_integrity * agglutinative + log10_wiki_articles",
        **({"M7 +vocabulary budget":
            f"fertility ~ {metric} * agglutinative + log10_wiki_articles"
            f"{source} + log10_script_vocab"} if budget_control else {}),
        **({"M8 budget + tokenizer FE":
            f"fertility ~ {metric} * agglutinative + log10_wiki_articles"
            f"{source} + log10_script_vocab + C(tokenizer)"} if budget_control else {}),
    }


def analyse_h1(merged: pd.DataFrame, include_byte: bool = False,
               min_annotated_words: int = 50) -> pd.DataFrame:
    """H1 across three nested robustness scopes.

    * ``indic`` — the paper's primary sample (Indo-Aryan vs Dravidian);
    * ``all`` — plus the non-Indic typological controls (Hungarian, Turkish,
      Mongolian, Czech, Russian, Spanish, German, Swahili), which show the
      agglutinative slope is not an artifact of Indic scripts or resource levels;
    * ``annotated`` — the same model on ``boundary_f1_annotated``, i.e. gold from
      SIGMORPHON/UniMorph only, never from the unsupervised Morfessor analyzer.

    The third scope is the one that answers "you are measuring agreement between two
    unsupervised segmenters"; it is reported next to the primary result, not instead.
    """
    df = merged.copy()
    if not include_byte:
        df = df[df.is_byte == 0]
    df = df.dropna(subset=["boundary_f1", "fertility"])
    if df.empty:
        log.error("H1: no rows with both fertility and boundary_f1 — run Part B")
        return pd.DataFrame()

    has_budget = "log10_script_vocab" in df.columns and df.log10_script_vocab.notna().any()
    scopes: list[tuple[str, pd.DataFrame, str]] = [
        ("indic", df[df.is_indic == 1], "boundary_f1"),
        ("all", df, "boundary_f1"),
    ]
    if "boundary_f1_annotated" in df.columns:
        ann = df[df.get("n_gold_words_annotated", 0) >= min_annotated_words].copy()
        ann = ann.dropna(subset=["boundary_f1_annotated"])
        langs = sorted(ann.language.unique())
        log.info("H1 annotated scope: %d rows over %d languages (%s) with >= %d "
                 "annotated gold words", len(ann), len(langs), ", ".join(langs),
                 min_annotated_words)
        scopes.append(("annotated", ann, "boundary_f1_annotated"))

    tables = []
    for scope, sub, metric in scopes:
        if sub.empty:
            log.warning("H1 scope %s is empty — skipped", scope)
            continue
        needs_source = (scope == "annotated" and "gold_source" in sub.columns
                        and sub.gold_source.nunique() > 1)
        if needs_source:
            log.info("H1 annotated scope pools %d gold sources (%s); each model "
                     "controls for it", sub.gold_source.nunique(),
                     ", ".join(sorted(sub.gold_source.unique())))
        for label, formula in _h1_models(metric, has_budget, needs_source).items():
            res = _fit(formula, sub)
            t = _coef_table(res, label)
            if not t.empty:
                t.insert(0, "scope", scope)
                t["metric"] = metric
                t["n_languages"] = sub.language.nunique()
                tables.append(t)
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if out.empty:
        log.warning("H1: every model was skipped (too few rows) — "
                    "run Part B on more tokenizers/languages")
    _write_table(out, "h1_fertility_regressions")

    # per-family simple slopes (the sentence the paper quotes), per scope
    slopes = []
    for scope, sub, metric in scopes:
        for fam, grp in sub.groupby("morph_type"):
            res = _fit(f"fertility ~ {metric}", grp, cluster="language")
            if res is None:
                continue
            slopes.append(
                {
                    "scope": scope,
                    "morph_type": fam,
                    "n": int(res.nobs),
                    "n_languages": grp.language.nunique(),
                    "slope": res.params.get(metric, np.nan),
                    "std_err": res.bse.get(metric, np.nan),
                    "p_value": res.pvalues.get(metric, np.nan),
                    "r2": res.rsquared,
                    "mean_fertility": grp.fertility.mean(),
                    "mean_boundary_f1": grp[metric].mean(),
                }
            )
    _write_table(pd.DataFrame(slopes), "h1_family_slopes")

    corr = (
        df.groupby("morph_type")[["fertility", "boundary_f1", "morpheme_integrity"]]
        .corr()
        .reset_index()
    )
    _write_table(corr, "h1_correlations")

    # gold provenance table — the paper must state what the gold actually is
    prov_cols = [c for c in ["language", "language_name", "family", "morph_type",
                             "n_gold_words", "n_gold_words_annotated",
                             "n_gold_words_agreed", "boundary_f1", "boundary_f1_annotated"]
                 if c in df.columns]
    prov = (df[prov_cols].groupby(
        [c for c in ["language", "language_name", "family", "morph_type"]
         if c in prov_cols], as_index=False).mean(numeric_only=True))
    _write_table(prov, "h1_gold_provenance")
    return out


# ---------------------------------------------------------------------------
# H2 — specialized tokenizers mark the attainable frontier
# ---------------------------------------------------------------------------
def analyse_h2(merged: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    indic = merged[
        merged.family.isin(["Indo-Aryan", "Dravidian"]) & (merged.is_byte == 0)
    ]
    if indic.empty:
        return pd.DataFrame()
    metrics = ["fertility", "premium_corpus", "boundary_f1", "morpheme_integrity",
               "fragmentation_rate", "tokens_per_gold_word"]
    rows = []
    for m in metrics:
        if m not in indic.columns:
            continue
        spec = indic.loc[indic.is_specialized == 1, m].dropna()
        main = indic.loc[indic.is_specialized == 0, m].dropna()
        if len(spec) < 3 or len(main) < 3:
            continue
        u, p_u = stats.mannwhitneyu(spec, main, alternative="two-sided")
        t, p_t = stats.ttest_ind(spec, main, equal_var=False)
        pooled = np.sqrt((spec.var(ddof=1) + main.var(ddof=1)) / 2)
        rows.append(
            {
                "metric": m,
                "specialized_mean": spec.mean(),
                "mainstream_mean": main.mean(),
                "difference": spec.mean() - main.mean(),
                "cohens_d": (spec.mean() - main.mean()) / pooled if pooled else np.nan,
                "mannwhitney_u": u,
                "p_mannwhitney": p_u,
                "welch_t": t,
                "p_welch": p_t,
                "n_specialized": len(spec),
                "n_mainstream": len(main),
            }
        )
    out = pd.DataFrame(rows)
    _write_table(out, "h2_specialized_vs_mainstream")

    frontier_cols = [c for c in ["fertility", "premium_corpus", "boundary_f1",
                                 "morpheme_integrity", "fragmentation_rate"]
                     if c in merged.columns]
    frontier = (
        merged[merged.family.isin(["Indo-Aryan", "Dravidian"])]
        .groupby(["group", "tokenizer", "tokenizer_name"])[frontier_cols]
        .mean()
        .reset_index()
        .sort_values("fertility")
    )
    _write_table(frontier, "h2_frontier_by_tokenizer")

    per_lang = (
        merged[merged.is_byte == 0]
        .pivot_table(index="language", columns="group", values="fertility", aggfunc="mean")
        .reset_index()
    )
    _write_table(per_lang, "h2_fertility_by_group_language")
    _h2_budget_control(indic)
    return out


def _h2_budget_control(indic: pd.DataFrame) -> pd.DataFrame:
    """Is the frontier just a bigger vocabulary allowance?

    A specialized tokenizer spends more of its vocabulary on Indic scripts, which by
    itself buys compression.  The question that matters for this paper is whether it
    *also* buys morphological alignment beyond what its compression and its script
    budget already explain.  These models hold both constant:

        morpheme_integrity ~ fertility + log10_script_vocab + is_specialized
        boundary_f1        ~ fertility + log10_script_vocab + is_specialized

    A surviving ``is_specialized`` coefficient is the answer to the objection; a
    vanishing one would mean H2 is a budget story and the paper must say so.
    """
    df = indic.copy()
    if "log10_script_vocab" not in df.columns:
        log.warning("H2 budget control skipped: no script-allocation columns "
                    "(re-run Part A to add them)")
        return pd.DataFrame()
    df = df.dropna(subset=["log10_script_vocab", "fertility"])
    tables = []
    models = {
        "B1 integrity~specialized": "morpheme_integrity ~ is_specialized",
        "B2 +compression": "morpheme_integrity ~ fertility + is_specialized",
        "B3 +script budget (headline)":
            "morpheme_integrity ~ fertility + log10_script_vocab + is_specialized",
        "B4 F1 +script budget":
            "boundary_f1 ~ fertility + log10_script_vocab + is_specialized",
        "B5 fertility~budget only":
            "fertility ~ log10_script_vocab + log10_wiki_articles",
        "B6 fertility +alignment":
            "fertility ~ log10_script_vocab + log10_wiki_articles + boundary_f1",
    }
    for label, formula in models.items():
        res = _fit(formula, df)
        t = _coef_table(res, label)
        if not t.empty:
            tables.append(t)
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    _write_table(out, "h2_budget_control")

    alloc = (df.groupby(["group", "tokenizer"])
             .agg(vocab_size=("vocab_size", "first"),
                  script_vocab_count=("script_vocab_count", "mean"),
                  script_vocab_share=("script_vocab_share", "mean"),
                  fertility=("fertility", "mean"),
                  boundary_f1=("boundary_f1", "mean"),
                  morpheme_integrity=("morpheme_integrity", "mean"))
             .reset_index()
             .sort_values("script_vocab_share", ascending=False))
    _write_table(alloc, "h2_vocabulary_allocation")
    return out


# ---------------------------------------------------------------------------
# H3 — capability: fertility and boundary-F1 under a fixed budget
# ---------------------------------------------------------------------------
def analyse_h3(merged: pd.DataFrame, run_id: str | None = None,
               model_tokenizer: str | None = None, pool: bool = True) -> pd.DataFrame:
    c, _used = load_part_c_pooled(run_id, pool=pool)
    if c.empty:
        log.warning("H3 skipped: no Part C results")
        # drop stale H3 tables so figures/tables never mix runs silently
        for name in ("h3_budget_loss_joined", "h3_capability_regressions",
                     "h3_accuracy_by_condition"):
            for ext in ("csv", "md"):
                stale = paths.TABLES_DIR / f"{name}.{ext}"
                if stale.is_file():
                    stale.unlink()
                    log.info("removed stale %s", paths.relpath(stale))
        return pd.DataFrame()

    # each Part C row is joined to the metrics of *its own* model's tokenizer
    metric_cols = ["language", "fertility", "boundary_f1", "morpheme_integrity",
                   "fragmentation_rate", "log10_wiki_articles", "agglutinative",
                   "morph_type"]
    available = set(merged.tokenizer)
    keys = sorted({k for k in c.get("model_tokenizer_key", pd.Series(dtype=str)).dropna()
                   .unique()} | ({model_tokenizer} if model_tokenizer else set()))
    fallback = "qwen3" if "qwen3" in available else (sorted(available)[0] if available
                                                    else None)
    frames = []
    for key in keys or [fallback]:
        use = model_tokenizer or key
        if use not in available:
            log.warning("H3: tokenizer %r not in the audit; falling back to %s",
                        use, fallback)
            use = fallback
        if use is None:
            continue
        m = merged[merged.tokenizer == use][metric_cols].copy()
        m["model_tokenizer_key"] = key
        frames.append(m)
    tokmetrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    base = (c[c.condition == "unconstrained"]
            .set_index(["model", "task", "language"])["accuracy"]
            .rename("accuracy_unconstrained"))
    df = c[c.condition != "unconstrained"].join(base, on=["model", "task", "language"])
    if not tokmetrics.empty and "model_tokenizer_key" in df.columns:
        df = df.merge(tokmetrics, on=["model_tokenizer_key", "language"], how="left",
                      suffixes=("", "_tok"))
    elif not tokmetrics.empty:
        df = df.merge(tokmetrics.drop(columns=["model_tokenizer_key"]), on="language",
                      how="left", suffixes=("", "_tok"))
    df["accuracy_loss"] = df.accuracy_unconstrained - df.accuracy
    df["relative_loss"] = df.accuracy_loss / df.accuracy_unconstrained.clip(lower=1e-6)
    df["budget_num"] = df.budget.fillna(0).astype(int)
    _write_table(df, "h3_budget_loss_joined")

    multi_model = df.model.nunique() > 1 if "model" in df.columns else False
    model_fe = " + C(model)" if multi_model else ""
    if not multi_model:
        log.warning("H3 runs on a single model: fertility and boundary-F1 vary only "
                    "across languages, so the partial effect of boundary-F1 is "
                    "underpowered. Run Part C with a second model to pool.")
    tables = []
    for task, sub in df.groupby("task"):
        for label, formula in {
            "L1 loss~fertility": f"accuracy_loss ~ fertility + C(budget_num){model_fe}",
            "L2 loss~fertility+F1":
                f"accuracy_loss ~ fertility + boundary_f1 + C(budget_num){model_fe}",
            "L3 shots~fertility": f"mean_shots ~ fertility{model_fe}",
            "L4 acc~fertility+F1":
                f"accuracy ~ fertility + boundary_f1 + C(budget_num){model_fe}",
        }.items():
            res = _fit(formula, sub)
            t = _coef_table(res, f"{task}: {label}")
            if not t.empty:
                t["task"] = task
                t["n_models"] = df.model.nunique() if "model" in df.columns else 1
                tables.append(t)
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    _write_table(out, "h3_capability_regressions")

    # collinearity is the reason H3 is hard — report it rather than hiding it
    if {"fertility", "boundary_f1"} <= set(df.columns):
        sub = df[["fertility", "boundary_f1"]].dropna()
        if len(sub) > 3:
            r = float(np.corrcoef(sub.fertility, sub.boundary_f1)[0, 1])
            log.info("H3 collinearity: corr(fertility, boundary_f1) = %+.3f over %d "
                     "cells (VIF ≈ %.1f)", r, len(sub), 1 / max(1e-6, 1 - r * r))

    summary = (
        c.pivot_table(index=["task", "language", "family", "morph_type"],
                      columns="condition", values=["accuracy", "mean_shots",
                                                   "mean_prompt_tokens"])
        .reset_index()
    )
    summary.columns = ["_".join(str(x) for x in col if x) for col in summary.columns]
    _write_table(summary, "h3_accuracy_by_condition")
    return out


# ---------------------------------------------------------------------------
# H4 — counter-cost check on English
# ---------------------------------------------------------------------------
def analyse_h4(merged: pd.DataFrame) -> pd.DataFrame:
    eng = merged[merged.language == "eng"]
    if eng.empty:
        log.warning("H4 skipped: English row missing")
        return pd.DataFrame()
    cols = [c for c in ["tokenizer", "tokenizer_name", "group", "vocab_size", "fertility",
                        "chars_per_token", "fragmentation_rate", "boundary_f1",
                        "morpheme_integrity"] if c in eng.columns]
    table = eng[cols].sort_values(["group", "fertility"])
    _write_table(table, "h4_english_cost")

    indic_mean = (
        merged[merged.family.isin(["Indo-Aryan", "Dravidian"])]
        .groupby("tokenizer")["fertility"].mean().rename("indic_fertility")
    )
    trade = eng.set_index("tokenizer")[["group", "fertility"]].rename(
        columns={"fertility": "english_fertility"}).join(indic_mean).reset_index()
    trade["indic_minus_english"] = trade.indic_fertility - trade.english_fertility
    _write_table(trade, "h4_tradeoff")
    return table


# ---------------------------------------------------------------------------
def run_analysis(run_id: str | None = None, include_byte: bool = False,
                 model_tokenizer: str | None = None,
                 pool_part_c: bool = True) -> dict[str, pd.DataFrame]:
    paths.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged = build_merged(run_id)
    results = {
        "merged": merged,
        "h1": analyse_h1(merged, include_byte=include_byte),
        "h2": analyse_h2(merged),
        "h3": analyse_h3(merged, run_id, model_tokenizer, pool=pool_part_c),
        "h4": analyse_h4(merged),
    }
    _headline(results)
    _snapshot_tables(run_id)
    return results


def _snapshot_tables(run_id: str | None) -> None:
    """Copy this analysis into ``runs/<run_id>/analysis/`` so each run keeps its own
    tables; ``results/tables/`` always mirrors the most recent analyze."""
    if not run_id:
        return
    import shutil

    dest = paths.RUNS_DIR / run_id / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in paths.TABLES_DIR.glob("*.*"):
        shutil.copy2(src, dest / src.name)
        n += 1
    log.info("snapshot of %d tables -> %s", n, paths.relpath(dest))


def _headline(results: dict[str, pd.DataFrame]) -> None:
    """Log the two or three numbers the paper leads with."""
    h1 = results.get("h1")
    log.info("-" * 88)
    if h1 is not None and not h1.empty:
        for scope in [s for s in ("indic", "all", "annotated") if s in set(h1.scope)]:
            head = h1[(h1.scope == scope) & h1.model.str.startswith("M4")]
            metric = head.metric.iloc[0] if not head.empty else "boundary_f1"
            for term in (metric, f"{metric}:agglutinative", "agglutinative"):
                row = head[head.term == term]
                if not row.empty:
                    r = row.iloc[0]
                    log.info("H1[%-9s] %-30s beta=%+.3f (SE %.3f) p=%.4g "
                             "[n=%d, langs=%d, R2=%.3f]", scope, term, r.coef,
                             r.std_err, r.p_value, r.n, r.n_languages, r.r2)
    h2 = results.get("h2")
    if h2 is not None and not h2.empty:
        for _, r in h2.iterrows():
            log.info("H2 %-22s specialized=%.3f vs mainstream=%.3f (d=%+.2f, p=%.4g)",
                     r.metric, r.specialized_mean, r.mainstream_mean, r.cohens_d,
                     r.p_mannwhitney)
    budget = paths.TABLES_DIR / "h2_budget_control.csv"
    if budget.is_file():
        bt = pd.read_csv(budget)
        row = bt[(bt.model.str.startswith("B3")) & (bt.term == "is_specialized")]
        if not row.empty:
            r = row.iloc[0]
            log.info("H2 integrity | fertility + script budget: is_specialized "
                     "beta=%+.3f p=%.4g (n=%d) — %s", r.coef, r.p_value, r.n,
                     "survives the budget control" if r.p_value < 0.05
                     else "does NOT survive: report H2 as a budget effect")
    h3 = results.get("h3")
    if h3 is not None and not h3.empty:
        sub = h3[h3.model.str.contains("L2") & h3.term.isin(["fertility", "boundary_f1"])]
        for _, r in sub.iterrows():
            log.info("H3 %-30s beta=%+.4f p=%.4g (n=%d)", f"{r.model}/{r.term}",
                     r.coef, r.p_value, r.n)
    log.info("tables written to %s", paths.relpath(paths.TABLES_DIR))
    log.info("-" * 88)
