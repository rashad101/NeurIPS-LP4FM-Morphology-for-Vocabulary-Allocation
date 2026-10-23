"""Part B — morpheme–token boundary alignment (CPU).  **The core analysis.**

For each (tokenizer × language) we compare the tokenizer's character-level splits of
gold word types against gold morpheme boundaries:

* **boundary precision / recall / F1** (exact and ±1 character), type- and
  frequency-weighted;
* **morpheme integrity** — share of gold morphemes wholly inside one token;
* **token purity** — share of tokens wholly inside one gold morpheme;
* **mono-split rate** — how often a monomorphemic word is split anyway;
* over/under-segmentation ratio and tokens-per-gold-word (the Part B fertility).

Per-word decisions are written to parquet so the released artifact contains not just
scores but every alignment judgement.  One tokenizer is resident at a time.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import pandas as pd

from . import paths
from .data_morph import GoldWord, load_gold
from .languages import Language
from .logging_utils import Timer, get_logger
from .progress import pbar
from .state import RunContext
from .tokenizer_zoo import LoadedTokenizer, TokenizerSpec, load, unload

log = get_logger("part_b")

WORD_DIR = "words"
QUALITATIVE_JSON = "qualitative_examples.json"


def _unit_key(tok_key: str, lang_code: str) -> str:
    return f"{tok_key}::{lang_code}"


def _greedy_match(gold: list[int], pred: list[int], tol: int) -> int:
    """Number of gold boundaries matched by a distinct predicted boundary."""
    if tol == 0:
        return len(set(gold) & set(pred))
    used = [False] * len(pred)
    matched = 0
    for g in gold:
        best, best_d = -1, tol + 1
        for i, p in enumerate(pred):
            if used[i]:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best, best_d = i, d
        if best >= 0:
            used[best] = True
            matched += 1
    return matched


def _spans_to_boundaries(spans: list[tuple[int, int]], length: int) -> list[int]:
    return sorted({e for _, e in spans if 0 < e < length})


def _morph_spans(morphs: list[str]) -> list[tuple[int, int]]:
    out, cum = [], 0
    for m in morphs:
        out.append((cum, cum + len(m)))
        cum += len(m)
    return out


def score_word(lt: LoadedTokenizer, g: GoldWord, tol: int = 1) -> dict:
    spans, aligned, mids = lt.word_spans(g.word)
    n = len(g.word)
    pred = _spans_to_boundaries(spans, n)
    gold = [b for b in g.boundaries if 0 < b < n]

    tp = _greedy_match(gold, pred, 0)
    tp_tol = _greedy_match(gold, pred, tol)

    gold_spans = _morph_spans(g.morphs)
    intact = sum(
        1 for (s, e) in gold_spans if any(ts <= s and e <= te for ts, te in spans)
    )
    pure = sum(
        1 for (ts, te) in spans if any(s <= ts and te <= e for s, e in gold_spans)
    )
    return {
        "word": g.word,
        "freq": g.freq,
        "tier": g.tier,
        "sources": g.sources,
        "n_chars": n,
        "n_morphs": g.n_morphs,
        # n_tokens is the true token count; n_spans is the number of *character*
        # spans, which differs for byte-level tokenizers whose splits fall inside
        # a codepoint (those cannot be character boundaries).
        "n_tokens": lt.tokens_per_word(g.word),
        "n_spans": len(spans),
        "gold_boundaries": gold,
        "pred_boundaries": pred,
        "n_gold": len(gold),
        "n_pred": len(pred),
        "tp": tp,
        "tp_tol": tp_tol,
        "morphs_intact": intact,
        "tokens_pure": pure,
        "aligned": bool(aligned),
        "mid_char_splits": mids,
    }


def _aggregate_words(rows: list[dict]) -> dict:
    def _f1(p: float, r: float) -> float:
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    def _micro(weighted: bool, subset: list[dict] | None = None,
               suffix: str | None = None) -> dict:
        rows_ = rows if subset is None else subset
        w = (lambda r: r["freq"]) if weighted else (lambda r: 1)
        tp = sum(r["tp"] * w(r) for r in rows_)
        tp_tol = sum(r["tp_tol"] * w(r) for r in rows_)
        n_pred = sum(r["n_pred"] * w(r) for r in rows_)
        n_gold = sum(r["n_gold"] * w(r) for r in rows_)
        p = tp / n_pred if n_pred else 0.0
        r_ = tp / n_gold if n_gold else 0.0
        p_t = tp_tol / n_pred if n_pred else 0.0
        r_t = tp_tol / n_gold if n_gold else 0.0
        if suffix is None:
            suffix = "_freq" if weighted else ""
        return {
            f"boundary_precision{suffix}": p,
            f"boundary_recall{suffix}": r_,
            f"boundary_f1{suffix}": _f1(p, r_),
            f"boundary_f1_tol1{suffix}": _f1(p_t, r_t),
            f"n_pred_boundaries{suffix}": n_pred,
            f"n_gold_boundaries{suffix}": n_gold,
        }

    n_words = len(rows)
    n_morphs = sum(r["n_morphs"] for r in rows)
    n_tokens = sum(r["n_tokens"] for r in rows)
    n_spans = sum(r["n_spans"] for r in rows)
    mono = [r for r in rows if r["n_gold"] == 0]
    out = {
        "n_gold_words": n_words,
        "n_gold_morphs": n_morphs,
        "morpheme_integrity": sum(r["morphs_intact"] for r in rows) / max(1, n_morphs),
        "token_purity": sum(r["tokens_pure"] for r in rows) / max(1, n_spans),
        "tokens_per_gold_word": n_tokens / max(1, n_words),
        "spans_per_gold_word": n_spans / max(1, n_words),
        "morphs_per_gold_word": n_morphs / max(1, n_words),
        "oversegmentation_ratio": (
            sum(r["n_pred"] for r in rows) / max(1, sum(r["n_gold"] for r in rows))
        ),
        "mono_split_rate": (
            sum(1 for r in mono if r["n_pred"] > 0) / len(mono) if mono else float("nan")
        ),
        "n_monomorphemic": len(mono),
        "unalignable_rate": sum(1 for r in rows if not r["aligned"]) / max(1, n_words),
        "mid_char_split_word_rate": (
            sum(1 for r in rows if r["mid_char_splits"]) / max(1, n_words)
        ),
    }
    out.update(_micro(False))
    out.update(_micro(True))

    # --- annotated subset (tiers A+B): gold that does not come from an -----------
    # unsupervised analyzer.  H1's primary robustness check runs on these columns.
    annotated = [r for r in rows if r["tier"] in ("A", "B")]
    out.update(_micro(False, annotated, suffix="_annotated"))
    ann_morphs = sum(r["n_morphs"] for r in annotated)
    ann_spans = sum(r["n_spans"] for r in annotated)
    # Gold sources differ in granularity: a UniMorph-derived analysis marks exactly one
    # stem/affix boundary, a SIGMORPHON analysis marks every morpheme.  Boundary recall
    # is therefore not comparable across sources, so the source is recorded and enters
    # the annotated-scope regression as a control.
    src_counts: Counter = Counter()
    for r in annotated:
        for s in r.get("sources") or []:
            src_counts[s] += 1
    out.update(
        {
            "gold_source": (src_counts.most_common(1)[0][0] if src_counts else "none"),
            "gold_source_mix": ";".join(f"{k}:{v}" for k, v in src_counts.most_common()),
            "gold_boundaries_per_word_annotated": (
                sum(r["n_gold"] for r in annotated) / max(1, len(annotated))),
            "n_gold_words_annotated": len(annotated),
            "n_gold_words_agreed": sum(1 for r in rows if r["tier"] == "A"),
            "morpheme_integrity_annotated": (
                sum(r["morphs_intact"] for r in annotated) / max(1, ann_morphs)),
            "token_purity_annotated": (
                sum(r["tokens_pure"] for r in annotated) / max(1, ann_spans)),
            "tokens_per_gold_word_annotated": (
                sum(r["n_tokens"] for r in annotated) / max(1, len(annotated))),
        }
    )

    # per-tier F1 (analyzer-noise robustness check)
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
    for tier, sub in by_tier.items():
        tp = sum(r["tp"] for r in sub)
        np_ = sum(r["n_pred"] for r in sub)
        ng = sum(r["n_gold"] for r in sub)
        p = tp / np_ if np_ else 0.0
        rr = tp / ng if ng else 0.0
        out[f"boundary_f1_tier{tier}"] = 0.0 if (p + rr) == 0 else 2 * p * rr / (p + rr)
        out[f"n_words_tier{tier}"] = len(sub)
    return out


def run_part_b(
    tok_specs: list[TokenizerSpec],
    languages: list[Language],
    ctx: RunContext,
    tolerance: int = 1,
    save_words: bool = True,
) -> pd.DataFrame:
    """Score every (tokenizer × language) pair, one tokenizer resident at a time."""
    gold_sets: dict[str, list[GoldWord]] = {}
    for lang in languages:
        try:
            gold = load_gold(lang)
        except FileNotFoundError as exc:
            log.error("%s: %s", lang.code, exc)
            continue
        if gold:
            gold_sets[lang.code] = gold
        else:
            log.warning("%s: empty gold set — skipped", lang.code)
    if not gold_sets:
        raise SystemExit("no gold segmentations available — run `morphtok gold` first")

    word_dir = ctx.path(WORD_DIR)
    if save_words:
        word_dir.mkdir(parents=True, exist_ok=True)

    units = [
        (_unit_key(spec.key, lang.code), (spec, lang))
        for spec in tok_specs
        for lang in languages
        if lang.code in gold_sets
    ]
    pending = [(k, v) for k, v in units if not ctx.is_done(k)]
    log.info("Part B: %d units total, %d pending (%d resumed)",
             len(units), len(pending), len(units) - len(pending))
    ctx.set_status(stage="part_b", units_total=len(units), units_pending=len(pending))

    skipped: dict[str, str] = {}
    loaded: dict[str, LoadedTokenizer] = {}
    current: str | None = None
    bar = pbar(pending, desc="Part B (boundary alignment)", unit="unit")
    for key, (spec, lang) in bar:
        bar.set_postfix_str(f"{spec.key}/{lang.code}")
        if spec.key in skipped:
            continue
        if current != spec.key:  # one tokenizer resident at a time
            if current is not None:
                unload(current)
                loaded.pop(current, None)
            try:
                loaded[spec.key] = load(spec.key)
            except Exception as exc:
                skipped[spec.key] = str(exc)
                log.error("SKIP tokenizer %s: %s", spec.key, exc)
                current = None
                continue
            current = spec.key
        lt = loaded[spec.key]
        gold = gold_sets[lang.code]
        with Timer(f"part B {key}", log):
            rows = [
                score_word(lt, g, tol=tolerance)
                for g in pbar(gold, desc=f"{spec.key}/{lang.code}", unit="word",
                              leave=False)
            ]
            if save_words:
                pd.DataFrame(rows).to_parquet(
                    word_dir / f"{spec.key}__{lang.code}.parquet", index=False
                )
            record = {
                "tokenizer": spec.key,
                "tokenizer_name": spec.name,
                "group": spec.group,
                "source": lt.source,
                "vocab_size": lt.n_vocab,
                "language": lang.code,
                "language_name": lang.name,
                "family": lang.family,
                "morph_type": lang.morph_type,
                "script": lang.script,
                "agglutinative": lang.agglutinative,
                "tolerance": tolerance,
                **_aggregate_words(rows),
            }
            ctx.commit(key, record)
        ctx.set_status(units_done=len(ctx.done))
    bar.close()
    if current is not None:
        unload(current)
        loaded.pop(current, None)
    if skipped:
        log.warning("%d tokenizer(s) skipped: %s", len(skipped), ", ".join(skipped))
    ctx.set_status(tokenizers_skipped=skipped)

    df = aggregate_part_b(ctx)
    _export_qualitative([s for s in tok_specs if s.key not in skipped], languages,
                        gold_sets, ctx)
    return df


def aggregate_part_b(ctx: RunContext) -> pd.DataFrame:
    records = ctx.records()
    if not records:
        log.warning("Part B: no records to aggregate")
        return pd.DataFrame()
    df = pd.DataFrame(records).drop(columns=["_unit", "_ts"], errors="ignore")
    df = df.drop_duplicates(subset=["tokenizer", "language"], keep="last")
    df = df.sort_values(["group", "tokenizer", "family", "language"])
    out = ctx.path("morph_alignment.csv")
    df.to_csv(out, index=False)
    log.info("Part B summary -> %s (%d rows)", paths.relpath(out), len(df))
    df.pivot_table(index="tokenizer", columns="language", values="boundary_f1",
                   aggfunc="mean").to_csv(ctx.path("boundary_f1_matrix.csv"))
    return df


def _export_qualitative(
    tok_specs: list[TokenizerSpec],
    languages: list[Language],
    gold_sets: dict[str, list[GoldWord]],
    ctx: RunContext,
    per_lang: int = 3,
) -> None:
    """Data for Figure 3: long, multi-morpheme words segmented by every tokenizer.

    Tokenizers are re-loaded one at a time here too — this runs after the main loop
    has already evicted them.
    """
    picks: dict[str, list[dict]] = {}
    for lang in languages:
        gold = gold_sets.get(lang.code)
        if not gold:
            continue
        cands = sorted(
            (g for g in gold if g.n_morphs >= 3 and g.tier in ("A", "B", "C")),
            key=lambda g: (-g.n_morphs, -len(g.word)),
        )[:per_lang]
        if cands:
            picks[lang.code] = [
                {"word": g.word, "gold_morphs": g.morphs,
                 "gold_boundaries": g.boundaries, "tier": g.tier,
                 "n_morphs": g.n_morphs, "segmentations": {}}
                for g in cands
            ]
    for spec in tok_specs:
        try:
            lt = load(spec.key)
        except Exception:
            continue
        for examples in picks.values():
            for entry in examples:
                spans, aligned, _ = lt.word_spans(entry["word"])
                entry["segmentations"][spec.key] = {
                    "pieces": [entry["word"][s:e] for s, e in spans],
                    "spans": spans,
                    "n_tokens": lt.tokens_per_word(entry["word"]),
                    "n_spans": len(spans),
                    "aligned": aligned,
                }
        unload(spec.key)
    out = {k: v for k, v in picks.items() if v}
    path = ctx.path(QUALITATIVE_JSON)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("qualitative examples -> %s (%d languages)", paths.relpath(path), len(out))
