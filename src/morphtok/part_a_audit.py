"""Part A — the Indic fertility audit (CPU).

For every (tokenizer × language) pair on FLORES-200 devtest we record fertility per
word, tokens per sentence/character/byte, fragmentation, byte-fallback usage, and the
English-relative premium computed on *aligned* sentence pairs.  Specialized Indic
tokenizers are the attainable-frontier rows (H2); ByT5/UTF-8 bound the byte-level
alternative used in the objections section.

Work unit = one (tokenizer, language) pair; each unit is committed to
``records.jsonl`` + per-sentence and per-word-type parquets the moment it finishes, so
the stage resumes exactly where it stopped and every number in the paper can be traced
back to the rows that produced it.

The loop is tokenizer-major and holds **one tokenizer in memory at a time**.
"""

from __future__ import annotations

import statistics

import pandas as pd

from . import paths
from .data_flores import load_parallel
from .languages import Language
from .logging_utils import Timer, get_logger
from .progress import pbar
from .state import RunContext
from .textutil import n_bytes, words_unicode, words_whitespace
from .tokenizer_zoo import (LoadedTokenizer, TokenizerSpec, describe, load,
                            loaded_keys, script_allocation, unload)

log = get_logger("part_a")

SENT_DIR = "sentences"
WORD_DIR = "words"
BYTE_FALLBACK_SAMPLE = 200


def _unit_key(tok_key: str, lang_code: str) -> str:
    return f"{tok_key}::{lang_code}"


def _sentence_frame(lt: LoadedTokenizer, sentences: list[str], desc: str) -> pd.DataFrame:
    n_tokens = lt.batch_n_tokens(sentences, desc=desc)
    rows = []
    for i, (sent, ntok) in enumerate(zip(sentences, n_tokens)):
        ws = words_whitespace(sent)
        uni = words_unicode(sent)
        rows.append(
            {
                "sent_id": i,
                "n_tokens": ntok,
                "n_words_ws": len(ws),
                "n_words_uni": len(uni),
                "n_chars": len(sent),
                "n_bytes": n_bytes(sent),
            }
        )
    return pd.DataFrame(rows)


def _word_level_stats(lt: LoadedTokenizer, sentences: list[str],
                      ) -> tuple[dict, list[dict]]:
    """Fragmentation + mid-character splitting over word *types*, weighted back to
    occurrences.  Also returns the per-type rows, which are saved for reuse."""
    from collections import Counter

    counts: Counter = Counter()
    for sent in sentences:
        counts.update(words_unicode(sent))
    types = list(counts)
    tokens_per_type = lt.batch_n_tokens(types, desc=f"words {lt.spec.key}")
    n_occ = sum(counts.values())
    frag_occ = single_occ = 0
    mid_char_occ = 0
    tpw_type_sum = 0
    rows: list[dict] = []
    for word, ntok in zip(types, tokens_per_type):
        c = counts[word]
        tpw_type_sum += ntok
        if ntok > 1:
            frag_occ += c
        else:
            single_occ += c
        _, _, mids = lt.word_spans(word)
        if mids:
            mid_char_occ += c
        rows.append({"word": word, "count": c, "n_tokens": ntok, "n_chars": len(word),
                     "mid_char_splits": mids})
    stats = {
        "n_word_types": len(types),
        "n_word_occurrences": n_occ,
        "fragmentation_rate": frag_occ / max(1, n_occ),
        "single_token_word_rate": single_occ / max(1, n_occ),
        "mid_character_split_rate": mid_char_occ / max(1, n_occ),
        "tokens_per_word_type": tpw_type_sum / max(1, len(types)),
    }
    return stats, rows


def run_part_a(
    tok_specs: list[TokenizerSpec],
    languages: list[Language],
    ctx: RunContext,
    split: str = "devtest",
    max_sentences: int | None = None,
    save_word_stats: bool = True,
) -> pd.DataFrame:
    """Audit every (tokenizer × language) pair, **one tokenizer resident at a time**.

    The loop is tokenizer-major and evicts each tokenizer before loading the next, so
    peak memory is one vocabulary rather than thirteen — this stage may share a box
    with the vLLM run.
    """
    corpora = load_parallel(languages, split=split)
    if max_sentences:
        corpora = {k: v[:max_sentences] for k, v in corpora.items()}
    sent_dir = ctx.path(SENT_DIR)
    sent_dir.mkdir(parents=True, exist_ok=True)
    word_dir = ctx.path(WORD_DIR)
    if save_word_stats:
        word_dir.mkdir(parents=True, exist_ok=True)

    units = [
        (_unit_key(spec.key, lang.code), (spec, lang))
        for spec in tok_specs
        for lang in languages
    ]
    pending = [(k, v) for k, v in units if not ctx.is_done(k)]
    log.info("Part A: %d units total, %d pending (%d resumed)",
             len(units), len(pending), len(units) - len(pending))
    ctx.set_status(stage="part_a", units_total=len(units), units_pending=len(pending))

    described: list[dict] = []
    skipped: dict[str, str] = {}
    bar = pbar(pending, desc="Part A (tokenizer x language)", unit="unit")
    current: str | None = None
    lt: LoadedTokenizer | None = None
    for key, (spec, lang) in bar:
        bar.set_postfix_str(f"{spec.key}/{lang.code}")
        if spec.key in skipped:
            continue
        if current != spec.key:  # tokenizer-major: swap in, evict the previous one
            if current is not None:
                unload(current)
                log.info("unloaded %s; resident tokenizers: %s", current, loaded_keys())
            try:
                lt = load(spec.key)
            except Exception as exc:
                skipped[spec.key] = str(exc)
                log.error("SKIP tokenizer %s: %s", spec.key, exc)
                current = None
                continue
            current = spec.key
            described += describe({spec.key: lt})
        assert lt is not None
        sentences = corpora[lang.code]
        with Timer(f"part A {key}", log):
            df = _sentence_frame(lt, sentences, desc=f"{spec.key}/{lang.code}")
            out_parquet = sent_dir / f"{spec.key}__{lang.code}.parquet"
            df.to_parquet(out_parquet, index=False)

            wstats, word_rows = _word_level_stats(lt, sentences)
            if save_word_stats:
                pd.DataFrame(word_rows).to_parquet(
                    word_dir / f"{spec.key}__{lang.code}.parquet", index=False)
            sample = sentences[:BYTE_FALLBACK_SAMPLE]
            bf = [lt.byte_fallback_rate(s) for s in sample]
            record = {
                "tokenizer": spec.key,
                "tokenizer_name": spec.name,
                "group": spec.group,
                "source": lt.source,
                "vocab_size": lt.n_vocab,
                "language": lang.code,
                "language_name": lang.name,
                "flores": lang.flores,
                "script": lang.script,
                "family": lang.family,
                "morph_type": lang.morph_type,
                "split": split,
                "n_sentences": len(sentences),
                "n_tokens": int(df.n_tokens.sum()),
                "n_words_ws": int(df.n_words_ws.sum()),
                "n_words_uni": int(df.n_words_uni.sum()),
                "n_chars": int(df.n_chars.sum()),
                "n_bytes": int(df.n_bytes.sum()),
                # micro (corpus-level) fertility is the headline number
                "fertility": float(df.n_tokens.sum() / max(1, df.n_words_ws.sum())),
                "fertility_uni": float(df.n_tokens.sum() / max(1, df.n_words_uni.sum())),
                "fertility_macro": float((df.n_tokens / df.n_words_ws.clip(lower=1)).mean()),
                "fertility_sd": float((df.n_tokens / df.n_words_ws.clip(lower=1)).std()),
                "tokens_per_sentence": float(df.n_tokens.mean()),
                "chars_per_token": float(df.n_chars.sum() / max(1, df.n_tokens.sum())),
                "bytes_per_token": float(df.n_bytes.sum() / max(1, df.n_tokens.sum())),
                "byte_fallback_rate": float(statistics.fmean(bf)) if bf else float("nan"),
                "parquet": paths.relpath(out_parquet),
                **wstats,
                # vocabulary budget actually spent on this language's script — the
                # control for "specialized tokenizers just have a bigger allowance"
                **script_allocation(spec.key, lang.script),
            }
            ctx.commit(key, record)
        ctx.set_status(units_done=len(ctx.done))
    bar.close()
    if current is not None:
        unload(current)
    if skipped:
        log.warning("%d tokenizer(s) skipped: %s. Set HF_TOKEN or write .hf_token to "
                    "include gated ones.", len(skipped), ", ".join(skipped))
    ctx.set_status(tokenizers_skipped=skipped)

    df_summary = aggregate_part_a(ctx)
    if described:
        info_path = ctx.path("tokenizer_info.csv")
        prev = pd.read_csv(info_path) if info_path.is_file() else pd.DataFrame()
        info = pd.concat([prev, pd.DataFrame(described)], ignore_index=True)
        info.drop_duplicates(subset=["tokenizer"], keep="last").to_csv(info_path,
                                                                      index=False)
    return df_summary


def _report_duplicate_tokenizers(df: pd.DataFrame, ctx: RunContext) -> None:
    """Flag audited tokenizers that are the *same vocabulary* under two names.

    Model families increasingly ship regionally adapted checkpoints that reuse a
    base tokenizer verbatim (e.g. Sarvam-M over Mistral-NeMo); counting those twice
    would inflate the audit and mislabel them as specialized.
    """
    fp = (df.pivot_table(index="tokenizer", columns="language", values="n_tokens",
                         aggfunc="sum")
          .apply(lambda r: tuple(r.fillna(-1).astype(int)), axis=1))
    groups: dict[tuple, list[str]] = {}
    for tok, key in fp.items():
        groups.setdefault(key, []).append(str(tok))
    dups = [v for v in groups.values() if len(v) > 1]
    rows = [{"group_id": i, "tokenizers": ", ".join(sorted(g))}
            for i, g in enumerate(dups)]
    if rows:
        pd.DataFrame(rows).to_csv(ctx.path("tokenizer_duplicates.csv"), index=False)
        for g in dups:
            log.warning("identical token counts across all languages: %s — same "
                        "vocabulary under different names", ", ".join(sorted(g)))


def aggregate_part_a(ctx: RunContext) -> pd.DataFrame:
    """Build ``indic_audit.csv`` from committed records (works on partial runs)."""
    records = ctx.records()
    if not records:
        log.warning("Part A: no records to aggregate")
        return pd.DataFrame()
    df = pd.DataFrame(records).drop(columns=["_unit", "_ts"], errors="ignore")
    df = df.drop_duplicates(subset=["tokenizer", "language"], keep="last")

    # English-relative premium on aligned sentence pairs (per tokenizer)
    prem_rows = []
    for tok_key, grp in df.groupby("tokenizer"):
        eng_file = ctx.path(SENT_DIR) / f"{tok_key}__eng.parquet"
        if not eng_file.is_file():
            continue
        eng = pd.read_parquet(eng_file).set_index("sent_id")
        for lang in grp.language:
            f = ctx.path(SENT_DIR) / f"{tok_key}__{lang}.parquet"
            if not f.is_file():
                continue
            cur = pd.read_parquet(f).set_index("sent_id")
            joined = cur.join(eng, rsuffix="_eng", how="inner")
            ratio = joined.n_tokens / joined.n_tokens_eng.clip(lower=1)
            prem_rows.append(
                {
                    "tokenizer": tok_key,
                    "language": lang,
                    "premium_mean": float(ratio.mean()),
                    "premium_median": float(ratio.median()),
                    "premium_p90": float(ratio.quantile(0.9)),
                    "premium_corpus": float(
                        joined.n_tokens.sum() / max(1, joined.n_tokens_eng.sum())
                    ),
                    "effective_context_ratio": float(
                        joined.n_tokens_eng.sum() / max(1, joined.n_tokens.sum())
                    ),
                    "n_aligned": int(len(joined)),
                }
            )
    if prem_rows:
        df = df.merge(pd.DataFrame(prem_rows), on=["tokenizer", "language"], how="left")

    _report_duplicate_tokenizers(df, ctx)

    df = df.sort_values(["group", "tokenizer", "family", "language"])
    out = ctx.path("indic_audit.csv")
    df.to_csv(out, index=False)
    log.info("Part A summary -> %s (%d rows)", paths.relpath(out), len(df))

    # heatmap-shaped view for the paper
    if "premium_corpus" in df.columns:
        pivot = df.pivot_table(index="tokenizer", columns="language",
                               values="premium_corpus", aggfunc="mean")
        pivot.to_csv(ctx.path("premium_matrix.csv"))
    pivot_f = df.pivot_table(index="tokenizer", columns="language",
                             values="fertility", aggfunc="mean")
    pivot_f.to_csv(ctx.path("fertility_matrix.csv"))
    return df
