"""Part D — a morphology-governed allocator, in miniature.

The position claims a principled alternative exists.  This stage builds the smallest
honest instance of it and measures the same quantities as Parts A and B, over three
arms that share a corpus, a training stream, a trainer and a vocabulary budget:

* **baseline** — ordinary byte-level BPE: allocation by frequency alone;
* **morph_train** — merges are only ever *learned* within morphemes (the training
  stream is pre-segmented by the analyzer and the marker is split away), but encoding
  unseen text is still frequency-driven;
* **morph_full** — that same vocabulary plus morphology-aware pre-tokenization at
  inference, so no token can span a morpheme boundary.

The budget is matched by construction: the constrained arm saturates first, and the
baseline is then trained to *its* realised size.

Evaluation is on FLORES devtest (fertility) and the gold word list (alignment), with
word types absent from the training lexicon reported separately — a tokenizer that
only wins on memorised types has learned nothing about morphology.
"""

from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path

import pandas as pd

from . import paths
from .data_flores import load_split
from .data_morph import MORF_DIR, load_gold, read_morfessor_counts
from .languages import Language
from .logging_utils import Timer, get_logger
from .part_b_alignment import _aggregate_words, score_word
from .progress import pbar
from .state import RunContext
from .textutil import n_bytes, words_unicode, words_whitespace
from .tokenizer_zoo import LoadedTokenizer, TokenizerSpec

log = get_logger("part_d")

#: Three arms, in increasing order of how much morphology governs allocation:
#:
#: ``baseline``    ordinary byte-level BPE — allocation by frequency alone;
#: ``morph_train`` merges *learned* only within morphemes (constrained training data),
#:                 but segmentation at inference is still frequency-driven;
#: ``morph_full``  the same vocabulary plus morphology-aware pre-tokenization at
#:                 inference — the design class the position argues for.
#:
#: The ``morph_train`` arm exists to make a point the paper should state plainly:
#: because BPE merges are context-free rules, constraining the training data alone
#: does not make the tokenizer respect boundaries on unseen text.  A principled scheme
#: has to condition the segmentation procedure, not only the vocabulary.
ARMS = ("baseline", "morph_train", "morph_full")
MAX_REPEAT = 20  # cap on how often a frequent type is repeated in the training stream

#: Marker inserted at analyzer morpheme boundaries in the *morph* arm's training
#: stream.  It is removed by a pre-tokenizer split, so no BPE merge can ever span it,
#: while word-initial space semantics (the ``Ġ`` prefix) stay identical to the
#: baseline arm.  Real text never contains it, so at inference both arms are ordinary
#: byte-level BPE tokenizers — only what they *learned* differs.
BOUNDARY = "␟"


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte→unicode map, as used by the ByteLevel pre-tokenizer."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _byte_level_form(text: str) -> str:
    m = _bytes_to_unicode()
    return "".join(m[b] for b in text.encode("utf-8"))


def _training_lexicon(lang: Language, max_types: int = 200_000,
                      ) -> tuple[dict[str, list[str]], Counter]:
    """The analyzer's frequency list: ``{word: morphs}`` plus its corpus counts.

    Selection is by corpus **frequency**, which is what a tokenizer trainer would see.
    (Selecting by length instead pulls in concatenated junk strings and the prototype
    learns nothing — a mistake worth not repeating.)
    """
    if not lang.morfessor:
        return {}, Counter()
    lex = read_morfessor_counts(MORF_DIR / f"{lang.morfessor}.model", script=lang.script)
    if not lex:
        return {}, Counter()
    top = sorted(lex.items(), key=lambda kv: (-kv[1][1], kv[0]))[:max_types]
    return ({w: morphs for w, (morphs, _) in top},
            Counter({w: c for w, (_, c) in top}))


def _counts_from_corpus(lang: Language) -> Counter:
    counts: Counter = Counter()
    for split in ("dev", "devtest"):
        try:
            for sent in load_split(lang, split):
                counts.update(words_unicode(sent))
        except FileNotFoundError:
            continue
    return counts


def _segment_text(text: str, lexicon: dict[str, list[str]]) -> str:
    """Insert BOUNDARY at analyzer morpheme boundaries, leaving everything else alone."""
    out = []
    for token in text.split(" "):
        morphs = lexicon.get(token)
        out.append(BOUNDARY.join(morphs) if morphs and len(morphs) > 1 else token)
    return " ".join(out)


def _training_lines(lexicon: dict[str, list[str]], counts: Counter,
                    sentences: list[str], segmented: bool,
                    max_repeat: int = MAX_REPEAT) -> list[str]:
    """Identical training stream for both arms, modulo the boundary markers.

    The stream mixes the analyzer's frequency-weighted word list with real sentences,
    so both arms learn word-initial (``Ġ``-prefixed) tokens and neither is penalised
    by an unrealistic training distribution.
    """
    lines: list[str] = []
    for word, morphs in lexicon.items():
        repeat = min(max_repeat, 1 + counts.get(word, 0))
        text = BOUNDARY.join(morphs) if (segmented and len(morphs) > 1) else word
        lines.extend([text] * repeat)
    for sent in sentences:
        lines.append(_segment_text(sent, lexicon) if segmented else sent)
    return lines


def train_arm(arm: str, lines: list[str], vocab_size: int, out_dir: Path) -> tuple:
    """Train one BPE arm; returns ``(fast tokenizer, realised vocabulary size)``."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token=None))
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=True)
    if arm.startswith("morph"):
        # split (and drop) the boundary marker *after* byte-level mapping, so the
        # marker cannot be merged across and word-initial spacing is unaffected
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            byte_level,
            pre_tokenizers.Split(Regex(_byte_level_form(BOUNDARY)), behavior="removed"),
        ])
    else:
        tok.pre_tokenizer = byte_level
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(lines, trainer=trainer, length=len(lines))
    realised = tok.get_vocab_size()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm}.json"
    tok.save(str(path))
    log.info("[%s] trained BPE: target vocab=%d realised=%d -> %s",
             arm, vocab_size, realised, paths.relpath(path))
    return PreTrainedTokenizerFast(tokenizer_file=str(path), unk_token="<unk>"), realised


def _wrap(hf_tok, arm: str, vocab_size: int) -> LoadedTokenizer:
    spec = TokenizerSpec(
        key=f"proto_{arm}",
        name=("prototype BPE (frequency)" if arm == "baseline"
              else "prototype BPE (morpheme-constrained)"),
        kind="hf",
        repos=(f"local:{arm}",),
        group="prototype",
        year="this paper",
        note="matched corpus, counts and vocabulary budget",
        default=False,
    )
    return LoadedTokenizer(spec, f"local:{arm}", hf_tok, vocab_size, "hf",
                           _offsets_ok=True, span_mode="hf")


class MorphAwareTokenizer:
    """A trained BPE plus morphology-aware pre-tokenization at encoding time.

    Words are first segmented by the analyzer (lexicon lookup, Morfessor Viterbi for
    out-of-lexicon types), then each morpheme is encoded independently, so no token
    can ever span a morpheme boundary.  Duck-types the parts of ``LoadedTokenizer``
    that Parts A and B use, so the identical metrics apply.
    """

    def __init__(self, inner: LoadedTokenizer, lexicon: dict[str, list[str]],
                 viterbi=None):
        self.inner = inner
        self.lexicon = lexicon
        self.viterbi = viterbi
        self.spec = inner.spec
        self.stats = inner.stats
        self.span_mode = "hf"
        self.n_vocab = inner.n_vocab
        self._cache: dict[str, list[str]] = {}

    # -- analyzer ------------------------------------------------------------
    def morphs(self, word: str) -> list[str]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        out = self.lexicon.get(word)
        if out is None and self.viterbi is not None and len(word) > 3:
            try:
                out = list(self.viterbi(word)[0])
            except Exception:  # pragma: no cover - analyzer edge cases
                out = None
        out = out or [word]
        if "".join(out) != word:  # never trust an analysis that rewrites the surface
            out = [word]
        self._cache[word] = out
        return out

    # -- encoding ------------------------------------------------------------
    def _encode_word(self, word: str, leading_space: bool) -> list[int]:
        ids: list[int] = []
        for i, morph in enumerate(self.morphs(word)):
            prefix = " " if (leading_space and i == 0) else ""
            ids += self.inner.encode(prefix + morph)
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for i, chunk in enumerate(text.split(" ")):
            if not chunk:
                continue
            ids += self._encode_word(chunk, leading_space=i > 0)
        return ids

    def n_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def tokens_per_word(self, word: str) -> int:
        return len(self._encode_word(word, leading_space=False))

    def batch_n_tokens(self, texts, chunk: int = 128, desc: str | None = None):
        from .progress import pbar

        return [self.n_tokens(t) for t in pbar(list(texts), desc=desc or "encode",
                                               unit="txt", leave=False)]

    def pieces(self, text: str) -> list[str]:
        out: list[str] = []
        for word in text.split(" "):
            for morph in self.morphs(word):
                out += self.inner.pieces(morph)
        return out

    def byte_fallback_rate(self, text: str) -> float:
        return self.inner.byte_fallback_rate(text)

    # -- character spans -----------------------------------------------------
    def word_spans(self, word: str) -> tuple[list[tuple[int, int]], bool, int]:
        spans: list[tuple[int, int]] = []
        offset = 0
        ok = True
        mids = 0
        for morph in self.morphs(word):
            sub, sub_ok, sub_mid = self.inner.word_spans(morph)
            spans += [(s + offset, e + offset) for s, e in sub]
            ok = ok and sub_ok
            mids += sub_mid
            offset += len(morph)
        return spans or [(0, len(word))], ok, mids


def _viterbi_segmenter(lang: Language, lexicon: dict[str, list[str]]):
    """Morfessor Viterbi segmentation for words outside the analyzer lexicon."""
    try:
        import morfessor

        model = morfessor.BaselineModel()
        # (count, compound, segmentation) triples
        model.load_segmentations(
            [(1, word, tuple(morphs)) for word, morphs in lexicon.items()])
        log.info("%s: Morfessor Viterbi ready for out-of-lexicon words", lang.code)
        return model.viterbi_segment
    except Exception as exc:  # pragma: no cover - optional path
        log.warning("%s: no Viterbi fallback (%s); unknown words stay unsegmented",
                    lang.code, exc)
        return None


def run_part_d(
    languages: list[Language],
    ctx: RunContext,
    vocab_size: int = 16_000,
    max_types: int = 200_000,
    split: str = "devtest",
) -> pd.DataFrame:
    """Train and evaluate both arms per language."""
    units = [(lang.code, lang) for lang in languages if lang.morfessor]
    skipped = [lang.code for lang in languages if not lang.morfessor]
    if skipped:
        log.warning("Part D needs an analyzer to constrain merges; skipping %s",
                    ", ".join(skipped))
    log.info("Part D: %d languages, vocab_size=%d per arm", len(units), vocab_size)
    ctx.set_status(stage="part_d", units_total=len(units), vocab_size=vocab_size)

    bar = pbar(units, desc="Part D (prototype allocator)", unit="lang")
    for key, lang in bar:
        bar.set_postfix_str(lang.code)
        if ctx.is_done(key):
            continue
        with Timer(f"part D {lang.code}", log):
            _run_language(lang, ctx, vocab_size, max_types, split)
        ctx.set_status(units_done=len(ctx.done))
    bar.close()
    return aggregate_part_d(ctx)


def _run_language(lang: Language, ctx: RunContext, vocab_size: int, max_types: int,
                  split: str) -> None:
    lexicon, lex_counts = _training_lexicon(lang, max_types)
    if not lexicon:
        log.error("%s: no analyzer lexicon — skipped", lang.code)
        return
    n_multi = sum(1 for v in lexicon.values() if len(v) > 1)
    log.info("%s: %d training types (%.0f%% multi-morph), median count %d",
             lang.code, len(lexicon), 100 * n_multi / max(1, len(lexicon)),
             sorted(lex_counts.values())[len(lex_counts) // 2] if lex_counts else 0)
    counts = _counts_from_corpus(lang)
    counts.update(lex_counts)
    try:
        gold = load_gold(lang)
    except FileNotFoundError:
        gold = []
    # train on FLORES dev, evaluate on devtest: the eval sentences are unseen
    try:
        train_sents = load_split(lang, "dev")
    except FileNotFoundError:
        train_sents = []
    sentences = load_split(lang, split)
    seen_types = set(lexicon)

    # The constrained arm saturates earlier (fewer legal merges), so it is trained
    # first and the baseline is then given *its* realised size: the budget is matched
    # by construction rather than by hope.  morph_full reuses morph_train's vocabulary
    # and adds analyzer pre-tokenization at inference.
    viterbi = _viterbi_segmenter(lang, lexicon)
    budget = {"morph_train": vocab_size, "baseline": vocab_size}
    trained: dict[str, tuple] = {}
    for arm in ("morph_train", "baseline"):
        lines = _training_lines(lexicon, counts, train_sents,
                                segmented=(arm == "morph_train"))
        trained[arm] = train_arm(arm, lines, budget[arm],
                                 ctx.path("tokenizers") / lang.code)
        if arm == "morph_train":
            budget["baseline"] = trained[arm][1]

    for arm in ARMS:
        if arm == "morph_full":
            hf_tok, realised = trained["morph_train"]
            lt = MorphAwareTokenizer(_wrap(hf_tok, arm, realised), lexicon, viterbi)
        else:
            hf_tok, realised = trained[arm]
            lt = _wrap(hf_tok, arm, realised)

        n_tokens = sum(lt.batch_n_tokens(sentences, desc=f"{arm}/{lang.code}"))
        n_words = sum(len(words_whitespace(s)) for s in sentences)
        n_chars = sum(len(s) for s in sentences)
        rows = [score_word(lt, g) for g in
                pbar(gold, desc=f"align {arm}/{lang.code}", unit="word", leave=False)]
        unseen = [r for r in rows if r["word"] not in seen_types]
        if rows:  # keep every per-word decision, as Parts A and B do
            word_dir = ctx.path("words")
            word_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(rows)
            frame["seen_in_training"] = frame.word.isin(seen_types)
            frame.to_parquet(word_dir / f"{lang.code}__{arm}.parquet", index=False)

        record = {
            "language": lang.code,
            "language_name": lang.name,
            "family": lang.family,
            "morph_type": lang.morph_type,
            "arm": arm,
            "vocab_size": vocab_size,
            "vocab_realised": realised,
            "n_training_types": len(lexicon),
            "n_training_lines": len(lines),
            "fertility": n_tokens / max(1, n_words),
            "chars_per_token": n_chars / max(1, n_tokens),
            "bytes_per_token": sum(n_bytes(s) for s in sentences) / max(1, n_tokens),
            "n_gold_words_unseen": len(unseen),
            **({} if not rows else _aggregate_words(rows)),
            **({f"unseen_{k}": v for k, v in _aggregate_words(unseen).items()}
               if unseen else {}),
        }
        ctx.commit(f"{lang.code}::{arm}", record)
        log.info("%s/%s: fertility=%.3f boundary_f1=%.3f integrity=%.3f (unseen F1=%.3f)",
                 lang.code, arm, record["fertility"], record.get("boundary_f1", float("nan")),
                 record.get("morpheme_integrity", float("nan")),
                 record.get("unseen_boundary_f1", float("nan")))
        del lt
    # the trained tokenizers and the Viterbi model are per-language; drop them before
    # the next language is processed
    trained.clear()
    viterbi = None
    gc.collect()


def aggregate_part_d(ctx: RunContext) -> pd.DataFrame:
    records = ctx.records()
    if not records:
        log.warning("Part D: no records to aggregate")
        return pd.DataFrame()
    df = pd.DataFrame(records).drop(columns=["_unit", "_ts"], errors="ignore")
    df = df.drop_duplicates(subset=["language", "arm"], keep="last")
    out = ctx.path("prototype_arms.csv")
    df.to_csv(out, index=False)

    # the comparison table the paper prints: morph arm minus baseline arm
    metrics = [m for m in ["fertility", "boundary_f1", "boundary_f1_annotated",
                           "morpheme_integrity", "token_purity", "chars_per_token",
                           "vocab_realised", "unseen_boundary_f1",
                           "unseen_morpheme_integrity"]
               if m in df.columns]
    wide = df.pivot_table(index=["language", "family", "morph_type"], columns="arm",
                          values=metrics)
    wide.columns = [f"{m}_{a}" for m, a in wide.columns]
    for m in metrics:
        for arm in ("morph_train", "morph_full"):
            if f"{m}_{arm}" in wide.columns and f"{m}_baseline" in wide.columns:
                wide[f"{m}_{arm}_delta"] = wide[f"{m}_{arm}"] - wide[f"{m}_baseline"]
    wide = wide.reset_index()
    wide.to_csv(ctx.path("prototype_comparison.csv"), index=False)
    log.info("Part D summary -> %s (%d languages)",
             paths.relpath(ctx.path("prototype_comparison.csv")), len(wide))
    nan = float("nan")
    for _, r in wide.iterrows():
        log.info("  %-4s | fertility  base %.2f  train %.2f  full %.2f", r.language,
                 r.get("fertility_baseline", nan), r.get("fertility_morph_train", nan),
                 r.get("fertility_morph_full", nan))
        log.info("       | boundaryF1 base %.3f  train %.3f  full %.3f (%+.3f)",
                 r.get("boundary_f1_baseline", nan),
                 r.get("boundary_f1_morph_train", nan),
                 r.get("boundary_f1_morph_full", nan),
                 r.get("boundary_f1_morph_full_delta", nan))
        log.info("       | integrity  base %.3f  train %.3f  full %.3f",
                 r.get("morpheme_integrity_baseline", nan),
                 r.get("morpheme_integrity_morph_train", nan),
                 r.get("morpheme_integrity_morph_full", nan))
    return wide
