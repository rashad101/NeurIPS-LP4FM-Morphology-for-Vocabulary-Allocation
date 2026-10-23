"""The tokenizer zoo: mainstream 2025-26 tokenizers, Indic-specialized ones, and a
byte-level reference — behind one interface that can report *character spans* for a
word, which is what morpheme-boundary alignment needs.

All weights/vocab files download into ``models/hf-hub`` (see ``paths.setup_env``).

Gated repos (Llama-3.1, Gemma-3, Aya) resolve to ungated community mirrors of the
*same vocabulary* when no ``HF_TOKEN`` is available; each attempt is logged so the
paper can state exactly which artifact produced each row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .logging_utils import get_logger

log = get_logger("tokenizers")

BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
SPM_MARK = "▁"  # ▁
GPT2_SPACE = "Ġ"  # Ġ


@dataclass(frozen=True)
class TokenizerSpec:
    key: str  # short id used in filenames/tables
    name: str  # display name for the paper
    kind: str  # hf | tiktoken | bytes
    repos: tuple[str, ...] = ()  # first is canonical, rest are ungated mirrors
    group: str = "mainstream"  # mainstream | specialized | byte
    year: str = ""
    note: str = ""
    default: bool = True

    @property
    def canonical(self) -> str:
        return self.repos[0] if self.repos else self.key


ZOO: tuple[TokenizerSpec, ...] = (
    # ---------------- mainstream ------------------------------------------------
    TokenizerSpec("o200k", "GPT-4o / o200k_base", "tiktoken", ("o200k_base",),
                  "mainstream", "2024", "OpenAI frontier BPE, 200k"),
    TokenizerSpec("llama31", "Llama-3.1", "hf",
                  ("meta-llama/Llama-3.1-8B-Instruct",
                   "NousResearch/Meta-Llama-3.1-8B-Instruct",
                   "unsloth/Meta-Llama-3.1-8B-Instruct"),
                  "mainstream", "2024", "128k byte-level BPE"),
    TokenizerSpec("gemma3", "Gemma-3", "hf",
                  ("google/gemma-3-12b-it", "unsloth/gemma-3-12b-it", "unsloth/gemma-3-4b-it"),
                  "mainstream", "2025", "262k SentencePiece"),
    TokenizerSpec("qwen3", "Qwen3", "hf", ("Qwen/Qwen3-8B",),
                  "mainstream", "2025", "151k byte-level BPE"),
    TokenizerSpec("mistral_nemo", "Mistral-NeMo", "hf",
                  ("mistralai/Mistral-Nemo-Instruct-2407",),
                  "mainstream", "2024", "131k Tekken"),
    TokenizerSpec("aya_expanse", "Aya-Expanse", "hf",
                  ("CohereLabs/aya-expanse-8b", "CohereForAI/aya-expanse-8b"),
                  "mainstream", "2024", "multilingual-first, 256k"),
    TokenizerSpec("deepseek_v3", "DeepSeek-V3", "hf", ("deepseek-ai/DeepSeek-V3",),
                  "mainstream", "2025", "129k byte-level BPE"),
    # ---------------- Indic-specialized (the attainable frontier) --------------
    TokenizerSpec("sarvam1", "Sarvam-1", "hf", ("sarvamai/sarvam-1",),
                  "specialized", "2024", "Indic-first, 68k, ~2 tok/word target"),
    # Sarvam-M is an Indic-adapted *model* that inherits the Mistral vocabulary
    # unchanged (verified: identical encodings on FLORES; only 6 special tokens
    # renamed).  It is therefore audited as mainstream — it is this paper's cleanest
    # evidence that adapting a model is not adapting its vocabulary.
    TokenizerSpec("sarvam_m", "Sarvam-M (Mistral vocabulary)", "hf",
                  ("sarvamai/sarvam-m",),
                  "mainstream", "2025", "Indic-adapted model, unchanged Mistral vocab"),
    TokenizerSpec("muril", "MuRIL", "hf", ("google/muril-base-cased",),
                  "specialized", "2021", "197k WordPiece over 17 Indian languages"),
    TokenizerSpec("indicbert_v2", "IndicBERT-v2", "hf", ("ai4bharat/IndicBERTv2-MLM-only",),
                  "specialized", "2023", "AI4Bharat, 250k, 24 languages"),
    # ---------------- byte-level reference -------------------------------------
    TokenizerSpec("byt5", "ByT5 (byte-level)", "hf", ("google/byt5-small",),
                  "byte", "2021", "token == UTF-8 byte"),
    TokenizerSpec("utf8", "UTF-8 bytes", "bytes", (),
                  "byte", "-", "analytic byte reference (no model)", default=False),
)

BY_KEY = {s.key: s for s in ZOO}
DEFAULT_KEYS = [s.key for s in ZOO if s.default]


def resolve_specs(keys: Sequence[str] | str | None) -> list[TokenizerSpec]:
    """Resolve a CLI tokenizer selection (``all``, ``default``, ``mainstream``,
    ``specialized``, ``byte``, or explicit keys)."""
    if keys is None:
        selected = list(DEFAULT_KEYS)
    else:
        if isinstance(keys, str):
            keys = [k for k in keys.replace(",", " ").split() if k]
        selected = []
        for k in keys:
            key = k.strip()
            if key in ("all",):
                selected += [s.key for s in ZOO]
            elif key in ("default",):
                selected += DEFAULT_KEYS
            elif key in ("mainstream", "specialized", "byte"):
                selected += [s.key for s in ZOO if s.group == key]
            elif key in BY_KEY:
                selected.append(key)
            else:
                raise SystemExit(f"unknown tokenizer {key!r}; known: {sorted(BY_KEY)}")
    out, seen = [], set()
    for k in selected:
        if k not in seen:
            seen.add(k)
            out.append(BY_KEY[k])
    return out


# ---------------------------------------------------------------------------
# unified tokenizer wrapper
# ---------------------------------------------------------------------------
@dataclass
class LoadedTokenizer:
    spec: TokenizerSpec
    source: str  # repo/encoding actually used
    backend: object
    n_vocab: int
    _kind: str
    _offsets_ok: bool = True
    #: how character spans are derived: bytes | tiktoken | hf.  Byte-level models
    #: (ByT5) encode through HF but must be spanned byte-wise, not by piece matching.
    span_mode: str = ""
    stats: dict = field(default_factory=lambda: {"unalignable_words": 0, "words_seen": 0})

    def __post_init__(self) -> None:
        if not self.span_mode:
            self.span_mode = "bytes" if self.spec.group == "byte" else self._kind

    # -- basic encoding -----------------------------------------------------
    def encode(self, text: str) -> list[int]:
        if self._kind == "tiktoken":
            return self.backend.encode(text, disallowed_special=())  # type: ignore[attr-defined]
        if self._kind == "bytes":
            return list(text.encode("utf-8"))
        return self.backend(text, add_special_tokens=False)["input_ids"]  # type: ignore[operator]

    def n_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def batch_n_tokens(self, texts: Sequence[str], chunk: int = 128,
                       desc: str | None = None) -> list[int]:
        """Token counts for many strings, batched where the backend supports it."""
        from .progress import pbar

        out: list[int] = []
        chunks = [texts[i : i + chunk] for i in range(0, len(texts), chunk)]
        bar = pbar(chunks, desc=desc or f"encode {self.spec.key}", unit="batch",
                   leave=False) if len(chunks) > 1 else chunks
        for part in bar:
            if self._kind == "hf":
                try:
                    ids = self.backend(list(part), add_special_tokens=False)["input_ids"]  # type: ignore[operator]
                    out += [len(x) for x in ids]
                    continue
                except Exception:  # pragma: no cover - slow tokenizer edge cases
                    pass
            elif self._kind == "tiktoken":
                try:
                    ids = self.backend.encode_ordinary_batch(list(part))  # type: ignore[attr-defined]
                    out += [len(x) for x in ids]
                    continue
                except Exception:  # pragma: no cover
                    pass
            out += [self.n_tokens(t) for t in part]
        return out

    def pieces(self, text: str) -> list[str]:
        """Human-readable token strings (for the qualitative overlay figure)."""
        if self._kind == "tiktoken":
            ids = self.encode(text)
            return [
                self.backend.decode_single_token_bytes(i).decode("utf-8", errors="replace")  # type: ignore[attr-defined]
                for i in ids
            ]
        if self.span_mode == "bytes":
            return [f"<0x{b:02X}>" for b in text.encode("utf-8")]
        ids = self.encode(text)
        toks = self.backend.convert_ids_to_tokens(ids)  # type: ignore[attr-defined]
        return [t.replace(SPM_MARK, "▁").replace(GPT2_SPACE, "·") for t in toks]

    def byte_fallback_rate(self, text: str) -> float:
        """Fraction of tokens that are raw ``<0xXX>`` byte-fallback pieces."""
        if self.span_mode == "bytes":
            return 1.0
        try:
            toks = self.pieces(text)
        except Exception:  # pragma: no cover - defensive
            return float("nan")
        if not toks:
            return 0.0
        return sum(bool(BYTE_FALLBACK_RE.match(t)) for t in toks) / len(toks)

    # -- character spans (the Part B primitive) -----------------------------
    def word_spans(self, word: str) -> tuple[list[tuple[int, int]], bool, int]:
        """Segment a single word into character spans.

        Returns ``(spans, aligned, mid_char_splits)``:
        ``spans``  non-empty, ordered, covering the word exactly (when aligned);
        ``aligned`` False if the surface form could not be reconstructed;
        ``mid_char_splits`` byte-level splits that fall inside a codepoint
        (byte-level BPE fragmenting a single Indic character).
        """
        self.stats["words_seen"] += 1
        if self.span_mode == "bytes":
            n_tok = max(1, len(self.encode(word)))
            per_token = len(word.encode("utf-8")) / n_tok
            lens = [1] * len(word.encode("utf-8")) if per_token <= 1.0 else None
            if lens is None:  # byte-level model that merges bytes (should not happen)
                lens = [round(per_token)] * n_tok
            spans, mids = _spans_from_byte_prefixes(word, lens)
            return spans, True, mids
        if self.span_mode == "tiktoken":
            ids = self.encode(word)
            lens = [len(self.backend.decode_single_token_bytes(i)) for i in ids]  # type: ignore[attr-defined]
            spans, mids = _spans_from_byte_prefixes(word, lens)
            return spans, True, mids
        # --- HF ------------------------------------------------------------
        spans = _hf_offsets(self.backend, word) if self._offsets_ok else None
        if spans is None:
            spans = _hf_reconstruct(self.backend, word)
        if spans is None:
            self.stats["unalignable_words"] += 1
            return [(0, len(word))], False, 0
        return spans, True, 0

    def tokens_per_word(self, word: str) -> int:
        return len(self.encode(word))


def _spans_from_byte_prefixes(word: str, byte_lens: Iterable[int]) -> tuple[list[tuple[int, int]], int]:
    """Convert per-token byte lengths into character spans.

    A boundary that falls inside a multi-byte codepoint cannot be a character
    boundary; it is counted as a ``mid_char_split`` and collapsed away.
    """
    raw = word.encode("utf-8")
    cum = 0
    mid = 0
    boundaries: list[int] = []
    for blen in byte_lens:
        cum += blen
        prefix = raw[:cum]
        try:
            boundaries.append(len(prefix.decode("utf-8")))
        except UnicodeDecodeError:
            mid += 1
            boundaries.append(boundaries[-1] if boundaries else 0)
    spans: list[tuple[int, int]] = []
    start = 0
    for b in boundaries:
        b = min(max(b, 0), len(word))
        if b > start:
            spans.append((start, b))
            start = b
    if start < len(word):
        spans.append((start, len(word)))
    if not spans:
        spans = [(0, len(word))]
    return spans, mid


def _clean_spans(spans: list[tuple[int, int]], word: str) -> list[tuple[int, int]] | None:
    """Drop empty/duplicate spans, verify exact coverage of ``word``."""
    out: list[tuple[int, int]] = []
    for s, e in spans:
        s, e = max(0, min(s, len(word))), max(0, min(e, len(word)))
        if e <= s:
            continue
        if out and s < out[-1][1]:  # overlapping -> unusable
            s = out[-1][1]
            if e <= s:
                continue
        if out and s > out[-1][1]:  # gap (dropped characters) -> absorb
            out[-1] = (out[-1][0], s)
        out.append((s, e))
    if not out:
        return None
    if out[0][0] != 0:
        out[0] = (0, out[0][1])
    if out[-1][1] != len(word):
        out[-1] = (out[-1][0], len(word))
    if "".join(word[s:e] for s, e in out) != word:
        return None
    return out


def _hf_offsets(tok, word: str) -> list[tuple[int, int]] | None:
    try:
        enc = tok(word, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
    except Exception:
        return None
    if not offsets:
        return None
    return _clean_spans([(int(s), int(e)) for s, e in offsets], word)


def _hf_reconstruct(tok, word: str) -> list[tuple[int, int]] | None:
    """Offset-free fallback: rebuild spans by matching cleaned pieces in order."""
    try:
        ids = tok(word, add_special_tokens=False)["input_ids"]
        pieces = tok.convert_ids_to_tokens(ids)
    except Exception:
        return None
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        text = _piece_surface(tok, piece)
        if not text:
            continue
        idx = word.find(text, cursor)
        if idx < 0:
            return None
        if idx > cursor and spans:  # skipped characters -> extend previous span
            spans[-1] = (spans[-1][0], idx)
        elif idx > cursor:
            spans.append((cursor, idx))
        spans.append((idx, idx + len(text)))
        cursor = idx + len(text)
    return _clean_spans(spans, word)


def _piece_surface(tok, piece: str) -> str:
    """Best-effort surface string of a single vocabulary piece."""
    if piece is None:
        return ""
    if BYTE_FALLBACK_RE.match(piece):
        return ""  # single raw byte: cannot be placed at char level
    special = set(getattr(tok, "all_special_tokens", []) or [])
    if piece in special:
        return ""
    text = piece
    if text.startswith("##"):
        text = text[2:]
    text = text.replace(SPM_MARK, "").replace(GPT2_SPACE, "")
    try:
        text = text.encode("latin-1").decode("utf-8")  # gpt-2 byte map
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text.strip()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
class TokenizerUnavailable(RuntimeError):
    pass


#: Explicit caches (not ``lru_cache``) so a tokenizer can be *evicted*: Parts A/B
#: hold one tokenizer at a time, which matters when the same box is also running vLLM.
_LOADED: dict[str, "LoadedTokenizer"] = {}
_VOCAB_SURFACES: dict[str, tuple[str, ...]] = {}
_SCRIPT_ALLOC: dict[tuple[str, str], dict] = {}


def loaded_keys() -> list[str]:
    return sorted(_LOADED)


def unload(key: str) -> None:
    """Drop a tokenizer and everything derived from it, then collect."""
    import gc

    _LOADED.pop(key, None)
    _VOCAB_SURFACES.pop(key, None)
    for k in [k for k in _SCRIPT_ALLOC if k[0] == key]:
        _SCRIPT_ALLOC.pop(k, None)
    gc.collect()
    log.debug("unloaded tokenizer %s (%d still resident)", key, len(_LOADED))


def unload_all() -> None:
    for key in list(_LOADED):
        unload(key)


def load(key: str) -> LoadedTokenizer:
    """Load a tokenizer by zoo key, trying canonical repo then ungated mirrors."""
    if key in _LOADED:
        return _LOADED[key]
    lt = _load_uncached(key)
    _LOADED[key] = lt
    return lt


def _load_uncached(key: str) -> LoadedTokenizer:
    spec = BY_KEY[key]
    if spec.kind == "bytes":
        return LoadedTokenizer(spec, "utf-8", object(), 256, "bytes")
    if spec.kind == "tiktoken":
        import tiktoken

        enc_name = spec.repos[0]
        log.info("[%s] loading tiktoken encoding %s", spec.key, enc_name)
        enc = tiktoken.get_encoding(enc_name)
        return LoadedTokenizer(spec, enc_name, enc, enc.n_vocab, "tiktoken")

    from transformers import AutoTokenizer

    errors: list[str] = []
    for repo in spec.repos:
        try:
            log.info("[%s] loading HF tokenizer %s", spec.key, repo)
            tok = AutoTokenizer.from_pretrained(repo, use_fast=True, trust_remote_code=False)
        except Exception as exc:  # gated, missing, or conversion failure
            msg = f"{repo}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            log.warning("[%s] unavailable -> %s", spec.key, msg)
            errors.append(msg)
            continue
        n_vocab = int(getattr(tok, "vocab_size", 0) or len(tok))
        probe = _hf_offsets(tok, "abcdef")
        offsets_ok = probe is not None and len(probe) >= 1
        lt = LoadedTokenizer(spec, repo, tok, n_vocab, "hf", offsets_ok)
        log.info(
            "[%s] ready source=%s vocab=%d fast=%s offsets=%s",
            spec.key, repo, n_vocab, getattr(tok, "is_fast", False), offsets_ok,
        )
        return lt
    raise TokenizerUnavailable(f"{spec.key}: none of {spec.repos} could be loaded: {errors}")


def load_many(specs: Sequence[TokenizerSpec]) -> tuple[dict[str, LoadedTokenizer], dict[str, str]]:
    """Load all specs; return ``(loaded, skipped_reasons)`` — never raises."""
    loaded: dict[str, LoadedTokenizer] = {}
    skipped: dict[str, str] = {}
    for spec in specs:
        try:
            loaded[spec.key] = load(spec.key)
        except Exception as exc:
            skipped[spec.key] = str(exc)
            log.error("SKIP tokenizer %s: %s", spec.key, exc)
    if skipped:
        log.warning(
            "%d/%d tokenizers skipped (%s). Set HF_TOKEN or write .hf_token to include gated ones.",
            len(skipped), len(specs), ", ".join(skipped),
        )
    return loaded, skipped


def _vocab_surfaces(key: str) -> tuple[str, ...]:
    """Decoded surface string of every vocabulary entry (computed once per tokenizer)."""
    if key in _VOCAB_SURFACES:
        return _VOCAB_SURFACES[key]
    lt = load(key)
    pieces: list[str] = []
    if lt._kind == "tiktoken":
        for i in range(lt.n_vocab):
            try:
                pieces.append(
                    lt.backend.decode_single_token_bytes(i)  # type: ignore[attr-defined]
                    .decode("utf-8", errors="ignore")
                )
            except Exception:
                pieces.append("")
    else:
        try:
            vocab = lt.backend.get_vocab()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            return ()
        pieces = [_piece_surface(lt.backend, p) for p in vocab]
    _VOCAB_SURFACES[key] = tuple(pieces)
    return _VOCAB_SURFACES[key]


def script_allocation(key: str, script: str) -> dict:
    """How much of a tokenizer's vocabulary is actually spent on one script.

    This is the control for the standard objection to H2 — that Indic-specialized
    tokenizers win simply by spending a larger share of a fixed budget on fewer
    languages.  Regressions can then ask whether morphological alignment survives
    *after* per-script budget is held constant.
    """
    from .textutil import SCRIPT_RANGES, is_mostly_script

    if (key, script) in _SCRIPT_ALLOC:
        return _SCRIPT_ALLOC[(key, script)]
    ranges = SCRIPT_RANGES.get(script)
    lt = load(key)
    if not ranges or lt.span_mode == "bytes":
        return {"script_vocab_count": float("nan"), "script_vocab_share": float("nan")}

    pieces = _vocab_surfaces(key)
    if not pieces:
        return {"script_vocab_count": float("nan"), "script_vocab_share": float("nan")}

    n = sum(1 for p in pieces if p and is_mostly_script(p, ranges))
    total = max(1, lt.n_vocab)
    log.debug("[%s] %s vocabulary allocation: %d/%d (%.2f%%)",
              key, script, n, total, 100 * n / total)
    _SCRIPT_ALLOC[(key, script)] = {"script_vocab_count": n,
                                    "script_vocab_share": n / total}
    return _SCRIPT_ALLOC[(key, script)]


def describe(loaded: dict[str, LoadedTokenizer]) -> list[dict]:
    return [
        {
            "tokenizer": lt.spec.key,
            "name": lt.spec.name,
            "group": lt.spec.group,
            "year": lt.spec.year,
            "source": lt.source,
            "vocab_size": lt.n_vocab,
            "note": lt.spec.note,
        }
        for lt in loaded.values()
    ]
