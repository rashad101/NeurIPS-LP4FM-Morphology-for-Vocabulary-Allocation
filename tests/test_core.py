"""Unit tests for the pieces where a silent bug would corrupt every result.

Run with::

    uv run --extra dev pytest -q

These cover the metric arithmetic, the character-span machinery, gold-boundary
derivation, prompt budgeting and answer extraction.  The last one exists because a
loose ``[ABCD]`` fallback once matched the "A" inside "ANSWER" and scored every such
Belebele generation as option A.
"""

from __future__ import annotations

from pathlib import Path

from morphtok import paths

paths.setup_env()

from morphtok.data_morph import GoldWord, morphs_to_boundaries  # noqa: E402
from morphtok.part_b_alignment import (  # noqa: E402
    _aggregate_words, _greedy_match, _spans_to_boundaries, score_word)
from morphtok.part_c_capability import (  # noqa: E402
    _extracted_answer, _norm_number, score_prediction)
from morphtok.tokenizer_zoo import _clean_spans, _spans_from_byte_prefixes  # noqa: E402


# --------------------------------------------------------------------------- gold
def test_morphs_to_boundaries():
    assert morphs_to_boundaries(["abc", "de", "f"]) == [3, 5]
    assert morphs_to_boundaries(["whole"]) == []
    assert morphs_to_boundaries([]) == []


def test_gold_word_roundtrip():
    g = GoldWord.from_morphs("running", 3, ["run", "ning"], "A", ["unimorph"])
    assert g.boundaries == [3]
    assert g.n_morphs == 2
    assert "".join(g.morphs) == g.word


# ------------------------------------------------------------------------- spans
def test_clean_spans_covers_word_exactly():
    assert _clean_spans([(0, 2), (2, 5)], "abcde") == [(0, 2), (2, 5)]
    # gaps are absorbed rather than silently dropping characters
    assert _clean_spans([(0, 2), (3, 5)], "abcde") == [(0, 3), (3, 5)]
    # empty spans disappear; coverage is still exact
    assert _clean_spans([(0, 0), (0, 5)], "abcde") == [(0, 5)]


def test_byte_prefix_spans_handle_multibyte():
    word = "हिन्दी"  # every codepoint is 3 bytes in UTF-8
    n_bytes = len(word.encode("utf-8"))
    # one token per byte -> boundaries only where a codepoint completes
    spans, mids = _spans_from_byte_prefixes(word, [1] * n_bytes)
    assert "".join(word[s:e] for s, e in spans) == word
    assert mids > 0  # byte splits inside a character are counted, not invented
    assert len(spans) == len(word)


def test_byte_prefix_spans_single_token():
    spans, mids = _spans_from_byte_prefixes("abc", [3])
    assert spans == [(0, 3)] and mids == 0


# ----------------------------------------------------------------------- metrics
def test_greedy_match_tolerance():
    assert _greedy_match([3, 6], [3, 6], 0) == 2
    assert _greedy_match([3, 6], [4, 7], 0) == 0
    assert _greedy_match([3, 6], [4, 7], 1) == 2
    # one predicted boundary cannot satisfy two gold boundaries
    assert _greedy_match([3, 4], [4], 1) == 1


def test_spans_to_boundaries_excludes_edges():
    assert _spans_to_boundaries([(0, 2), (2, 5)], 5) == [2]
    assert _spans_to_boundaries([(0, 5)], 5) == []


class _FakeTok:
    """Segments a word at fixed character offsets."""

    def __init__(self, cuts):
        self.cuts = cuts

    def word_spans(self, word):
        pts = [0] + [c for c in self.cuts if 0 < c < len(word)] + [len(word)]
        return [(a, b) for a, b in zip(pts, pts[1:])], True, 0

    def tokens_per_word(self, word):
        return len(self.word_spans(word)[0])


def test_score_word_perfect_and_wrong():
    g = GoldWord.from_morphs("runnings", 1, ["run", "ning", "s"], "A", ["x"])
    perfect = score_word(_FakeTok([3, 7]), g, tol=0)
    assert perfect["tp"] == 2 and perfect["n_gold"] == 2 and perfect["n_pred"] == 2
    assert perfect["morphs_intact"] == 3

    # Cutting in the wrong places destroys boundary recall, but a coarse final token
    # still *contains* whole morphemes — integrity and boundary-F1 are deliberately
    # different quantities, and the paper reports both.
    wrong = score_word(_FakeTok([1, 2]), g, tol=0)
    assert wrong["tp"] == 0
    assert wrong["morphs_intact"] == 2  # "ning" and "s" sit inside the (2, 8) token
    assert wrong["n_pred"] == 2


def test_aggregate_words_f1_and_integrity():
    g1 = GoldWord.from_morphs("abcdef", 1, ["abc", "def"], "A", ["unimorph"])
    g2 = GoldWord.from_morphs("xyz", 1, ["xyz"], "D", ["morfessor"])  # monomorphemic
    rows = [score_word(_FakeTok([3]), g1, tol=0), score_word(_FakeTok([1]), g2, tol=0)]
    out = _aggregate_words(rows)
    # 1 correct boundary out of 2 predicted, 1 gold boundary -> P=.5 R=1 F1=.667
    assert abs(out["boundary_precision"] - 0.5) < 1e-9
    assert abs(out["boundary_recall"] - 1.0) < 1e-9
    assert abs(out["boundary_f1"] - 2 / 3) < 1e-9
    assert out["n_monomorphemic"] == 1
    assert out["mono_split_rate"] == 1.0  # the monomorphemic word was split
    # the annotated subset excludes the tier-D word
    assert out["n_gold_words_annotated"] == 1
    assert out["gold_source"] == "unimorph"


# ---------------------------------------------------------------- answer scoring
def test_norm_number_variants():
    assert _norm_number("the answer is 1,200.") == "1200"
    assert _norm_number("१८ apples") == "18"       # Devanagari digits
    assert _norm_number("no digits") is None
    assert _norm_number("first 3 then 7") == "7"   # last number wins


def test_mgsm_scoring():
    assert score_prediction("mgsm", "18", "so the answer is 18.") == 1
    assert score_prediction("mgsm", "18", "the answer is १८") == 1
    assert score_prediction("mgsm", "18", "the answer is 19") == 0
    assert score_prediction("mgsm", "18", "no number") == 0


def test_belebele_letter_not_matched_inside_words():
    # regression: "ANSWER" contains an A and must not be read as option A
    assert _extracted_answer("belebele", "The answer is 2") == "B"
    assert score_prediction("belebele", "B", "The answer is 2") == 1
    assert score_prediction("belebele", "A", "The answer is 2") == 0


def test_belebele_scoring_forms():
    for out in ("B", "b", "B) second", "Answer: B", "The correct answer is (B).",
                "The answer is option 2"):
        assert score_prediction("belebele", "B", out) == 1, out
    for out in ("C", "The answer is D", "", "nothing relevant"):
        assert score_prediction("belebele", "B", out) == 0, out


def test_extracted_answer_matches_score():
    """The stored `extracted` field must be what the score was computed from."""
    for out in ("Answer: C", "The answer is 3", "D)", "", "blah"):
        pred = _extracted_answer("belebele", out)
        assert score_prediction("belebele", "C", out) == int(pred == "C")


# ------------------------------------------------------------------ prompt budget
class _TinyTok:
    """Whitespace tokenizer with the HF call signature used by build_prompt."""

    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("w" for _ in ids)


def test_build_prompt_respects_budget_and_records_truncation():
    from morphtok.data_tasks import Item
    from morphtok.part_c_capability import build_prompt

    item = Item(uid="q", task="mgsm", lang="eng", question="a b c d e", answer="1")
    shots = [Item(uid=f"s{i}", task="mgsm", lang="eng", question="x y z",
                  answer="1", cot="p q r") for i in range(5)]

    unlimited = build_prompt("mgsm", item, shots, _TinyTok(), None, 5, False)
    assert unlimited.n_shots == 5

    tight = build_prompt("mgsm", item, shots, _TinyTok(), 20, 5, False)
    assert tight.n_shots < 5
    assert tight.n_prompt_tokens <= 20

    starved = build_prompt("mgsm", item, shots, _TinyTok(), 3, 5, False)
    assert starved.n_shots == 0
    assert starved.truncated is True
    assert starved.n_prompt_tokens == 3


# ---------------------------------------------------------------------------
# TMPDIR must stay short enough to hold a Unix-domain socket path, or vLLM dies
# with "ZMQError: ipc path ... is longer than 107 characters" before loading a
# single weight (see paths.setup_short_tmpdir).
# ---------------------------------------------------------------------------
def test_tmpdir_is_socket_safe():
    import os
    import socket
    from uuid import uuid4

    tmpdir = paths.setup_short_tmpdir()
    assert paths._tmpdir_fits(tmpdir), f"{tmpdir} leaves no room for a socket name"
    assert os.environ["TMPDIR"] == tmpdir
    assert os.environ["VLLM_RPC_BASE_PATH"] == tmpdir

    # bind the address vLLM would actually use
    addr = f"{tmpdir}/{uuid4()}"
    assert len(addr) <= paths.IPC_PATH_MAX
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(addr)
    finally:
        sock.close()
        try:
            os.unlink(addr)
        except OSError:
            pass
    # and the socket must land inside the repository, not outside it
    assert Path(tmpdir).resolve() == paths.TMP_DIR.resolve()


def test_setup_short_tmpdir_is_idempotent():
    assert paths.setup_short_tmpdir() == paths.setup_short_tmpdir()


def test_part_c_model_resolution():
    import argparse

    from morphtok.cli import DEFAULT_MODELS, resolve_part_c_models

    # two models by default: H3 needs a tokenizer contrast, not one model twice
    assert len(DEFAULT_MODELS) >= 2
    assert resolve_part_c_models(argparse.Namespace(model=None, models=None)) \
        == list(DEFAULT_MODELS)
    # an explicit choice always wins, --models over --model
    assert resolve_part_c_models(argparse.Namespace(model="a/b", models=None)) == ["a/b"]
    assert resolve_part_c_models(argparse.Namespace(model="a/b", models="c/d, e/f")) \
        == ["c/d", "e/f"]


def test_default_models_map_to_audited_tokenizers():
    from morphtok.cli import DEFAULT_MODELS, infer_model_tokenizer_key
    from morphtok.tokenizer_zoo import BY_KEY

    keys = {infer_model_tokenizer_key(m) for m in DEFAULT_MODELS}
    assert None not in keys, "every Part C model needs a zoo tokenizer for H3"
    assert len(keys) == len(DEFAULT_MODELS), "models must contribute distinct tokenizers"
    assert keys <= set(BY_KEY)


def test_stage_lock_rejects_a_second_process(tmp_path):
    """A second run on the same (run_id, stage) must fail loudly, not interleave."""
    import pytest

    from morphtok.state import StageLock

    first = StageLock(tmp_path / ".lock")
    first.acquire()
    second = StageLock(tmp_path / ".lock")
    with pytest.raises(SystemExit) as exc:
        second.acquire()
    assert "already being written" in str(exc.value)
    first.release()
    # once released the directory is claimable again
    third = StageLock(tmp_path / ".lock")
    third.acquire()
    third.release()
