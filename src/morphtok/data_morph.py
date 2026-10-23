"""Gold morpheme segmentations — the paper's released artifact.

Two independent analyzers, combined into confidence tiers:

* **SIGMORPHON-2022 Morpheme Segmentation** (`sigmorphon/2022SegmentationST`):
  hand-curated full morpheme segmentations for eng, hun, ces, rus, spa, mon.  Only
  surface-preserving rows are kept (the shared task also contains canonical
  segmentations whose morphs do not concatenate back to the surface form).
* **UniMorph** inflection tables: a (lemma, form) pair yields a stem/affix boundary
  from the longest common prefix — a hand-curated morphological boundary.
* **IndicNLP Morfessor** (Kunchukuttan's ``indic_nlp_resources``): pretrained
  Morfessor-Baseline segmentation lexicons for hi, mr, bn, gu, pa, ta, te, kn, ml.
  A lexicon entry *is* an analyzer decision for that word type — but Morfessor is
  *unsupervised* and frequency-driven, so it is the weakest of the three sources and
  is tiered accordingly.

Tiers (recorded per word, reported as analyzer coverage in the paper):

| tier | meaning                                                            |
|------|--------------------------------------------------------------------|
| A    | two independent analyzers agree on ≥1 boundary (highest confidence) |
| B    | annotated source alone (SIGMORPHON segmentation or UniMorph affix)  |
| C    | Morfessor multi-morph segmentation only (all morphs ≥2 chars)       |
| D    | Morfessor says monomorphemic (zero gold boundaries; precision)      |

Tiers **A+B** are the *annotated* subset that does not depend on an unsupervised
analyzer; ``analysis.py`` re-estimates H1 on that subset alone, because "you measured
agreement between two unsupervised segmenters" is the first objection this analysis
has to survive.

Output: ``data/gold/<code>.gold.jsonl`` + ``data/gold/coverage.csv``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from . import paths
from .data_flores import load_split
from .download import fetch_text
from .languages import Language
from .logging_utils import get_logger
from .progress import pbar
from .textutil import SCRIPT_RANGES, is_mostly_script, word_type_counts

log = get_logger("morph")

MORFESSOR_URL = (
    "https://raw.githubusercontent.com/anoopkunchukuttan/indic_nlp_resources/"
    "master/morph/morfessor/{stem}.model"
)
UNIMORPH_URL = "https://raw.githubusercontent.com/unimorph/{repo}/master/{repo}"
SIGMORPHON_URL = (
    "https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/"
    "{code}.word.{split}.tsv"
)
SIGMORPHON_SPLITS = ("train", "dev", "test.gold")

MORF_DIR = paths.RESOURCES_DIR / "morfessor"
UNIMORPH_DIR = paths.RESOURCES_DIR / "unimorph"
SIGMORPHON_DIR = paths.RESOURCES_DIR / "sigmorphon"

TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass
class GoldWord:
    word: str
    freq: int
    morphs: list[str]
    boundaries: list[int]  # character offsets, exclusive of 0 and len(word)
    tier: str
    sources: list[str]
    n_morphs: int

    @staticmethod
    def from_morphs(word: str, freq: int, morphs: list[str], tier: str,
                    sources: list[str]) -> "GoldWord":
        return GoldWord(
            word=word,
            freq=freq,
            morphs=list(morphs),
            boundaries=morphs_to_boundaries(morphs),
            tier=tier,
            sources=sources,
            n_morphs=len(morphs),
        )


def morphs_to_boundaries(morphs: list[str]) -> list[int]:
    out, cum = [], 0
    for m in morphs[:-1]:
        cum += len(m)
        out.append(cum)
    return out


# ---------------------------------------------------------------------------
# resource acquisition
# ---------------------------------------------------------------------------
def fetch_resources(languages: list[Language], force: bool = False) -> dict[str, dict]:
    """Download analyzer resources for the requested languages."""
    for d in (MORF_DIR, UNIMORPH_DIR, SIGMORPHON_DIR):
        d.mkdir(parents=True, exist_ok=True)
    status: dict[str, dict] = {}
    for lang in pbar(languages, desc="fetch morphology resources", unit="lang"):
        entry: dict[str, str | None] = {"morfessor": None, "unimorph": None,
                                        "sigmorphon": None}
        if lang.morfessor:
            dest = MORF_DIR / f"{lang.morfessor}.model"
            try:
                fetch_text(MORFESSOR_URL.format(stem=lang.morfessor), dest, force=force)
                entry["morfessor"] = str(paths.relpath(dest))
            except Exception as exc:
                log.warning("%s: morfessor model unavailable (%s)", lang.code, exc)
        if lang.unimorph:
            dest = UNIMORPH_DIR / lang.unimorph
            try:
                fetch_text(UNIMORPH_URL.format(repo=lang.unimorph), dest, force=force)
                entry["unimorph"] = str(paths.relpath(dest))
            except Exception as exc:
                log.warning("%s: unimorph table unavailable (%s)", lang.code, exc)
        if lang.sigmorphon:
            got = []
            for split in SIGMORPHON_SPLITS:
                dest = SIGMORPHON_DIR / f"{lang.sigmorphon}.{split}.tsv"
                try:
                    fetch_text(SIGMORPHON_URL.format(code=lang.sigmorphon, split=split),
                               dest, force=force)
                    if dest.stat().st_size > 0:
                        got.append(str(paths.relpath(dest)))
                except Exception as exc:
                    log.debug("%s: sigmorphon %s unavailable (%s)", lang.code, split, exc)
            if got:
                entry["sigmorphon"] = ";".join(got)
            else:
                log.warning("%s: no SIGMORPHON segmentation splits available", lang.code)
        status[lang.code] = entry
    return status


# ---------------------------------------------------------------------------
# analyzer readers
# ---------------------------------------------------------------------------
def read_morfessor_counts(path: Path, script: str | None = None,
                          ) -> dict[str, tuple[list[str], int]]:
    """Same as :func:`read_morfessor_lexicon` but keeps the corpus count column.

    The count is the analyzer's training frequency for that word type — the natural
    weighting for anything that trains on the lexicon (see ``part_d_prototype``).
    """
    out: dict[str, tuple[list[str], int]] = {}
    if not path.is_file():
        return out
    if script and not _lexicon_script_ok(path, script):
        log.error("%s does not contain %s text — resource rejected",
                  paths.relpath(path), script)
        return out
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            head, _, body = line.partition(" ")
            if not body or not head.isdigit():
                continue
            morphs = [m for m in (p.strip() for p in body.split(" + ")) if m]
            if not morphs:
                continue
            out["".join(morphs)] = (morphs, int(head))
    log.info("morfessor %s: %d word types with counts", paths.relpath(path), len(out))
    return out


def read_morfessor_lexicon(path: Path, wanted: set[str] | None = None,
                           script: str | None = None) -> dict[str, list[str]]:
    """Parse a Morfessor ``--save-segmentation`` file into ``word -> morphs``.

    ``script`` enables a sanity check on the resource itself: upstream
    ``indic_nlp_resources`` ships at least one mislabelled model (``ur.model`` is a
    copy of ``kn.model``), and silently scoring Urdu against Kannada segmentations
    would be worse than having no analyzer.
    """
    lex: dict[str, list[str]] = {}
    if not path.is_file():
        return lex
    if script and not _lexicon_script_ok(path, script):
        log.error("%s does not contain %s text — resource rejected",
                  paths.relpath(path), script)
        return lex
    n_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_lines += 1
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            body = parts[1] if parts[0].isdigit() else line
            morphs = [m for m in (p.strip() for p in body.split(" + ")) if m]
            if not morphs:
                continue
            word = "".join(morphs)
            if wanted is not None and word not in wanted:
                continue
            lex[word] = morphs
    log.info("morfessor %s: %d lines -> %d relevant entries",
             paths.relpath(path), n_lines, len(lex))
    return lex


def _lexicon_script_ok(path: Path, script: str, n_sample: int = 2000,
                       min_share: float = 0.5, stride: int = 97) -> bool:
    """Is this lexicon actually written in ``script``?

    Sampling is strided across the whole file: these lexicons are sorted, so the
    first thousands of entries are punctuation and Latin transliterations in every
    language and a head-only sample would reject every valid resource.
    """
    ranges = SCRIPT_RANGES.get(script)
    if not ranges:
        return True
    hits = seen = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i % stride or line.startswith("#") or " " not in line:
                continue
            word = "".join(p.strip() for p in line.split(" ", 1)[1].split(" + "))
            if len(word) < 3:
                continue
            seen += 1
            hits += is_mostly_script(word, ranges)
            if seen >= n_sample:
                break
    share = hits / max(1, seen)
    log.debug("%s: %.0f%% of sampled entries are %s", path.name, 100 * share, script)
    return share >= min_share


def read_unimorph_boundaries(path: Path, wanted: set[str] | None = None,
                             min_stem: int = 3) -> dict[str, list[str]]:
    """Derive stem/affix segmentations from UniMorph (lemma, form) pairs.

    Only single-boundary, whitespace-free, prefix-preserving pairs are kept —
    the high-precision subset (suffixing morphology, which is what Indo-Aryan and
    Dravidian inflection overwhelmingly is).
    """
    seg: dict[str, list[str]] = {}
    if not path.is_file():
        return seg
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            lemma, form = cols[0].strip(), cols[1].strip()
            rows += 1
            if not lemma or not form or " " in form or " " in lemma:
                continue
            if wanted is not None and form not in wanted:
                continue
            p = _lcp(lemma, form)
            if p < min_stem or p >= len(form):
                continue
            candidate = [form[:p], form[p:]]
            if len(candidate[1]) < 1:
                continue
            prev = seg.get(form)
            # prefer the longer stem (most conservative boundary) for stability
            if prev is None or len(prev[0]) < p:
                seg[form] = candidate
    log.info("unimorph %s: %d rows -> %d derived segmentations",
             paths.relpath(path), rows, len(seg))
    return seg


def read_sigmorphon_segmentations(lang_code: str, wanted: set[str] | None = None,
                                  ) -> dict[str, list[str]]:
    """Read SIGMORPHON-2022 morpheme segmentations (``word \\t m1 @@m2 @@m3 \\t cat``).

    The shared task mixes surface and canonical segmentations; only rows whose morphs
    concatenate back to the surface form can carry character offsets, so the rest are
    dropped and the retention rate is logged (it is part of the coverage report).
    """
    seg: dict[str, list[str]] = {}
    rows = kept = 0
    for split in SIGMORPHON_SPLITS:
        path = SIGMORPHON_DIR / f"{lang_code}.{split}.tsv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2:
                    continue
                rows += 1
                word = cols[0].strip()
                morphs = [m.strip() for m in cols[1].split(" @@") if m.strip()]
                if not word or len(morphs) < 1 or " " in word:
                    continue
                if wanted is not None and word not in wanted:
                    continue
                if "".join(morphs) != word:  # canonical, not surface — unusable
                    continue
                kept += 1
                seg[word] = morphs
    if rows:
        log.info("sigmorphon %s: %d rows -> %d surface-preserving segmentations "
                 "(%d matched the corpus)", lang_code, rows, kept, len(seg))
    return seg


def _lcp(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def train_morfessor(word_counts: Counter, max_types: int = 50_000) -> dict[str, list[str]]:
    """Fallback analyzer: train Morfessor Baseline on the corpus word types."""
    import morfessor

    types = [(w, c) for w, c in word_counts.most_common(max_types)]
    log.info("training Morfessor Baseline on %d word types", len(types))
    model = morfessor.BaselineModel()
    # morfessor expects (count, atoms) pairs; atoms == the characters of the word
    model.load_data([(c, tuple(w)) for w, c in types])
    model.train_batch(algorithm="recursive", max_epochs=4)
    out = {}
    for word, _ in pbar(types, desc="morfessor segment", unit="w", leave=False):
        morphs, _ = model.viterbi_segment(word)
        out[word] = list(morphs)
    return out


# ---------------------------------------------------------------------------
# gold construction
# ---------------------------------------------------------------------------
def gold_path(lang: Language) -> Path:
    return paths.GOLD_DIR / f"{lang.code}.gold.jsonl"


def build_gold(
    lang: Language,
    n_types: int = 2000,
    min_word_len: int = 4,
    min_freq: int = 1,
    max_monomorphemic_frac: float = 0.25,
    splits: tuple[str, ...] = ("dev", "devtest"),
    train_fallback: bool = False,
    force: bool = False,
) -> tuple[list[GoldWord], dict]:
    """Build (or load) the gold segmentation set for one language."""
    out_path = gold_path(lang)
    if out_path.is_file() and not force:
        gold = load_gold(lang)
        report = json.loads((paths.GOLD_DIR / f"{lang.code}.coverage.json").read_text())
        log.info("gold cached: %s (%d words, tiers=%s)",
                 lang.code, len(gold), report.get("tier_counts"))
        return gold, report

    texts: list[str] = []
    for split in splits:
        try:
            texts += load_split(lang, split)
        except FileNotFoundError:
            log.warning("%s: missing FLORES split %s", lang.code, split)
    counts = word_type_counts(texts, min_len=min_word_len)
    ranges = SCRIPT_RANGES.get(lang.script)
    candidates = Counter(
        {w: c for w, c in counts.items()
         if c >= min_freq and len(w) >= min_word_len and is_mostly_script(w, ranges)}
    )
    log.info("%s: %d candidate word types from %d sentences",
             lang.code, len(candidates), len(texts))
    wanted = set(candidates)

    morf = {}
    if lang.morfessor:
        morf = read_morfessor_lexicon(MORF_DIR / f"{lang.morfessor}.model", wanted,
                                      script=lang.script)
    morf_source = "pretrained-indicnlp" if morf else "none"
    if not morf and train_fallback:
        morf = train_morfessor(candidates)
        morf_source = "trained-on-flores-types" if morf else "none"
    uni = {}
    if lang.unimorph:
        uni = read_unimorph_boundaries(UNIMORPH_DIR / lang.unimorph, wanted)
    sig = {}
    if lang.sigmorphon:
        sig = read_sigmorphon_segmentations(lang.sigmorphon, wanted)

    tiered: dict[str, list[GoldWord]] = {"A": [], "B": [], "C": [], "D": []}
    for word, freq in candidates.items():
        m, u, s = morf.get(word), uni.get(word), sig.get(word)
        analyses = {"morfessor": m, "unimorph": u, "sigmorphon": s}
        present = {k: v for k, v in analyses.items() if v}
        bsets = {k: set(morphs_to_boundaries(v)) for k, v in present.items()}

        # tier A: two independent analyzers agree on at least one boundary
        agreeing = [
            (k1, k2) for i, k1 in enumerate(present) for k2 in list(present)[i + 1:]
            if bsets[k1] & bsets[k2]
        ]
        if agreeing:
            # prefer the annotated analysis as the gold segmentation itself
            best = (s if s else (u if u else m))
            src = sorted({k for pair in agreeing for k in pair})
            tiered["A"].append(GoldWord.from_morphs(word, freq, best, "A", src))
            continue
        # tier B: an annotated source on its own (SIGMORPHON preferred over UniMorph)
        if s:
            tiered["B"].append(GoldWord.from_morphs(word, freq, s, "B", ["sigmorphon"]))
            continue
        if u:
            tiered["B"].append(GoldWord.from_morphs(word, freq, u, "B", ["unimorph"]))
            continue
        if m:
            if len(m) >= 2 and min(len(x) for x in m) >= 2:
                tiered["C"].append(GoldWord.from_morphs(word, freq, m, "C", ["morfessor"]))
            elif len(m) == 1:
                tiered["D"].append(GoldWord.from_morphs(word, freq, m, "D", ["morfessor"]))

    for tier in tiered:
        tiered[tier].sort(key=lambda g: -g.freq)

    # segmenting words fill the budget first; monomorphemic words (which carry the
    # boundary-precision signal) take at most `max_monomorphemic_frac` of the set.
    mono_budget = int(round(n_types * max_monomorphemic_frac))
    multi_budget = n_types - mono_budget
    multi: list[GoldWord] = []
    for tier in ("A", "B", "C"):
        multi += tiered[tier]
    selected = multi[:multi_budget]
    selected += tiered["D"][: min(mono_budget, n_types - len(selected))]
    if len(selected) < n_types:  # spare capacity -> more segmenting words
        selected += multi[multi_budget : multi_budget + (n_types - len(selected))]
    selected.sort(key=lambda g: (TIER_ORDER[g.tier], -g.freq))

    report = {
        "language": lang.code,
        "name": lang.name,
        "family": lang.family,
        "morph_type": lang.morph_type,
        "n_sentences": len(texts),
        "n_candidate_types": len(candidates),
        "analyzer_coverage_morfessor": round(len(morf) / max(1, len(candidates)), 4),
        "analyzer_coverage_unimorph": round(len(uni) / max(1, len(candidates)), 4),
        "analyzer_coverage_sigmorphon": round(len(sig) / max(1, len(candidates)), 4),
        "analyzer_coverage_annotated": round(
            len({*uni, *sig}) / max(1, len(candidates)), 4),
        "analyzer_coverage_any": round(
            len({*morf, *uni, *sig}) / max(1, len(candidates)), 4),
        "tier_counts_available": {t: len(v) for t, v in tiered.items()},
        "tier_counts": dict(Counter(g.tier for g in selected)),
        "n_gold": len(selected),
        "n_gold_with_boundary": sum(1 for g in selected if g.boundaries),
        # the annotated subset (tiers A+B) is what H1's robustness check runs on
        "n_gold_annotated": sum(1 for g in selected if g.tier in ("A", "B")),
        "n_gold_annotated_with_boundary": sum(
            1 for g in selected if g.tier in ("A", "B") and g.boundaries),
        "mean_morphs_per_word": round(
            sum(g.n_morphs for g in selected) / max(1, len(selected)), 3),
        "sources": {"morfessor": bool(morf), "unimorph": bool(uni),
                    "sigmorphon": bool(sig)},
        "morfessor_source": morf_source,
    }

    paths.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for g in selected:
            fh.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")
    (paths.GOLD_DIR / f"{lang.code}.coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "%s: gold=%d (tiers %s, annotated=%d) coverage: morfessor=%.2f unimorph=%.2f "
        "sigmorphon=%.2f -> %s",
        lang.code, len(selected), report["tier_counts"], report["n_gold_annotated"],
        report["analyzer_coverage_morfessor"], report["analyzer_coverage_unimorph"],
        report["analyzer_coverage_sigmorphon"], paths.relpath(out_path),
    )
    if not selected:
        log.error("%s: NO gold segmentations built — language will be absent from Part B",
                  lang.code)
    return selected, report


def load_gold(lang: Language) -> list[GoldWord]:
    path = gold_path(lang)
    if not path.is_file():
        raise FileNotFoundError(f"{paths.relpath(path)} missing — run `morphtok gold`")
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(GoldWord(**json.loads(line)))
    return out
