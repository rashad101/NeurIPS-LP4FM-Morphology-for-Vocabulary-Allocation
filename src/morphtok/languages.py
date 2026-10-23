"""Language registry: typology, scripts, and per-resource identifiers.

The typological split is the paper's independent variable of interest:
Indo-Aryan (fusional) vs. Dravidian (agglutinative), with Germanic (analytic) and
Bantu (agglutinative, non-Indic) contrast rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Language:
    code: str  # ISO-639-3, our primary key
    name: str
    flores: str  # FLORES-200 / Belebele code, e.g. hin_Deva
    script: str
    family: str  # Indo-Aryan | Dravidian | Germanic | Uralic | Turkic | ...
    morph_type: str  # fusional | agglutinative | analytic
    wiki: str  # Wikipedia language code (training-share proxy)
    morfessor: str | None = None  # IndicNLP morfessor model stem
    unimorph: str | None = None  # github.com/unimorph/<repo>
    mgsm: str | None = None  # juletxara/mgsm config
    tags: tuple[str, ...] = field(default_factory=tuple)
    sigmorphon: str | None = None  # SIGMORPHON-2022 segmentation code

    @property
    def is_indic(self) -> bool:
        return self.family in {"Indo-Aryan", "Dravidian"}

    @property
    def agglutinative(self) -> int:
        return int(self.morph_type == "agglutinative")

    @property
    def has_annotated_gold(self) -> bool:
        """True when a hand-curated (non-Morfessor) analyzer covers this language."""
        return bool(self.unimorph or self.sigmorphon)


LANGUAGES: tuple[Language, ...] = (
    # ---- anchors / contrasts ------------------------------------------------
    Language("eng", "English", "eng_Latn", "Latin", "Germanic", "analytic", "en",
             None, "eng", "en", ("anchor",), sigmorphon="eng"),
    Language("deu", "German", "deu_Latn", "Latin", "Germanic", "fusional", "de",
             None, "deu", "de", ("contrast",)),
    Language("swh", "Swahili", "swh_Latn", "Latin", "Bantu", "agglutinative", "sw",
             None, "swc", "sw", ("contrast",)),
    # ---- Indo-Aryan (fusional) ---------------------------------------------
    Language("hin", "Hindi", "hin_Deva", "Devanagari", "Indo-Aryan", "fusional", "hi",
             "hi", "hin", None, ("indic", "core")),
    Language("ben", "Bengali", "ben_Beng", "Bengali", "Indo-Aryan", "fusional", "bn",
             "bn", "ben", "bn", ("indic", "core")),
    Language("mar", "Marathi", "mar_Deva", "Devanagari", "Indo-Aryan", "fusional", "mr",
             "mr", "mar", None, ("indic", "core")),
    Language("guj", "Gujarati", "guj_Gujr", "Gujarati", "Indo-Aryan", "fusional", "gu",
             "gu", "guj", None, ("indic",)),
    Language("pan", "Punjabi", "pan_Guru", "Gurmukhi", "Indo-Aryan", "fusional", "pa",
             "pa", "pan", None, ("indic",)),
    Language("ory", "Odia", "ory_Orya", "Odia", "Indo-Aryan", "fusional", "or",
             None, "ory", None, ("indic",)),
    # NOTE: indic_nlp_resources ships morph/morfessor/ur.model as a byte-identical
    # copy of kn.model (Kannada), so there is no usable Urdu analyzer upstream —
    # Urdu gold therefore comes from UniMorph only.  Verified by md5, Jul 2026.
    Language("urd", "Urdu", "urd_Arab", "Arabic", "Indo-Aryan", "fusional", "ur",
             None, "urd", None, ("indic",)),
    Language("asm", "Assamese", "asm_Beng", "Bengali", "Indo-Aryan", "fusional", "as",
             None, "asm", None, ("indic",)),
    # ---- Dravidian (agglutinative) -----------------------------------------
    Language("tam", "Tamil", "tam_Taml", "Tamil", "Dravidian", "agglutinative", "ta",
             "ta", "tam", None, ("indic", "core")),
    Language("tel", "Telugu", "tel_Telu", "Telugu", "Dravidian", "agglutinative", "te",
             "te", "tel", "te", ("indic", "core")),
    Language("kan", "Kannada", "kan_Knda", "Kannada", "Dravidian", "agglutinative", "kn",
             "kn", "kan", None, ("indic", "core")),
    Language("mal", "Malayalam", "mal_Mlym", "Malayalam", "Dravidian", "agglutinative", "ml",
             "ml", "mal", None, ("indic", "core")),
    # ---- typological control set (outside Indic, with *annotated* gold) ------
    # These exist to answer the obvious objection to H1: that the agglutinative
    # slope is an artifact of Indic scripts, Indic resource levels, or of gold
    # segmentations produced by an unsupervised analyzer.  Every language here has
    # hand-curated morpheme segmentations (SIGMORPHON-2022) or UniMorph inflection
    # tables, and the agglutinative/fusional contrast is replicated across three
    # further families.
    Language("hun", "Hungarian", "hun_Latn", "Latin", "Uralic", "agglutinative", "hu",
             None, "hun", None, ("control",), sigmorphon="hun"),
    Language("tur", "Turkish", "tur_Latn", "Latin", "Turkic", "agglutinative", "tr",
             None, "tur", None, ("control",)),
    Language("khk", "Mongolian (Halh)", "khk_Cyrl", "Cyrillic", "Mongolic",
             "agglutinative", "mn", None, None, None, ("control",), sigmorphon="mon"),
    Language("ces", "Czech", "ces_Latn", "Latin", "Slavic", "fusional", "cs",
             None, "ces", None, ("control",), sigmorphon="ces"),
    Language("rus", "Russian", "rus_Cyrl", "Cyrillic", "Slavic", "fusional", "ru",
             None, "rus", None, ("control",), sigmorphon="rus"),
    Language("spa", "Spanish", "spa_Latn", "Latin", "Romance", "fusional", "es",
             None, "spa", None, ("control",), sigmorphon="spa"),
)

BY_CODE = {lang.code: lang for lang in LANGUAGES}
BY_FLORES = {lang.flores: lang for lang in LANGUAGES}

#: Part A — full audit set (English + 12 Indic + contrasts + typological controls).
AUDIT_LANGS = [lang.code for lang in LANGUAGES]

#: Part B — every language with any gold source.
MORPH_LANGS = [lang.code for lang in LANGUAGES
               if lang.morfessor or lang.unimorph or lang.sigmorphon]

#: Part B core five from the spec (2 Indo-Aryan + 3 Dravidian).
MORPH_CORE_LANGS = ["hin", "mar", "tam", "tel", "kan"]

#: Non-Indic typological controls with hand-curated gold.
CONTROL_LANGS = [lang.code for lang in LANGUAGES if "control" in lang.tags]

#: Languages whose gold does not depend on an unsupervised analyzer.
ANNOTATED_LANGS = [lang.code for lang in LANGUAGES if lang.has_annotated_gold]

#: Part C — capability study.
MGSM_LANGS = [lang.code for lang in LANGUAGES if lang.mgsm]
BELEBELE_LANGS = [lang.code for lang in LANGUAGES]


def get(code: str) -> Language:
    if code in BY_CODE:
        return BY_CODE[code]
    if code in BY_FLORES:
        return BY_FLORES[code]
    raise KeyError(f"unknown language {code!r}; known: {sorted(BY_CODE)}")


def resolve(codes: list[str] | str | None, default: list[str]) -> list[Language]:
    """Resolve a CLI language selection.

    Accepts ISO-639-3 codes, FLORES codes, or the group names ``all``, ``indic``,
    ``core``, ``morph``, ``mgsm``, ``belebele``.
    """
    if codes is None:
        selected = default
    else:
        if isinstance(codes, str):
            codes = [c for c in codes.replace(",", " ").split() if c]
        selected = []
        for c in codes:
            key = c.strip().lower()
            if key == "all":
                selected += AUDIT_LANGS
            elif key == "indic":
                selected += [lang.code for lang in LANGUAGES if lang.is_indic]
            elif key == "core":
                selected += MORPH_CORE_LANGS
            elif key == "morph":
                selected += MORPH_LANGS
            elif key == "control":
                selected += CONTROL_LANGS
            elif key == "annotated":
                selected += ANNOTATED_LANGS
            elif key == "typology":  # Indic + non-Indic controls
                selected += [c for c in MORPH_LANGS] + CONTROL_LANGS
            elif key == "mgsm":
                selected += MGSM_LANGS
            elif key == "belebele":
                selected += BELEBELE_LANGS
            else:
                selected.append(get(key).code)
    out: list[Language] = []
    seen: set[str] = set()
    for code in selected:
        if code not in seen:
            seen.add(code)
            out.append(BY_CODE[code])
    return out


# ---------------------------------------------------------------------------
# training-data share proxy
# ---------------------------------------------------------------------------
# Fallback Wikipedia article counts (rounded, mid-2025 snapshot of
# meta.wikimedia.org/wiki/List_of_Wikipedias). Used only if the live Wikimedia
# API is unreachable; `morphtok fetch-data` refreshes them into
# data/processed/wiki_sizes.json.
WIKI_ARTICLES_FALLBACK = {
    "en": 6_990_000,
    "de": 3_010_000,
    "sw": 82_000,
    "hi": 163_000,
    "bn": 155_000,
    "mr": 96_000,
    "gu": 30_000,
    "pa": 51_000,
    "or": 17_000,
    "ur": 216_000,
    "as": 12_000,
    "ta": 166_000,
    "te": 96_000,
    "kn": 34_000,
    "ml": 89_000,
    "hu": 545_000,
    "tr": 649_000,
    "mn": 22_000,
    "cs": 570_000,
    "ru": 2_050_000,
    "es": 2_040_000,
}
