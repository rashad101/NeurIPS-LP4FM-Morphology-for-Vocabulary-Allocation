"""Repo-local Noto fonts so Indic scripts render in figures instead of tofu boxes.

Fonts land in ``data/resources/fonts`` and are registered with matplotlib's font
manager at runtime.  Note that matplotlib does not do complex-script shaping
(no HarfBuzz): conjuncts and reordered vowel signs are drawn glyph-by-glyph, which
is legible but not typographically correct.  ``figures.py`` therefore also emits an
HTML version of the morpheme-overlay panel, which the browser shapes properly.
"""

from __future__ import annotations

from . import paths
from .download import fetch_text
from .logging_utils import get_logger

log = get_logger("fonts")

FONT_DIR = paths.RESOURCES_DIR / "fonts"
BASE = ("https://github.com/notofonts/notofonts.github.io/raw/main/fonts/"
        "{fam}/hinted/ttf/{fam}-Regular.ttf")

# script -> Noto family
SCRIPT_FONTS = {
    "Devanagari": "NotoSansDevanagari",
    "Bengali": "NotoSansBengali",
    "Gujarati": "NotoSansGujarati",
    "Gurmukhi": "NotoSansGurmukhi",
    "Odia": "NotoSansOriya",
    "Tamil": "NotoSansTamil",
    "Telugu": "NotoSansTelugu",
    "Kannada": "NotoSansKannada",
    "Malayalam": "NotoSansMalayalam",
    "Arabic": "NotoSansArabic",
}


def ensure_fonts(scripts: list[str] | None = None, force: bool = False) -> list[str]:
    """Download the Noto fonts for the given scripts; return local file paths."""
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = scripts or list(SCRIPT_FONTS)
    files: list[str] = []
    for script in wanted:
        fam = SCRIPT_FONTS.get(script)
        if not fam:
            continue
        dest = FONT_DIR / f"{fam}-Regular.ttf"
        try:
            fetch_text(BASE.format(fam=fam), dest, force=force)
            files.append(str(dest))
        except Exception as exc:
            log.warning("font %s unavailable (%s) — %s may render as boxes", fam, exc, script)
    return files


def register_matplotlib() -> list[str]:
    """Register every downloaded font and return the family names, most specific first."""
    from matplotlib import font_manager

    names: list[str] = []
    for ttf in sorted(FONT_DIR.glob("*.ttf")):
        try:
            font_manager.fontManager.addfont(str(ttf))
            names.append(font_manager.FontProperties(fname=str(ttf)).get_name())
        except Exception as exc:  # pragma: no cover
            log.warning("could not register %s: %s", ttf.name, exc)
    if names:
        log.info("registered %d Indic fonts: %s", len(names), ", ".join(sorted(set(names))))
    return names
