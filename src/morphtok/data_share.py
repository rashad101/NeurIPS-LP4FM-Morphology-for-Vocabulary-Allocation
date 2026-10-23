"""Training-data share proxy.

No lab publishes per-language token counts for the tokenizers we audit, so the H1
regression controls for resource level with a *proxy*: Wikipedia article counts
(live from the Wikimedia API, cached in ``data/processed/wiki_sizes.json``, static
fallback in ``languages.py``).  This is the same proxy family used by the fertility
audits we build on, and the point of the control is only to show boundary-F1
survives it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import requests

from . import paths
from .languages import LANGUAGES, WIKI_ARTICLES_FALLBACK, Language
from .logging_utils import get_logger
from .progress import pbar

log = get_logger("share")

CACHE = paths.PROC_DIR / "wiki_sizes.json"
API = "https://{wiki}.wikipedia.org/w/api.php"


def fetch_wiki_sizes(force: bool = False) -> dict[str, int]:
    if CACHE.is_file() and not force:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
        if data:
            return {k: int(v) for k, v in data.items()}
    sizes: dict[str, int] = {}
    for lang in pbar(LANGUAGES, desc="wikipedia sizes", unit="lang"):
        try:
            r = requests.get(
                API.format(wiki=lang.wiki),
                params={"action": "query", "meta": "siteinfo",
                        "siprop": "statistics", "format": "json"},
                headers={"User-Agent": "morphtok/0.1 (research)"},
                timeout=30,
            )
            r.raise_for_status()
            sizes[lang.wiki] = int(r.json()["query"]["statistics"]["articles"])
        except Exception as exc:
            fallback = WIKI_ARTICLES_FALLBACK.get(lang.wiki)
            log.warning("wiki size for %s failed (%s) — using fallback %s",
                        lang.wiki, exc, fallback)
            if fallback:
                sizes[lang.wiki] = fallback
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    Path(CACHE).write_text(json.dumps(sizes, indent=2), encoding="utf-8")
    log.info("wikipedia article counts cached -> %s", paths.relpath(CACHE))
    return sizes


def share_features(languages: list[Language], force: bool = False) -> dict[str, dict]:
    """Per-language resource features for the regressions."""
    sizes = fetch_wiki_sizes(force=force)
    if not sizes:
        sizes = dict(WIKI_ARTICLES_FALLBACK)
    eng = sizes.get("en", WIKI_ARTICLES_FALLBACK["en"])
    out = {}
    for lang in languages:
        n = sizes.get(lang.wiki, WIKI_ARTICLES_FALLBACK.get(lang.wiki, 1))
        out[lang.code] = {
            "wiki_articles": n,
            "log10_wiki_articles": round(math.log10(max(n, 1)), 4),
            "wiki_share_vs_eng": round(n / max(eng, 1), 6),
            "log10_share_vs_eng": round(math.log10(max(n, 1) / max(eng, 1)), 4),
        }
    return out
