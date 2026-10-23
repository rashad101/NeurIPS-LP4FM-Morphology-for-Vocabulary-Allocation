"""FLORES-200 parallel text loader (repo-local, ungated by default).

Source order:
1. ``--flores-dir`` / ``data/raw/flores200_dataset`` if already present;
2. the official NLLB tarball ``dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz``
   (no HF account needed) — the default;
3. HF ``openlanguagedata/flores_plus`` per-language JSONL (needs ``HF_TOKEN``;
   FLORES+ is the maintained successor and fixes some FLORES-200 typos).

Parsed corpora are cached as ``data/processed/flores/<split>/<flores_code>.txt``,
so Parts A/B never re-download or re-parse.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import paths
from .download import extract_archive, hf_snapshot_file, stream_download
from .languages import Language
from .logging_utils import get_logger
from .progress import pbar

log = get_logger("flores")

TARBALL_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
HF_REPO = "openlanguagedata/flores_plus"
SPLITS = ("dev", "devtest")

RAW_ROOT = paths.RAW_DIR / "flores200"
PROC_ROOT = paths.PROC_DIR / "flores"


def _dataset_dir(explicit: str | Path | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        RAW_ROOT / "flores200_dataset",
        paths.RAW_DIR / "flores200_dataset",
    ]
    for c in candidates:
        if (c / "devtest").is_dir():
            return c
    return None


def ensure_flores(languages: list[Language], explicit_dir: str | Path | None = None,
                  force: bool = False) -> str:
    """Make sure every requested language is available under ``data/processed/flores``.

    Returns the provenance string recorded in run configs.
    """
    PROC_ROOT.mkdir(parents=True, exist_ok=True)
    missing = [
        lang for lang in languages
        if not all(_proc_path(split, lang).is_file() for split in SPLITS)
    ]
    if not missing and not force:
        log.info("FLORES cache complete for %d languages", len(languages))
        return _provenance()

    source = "cache"
    ddir = _dataset_dir(explicit_dir)
    if ddir is None:
        try:
            tar = stream_download(TARBALL_URL, RAW_ROOT / "flores200_dataset.tar.gz",
                                  desc="flores200", force=force)
            extract_archive(tar, RAW_ROOT, marker="flores200_dataset/devtest")
            ddir = _dataset_dir(explicit_dir)
            source = "nllb-tarball"
        except Exception as exc:
            log.warning("FLORES-200 tarball unavailable (%s); falling back to HF FLORES+", exc)

    if ddir is not None:
        _ingest_tarball(ddir, missing)
        source = source if source != "cache" else "local-dir"
    else:
        _ingest_hf(missing)
        source = "hf:openlanguagedata/flores_plus"

    still = [lang.code for lang in languages
             if not all(_proc_path(s, lang).is_file() for s in SPLITS)]
    if still:
        raise SystemExit(
            f"FLORES text missing for {still}. Provide --flores-dir with a local "
            f"flores200_dataset/ or set HF_TOKEN for {HF_REPO}."
        )
    prov = f"{source}"
    (PROC_ROOT / "PROVENANCE.txt").write_text(prov + "\n", encoding="utf-8")
    log.info("FLORES ready (source=%s)", prov)
    return prov


def _provenance() -> str:
    p = PROC_ROOT / "PROVENANCE.txt"
    return p.read_text(encoding="utf-8").strip() if p.is_file() else "cache"


def _proc_path(split: str, lang: Language) -> Path:
    return PROC_ROOT / split / f"{lang.flores}.txt"


def _write_proc(split: str, lang: Language, lines: list[str]) -> None:
    out = _proc_path(split, lang)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ln.strip() for ln in lines) + "\n", encoding="utf-8")


def _ingest_tarball(ddir: Path, languages: list[Language]) -> None:
    for lang in pbar(languages, desc="flores ingest (tarball)", unit="lang"):
        for split in SPLITS:
            src = ddir / split / f"{lang.flores}.{split}"
            if not src.is_file():
                log.warning("missing %s", paths.relpath(src))
                continue
            lines = src.read_text(encoding="utf-8").splitlines()
            _write_proc(split, lang, lines)
            log.debug("%s/%s: %d sentences", split, lang.flores, len(lines))


def _ingest_hf(languages: list[Language]) -> None:
    for lang in pbar(languages, desc="flores ingest (HF flores+)", unit="lang"):
        for split in SPLITS:
            try:
                fp = hf_snapshot_file(HF_REPO, f"{split}/{lang.flores}.jsonl")
            except Exception as exc:
                log.error("HF FLORES+ %s/%s failed: %s", split, lang.flores, exc)
                continue
            rows = []
            with Path(fp).open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            rows.sort(key=lambda r: int(r.get("id", 0)))
            _write_proc(split, lang, [r["text"] for r in rows])


def load_split(lang: Language, split: str = "devtest") -> list[str]:
    path = _proc_path(split, lang)
    if not path.is_file():
        raise FileNotFoundError(
            f"{paths.relpath(path)} not found — run `morphtok fetch-data` first"
        )
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_parallel(languages: list[Language], split: str = "devtest") -> dict[str, list[str]]:
    """Sentence-aligned corpora keyed by language code (FLORES lines are aligned)."""
    out = {lang.code: load_split(lang, split) for lang in languages}
    sizes = {k: len(v) for k, v in out.items()}
    if len(set(sizes.values())) > 1:
        log.warning("FLORES splits are not equal length: %s", sizes)
    return out
