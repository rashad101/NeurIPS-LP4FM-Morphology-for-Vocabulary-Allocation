"""Resumable downloads and archive extraction — always into repo-local paths."""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

import requests

from . import paths
from .logging_utils import get_logger
from .progress import pbar

log = get_logger("download")

_UA = {"User-Agent": "morphtok/0.1 (research; tokenizer audit)"}
_CHUNK = 1 << 20


def stream_download(url: str, dest: Path, desc: str | None = None, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` with a progress bar; skip if already complete."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    desc = desc or dest.name

    head_size = None
    try:
        h = requests.head(url, headers=_UA, timeout=30, allow_redirects=True)
        if h.ok and "content-length" in h.headers:
            head_size = int(h.headers["content-length"])
    except requests.RequestException:
        pass

    if dest.is_file() and not force:
        if head_size is None or dest.stat().st_size == head_size:
            log.info("cached  %s (%.1f MB)", paths.relpath(dest), dest.stat().st_size / 1e6)
            return dest
        log.warning(
            "size mismatch for %s (%d != %d) — re-downloading",
            paths.relpath(dest), dest.stat().st_size, head_size,
        )

    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("GET %s -> %s", url, paths.relpath(dest))
    with requests.get(url, headers=_UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or head_size
        with tmp.open("wb") as fh, pbar(
            total=total, desc=f"download {desc}", unit="B"
        ) as bar:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    os.replace(tmp, dest)
    log.info("saved   %s (%.1f MB)", paths.relpath(dest), dest.stat().st_size / 1e6)
    return dest


def fetch_text(url: str, dest: Path, force: bool = False) -> Path:
    """Download a small text file (no progress bar noise)."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=_UA, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    log.info("saved   %s (%.2f MB)", paths.relpath(dest), len(r.content) / 1e6)
    return dest


def extract_archive(archive: Path, out_dir: Path, marker: str | None = None) -> Path:
    """Extract tar/zip with a progress bar; skip when ``marker`` already exists."""
    archive, out_dir = Path(archive), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if marker and (out_dir / marker).exists():
        log.info("already extracted: %s", paths.relpath(out_dir / marker))
        return out_dir

    if archive.suffixes[-1:] == [".zip"]:
        with zipfile.ZipFile(archive) as zf:
            members = zf.namelist()
            for m in pbar(members, desc=f"extract {archive.name}", unit="file"):
                zf.extract(m, out_dir)
    else:
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            for m in pbar(members, desc=f"extract {archive.name}", unit="file"):
                tf.extract(m, out_dir, filter="data")
    log.info("extracted %s -> %s", archive.name, paths.relpath(out_dir))
    return out_dir


def hf_snapshot_file(repo_id: str, filename: str, repo_type: str = "dataset") -> Path:
    """Download one file from the HF hub into the repo-local hub cache."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(p)
