"""Uniform tqdm progress bars (single place to tune style / disable)."""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Iterator

from tqdm.auto import tqdm

_DISABLED = os.environ.get("MORPHTOK_NO_PROGRESS", "").lower() in {"1", "true", "yes"}
_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def pbar(
    iterable: Iterable[Any] | None = None,
    desc: str = "",
    total: int | None = None,
    unit: str = "it",
    leave: bool = True,
    position: int | None = None,
) -> tqdm:
    return tqdm(
        iterable,
        desc=desc[:38].ljust(38) if desc else None,
        total=total,
        unit=unit,
        leave=leave,
        position=position,
        dynamic_ncols=True,
        bar_format=_BAR,
        disable=_DISABLED,
        file=sys.stderr,
        mininterval=0.3,
        smoothing=0.1,
    )


def track(iterable: Iterable[Any], desc: str = "", **kw: Any) -> Iterator[Any]:
    """``for x in track(items, 'desc'):`` with automatic total detection."""
    total = kw.pop("total", None)
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    yield from pbar(iterable, desc=desc, total=total, **kw)


def write(msg: str) -> None:
    tqdm.write(msg, file=sys.stderr)
