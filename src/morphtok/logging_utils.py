"""File + console logging that plays nicely with tqdm progress bars."""

from __future__ import annotations

import logging
import os
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from . import paths

_CONFIGURED = False
LOG_FILE: Path | None = None

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class TqdmHandler(logging.StreamHandler):
    """Console handler that writes through ``tqdm.write`` so bars stay intact."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - io
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_logging(
    stage: str = "main",
    run_id: str | None = None,
    level: str = "INFO",
    log_file: Path | None = None,
) -> Path:
    """Configure root logging: everything to a file, INFO+ to the console.

    One log file per (run, stage) plus an always-appended ``logs/morphtok.log``
    so a whole session's activity is recoverable from a single file.
    """
    global _CONFIGURED, LOG_FILE
    paths.ensure_dirs()

    if log_file is None:
        name = f"{run_id + '__' if run_id else ''}{stage}__{timestamp()}.log"
        log_file = paths.LOGS_DIR / name

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if _CONFIGURED:
        for h in list(root.handlers):
            root.removeHandler(h)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    session = logging.FileHandler(paths.LOGS_DIR / "morphtok.log", encoding="utf-8")
    session.setLevel(logging.INFO)
    session.setFormatter(formatter)
    root.addHandler(session)

    ch = TqdmHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # third-party noise
    for noisy in ("urllib3", "filelock", "matplotlib", "datasets", "fsspec", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Noto fonts ship no bold face; the fallback warning would flood the log
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    _CONFIGURED = True
    LOG_FILE = log_file

    log = logging.getLogger("morphtok.session")
    log.info("=" * 100)
    log.info("stage=%s run_id=%s log=%s", stage, run_id, paths.relpath(log_file))
    log.info(
        "host=%s platform=%s python=%s pid=%d",
        socket.gethostname(),
        platform.platform(),
        platform.python_version(),
        os.getpid(),
    )
    log.info("repo_root=%s", paths.ROOT)
    log.info("cmd=%s", " ".join(sys.argv))
    return log_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"morphtok.{name}")


def log_env_summary() -> None:
    log = get_logger("env")
    keys = [
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TIKTOKEN_CACHE_DIR",
        "TORCH_HOME",
        "VLLM_CACHE_ROOT",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "TMPDIR",
    ]
    for k in keys:
        log.info("%-22s = %s", k, paths.relpath(os.environ.get(k, "<unset>")))
    if paths.TMPDIR_NOTE:
        log.info("%-22s : %s", "TMPDIR note", paths.TMPDIR_NOTE)
    log.info("HF_TOKEN present: %s", bool(os.environ.get("HF_TOKEN")))


class Timer:
    """``with Timer('part A', log):`` -> logs wall time on exit."""

    def __init__(self, what: str, log: logging.Logger | None = None):
        self.what = what
        self.log = log or get_logger("timer")
        self.t0 = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.t0 = time.time()
        self.log.info("START  %s", self.what)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.time() - self.t0
        if exc_type is None:
            self.log.info("DONE   %s in %s", self.what, fmt_duration(self.elapsed))
        else:
            self.log.error(
                "FAILED %s after %s: %s: %s",
                self.what,
                fmt_duration(self.elapsed),
                exc_type.__name__,
                exc,
            )


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
