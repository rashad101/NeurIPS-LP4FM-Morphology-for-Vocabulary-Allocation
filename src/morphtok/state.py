"""Run bookkeeping, crash-safe incremental writes, and resume support.

Every stage writes results as it goes:

``runs/<run_id>/<stage>/records.jsonl``  one JSON object per finished work unit
``runs/<run_id>/<stage>/_done.txt``      unit keys already finished (resume index)
``runs/<run_id>/<stage>/config.json``    exact configuration of the stage
``runs/<run_id>/<stage>/*.parquet|csv``  aggregated tables (rewritten each flush)

Re-running the same ``--run-id`` skips finished units unless ``--overwrite``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import paths
from .logging_utils import get_logger, timestamp

log = get_logger("state")

LATEST_POINTER = "latest"


# ---------------------------------------------------------------------------
# small io helpers
# ---------------------------------------------------------------------------
def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("corrupt json %s — ignoring", paths.relpath(path))
        return default


def read_jsonl(path: Path) -> list[dict]:
    """Tolerant JSONL reader: a truncated final line (killed mid-write) is dropped."""
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("dropping malformed line %d of %s", i, paths.relpath(path))
    return out


class JsonlWriter:
    """Append-only JSONL writer that fsyncs, so a killed process loses nothing."""

    def __init__(self, path: Path, fsync_every: int = 1):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._n = 0
        self._fsync_every = max(1, fsync_every)

    def append(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._n += 1
        if self._n % self._fsync_every == 0:
            self.flush()

    def extend(self, records: Iterable[dict]) -> None:
        for r in records:
            self.append(r)
        self.flush()

    def flush(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class StageLock:
    """Exclusive advisory lock on one ``runs/<run_id>/<stage>`` directory.

    Two processes on the same (run_id, stage) is not a harmless duplicate: both
    append to the same ``records.jsonl`` and the same
    ``predictions/*.jsonl``, so interleaved partial lines can corrupt results that
    the whole resume machinery exists to protect — and on a GPU stage they also race
    for the same card and OOM each other.  This makes the second one fail loudly and
    immediately instead.

    ``flock`` is used rather than a bare PID file so the lock is released by the
    kernel on any exit, including SIGKILL, leaving no stale lock to clean up.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def acquire(self) -> None:
        import fcntl

        fh = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            holder = fh.read().strip() or "unknown process"
            fh.close()
            raise SystemExit(
                f"{paths.relpath(self.path.parent)} is already being written by "
                f"{holder}.\nWait for it to finish, or stop it, or use a different "
                "--run-id / --tag. Two processes on one stage corrupt its records."
            )
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid {os.getpid()} since {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()  # closing drops the flock
            finally:
                self._fh = None
            try:
                self.path.unlink()
            except OSError:
                pass


class DoneSet:
    """Durable set of completed unit keys (one key per line)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: set[str] = set()
        if self.path.is_file():
            self._keys = {
                ln.strip() for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()
            }

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str) -> None:
        if key in self._keys:
            return
        self._keys.add(key)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(key + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def clear(self) -> None:
        self._keys.clear()
        if self.path.is_file():
            self.path.unlink()

    @property
    def keys(self) -> set[str]:
        return set(self._keys)


# ---------------------------------------------------------------------------
# run context
# ---------------------------------------------------------------------------
def make_run_id(tag: str | None = None) -> str:
    """Time-separable run id: ``20260730-141530`` or ``20260730-141530__tag``."""
    rid = timestamp()
    if tag:
        safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in tag)
        rid = f"{rid}__{safe}"
    return rid


def resolve_run_id(run_id: str | None, tag: str | None = None) -> str:
    """``None`` -> new timestamped id; ``latest`` -> most recent existing run."""
    if run_id in (None, "", "new"):
        return make_run_id(tag)
    if run_id == LATEST_POINTER:
        rid = latest_run_id()
        if rid is None:
            raise SystemExit("no previous run found in runs/ — omit --run-id to start a new one")
        return rid
    return run_id


def latest_run_id() -> str | None:
    pointer = paths.RUNS_DIR / "LATEST"
    if pointer.is_file():
        rid = pointer.read_text(encoding="utf-8").strip()
        if rid and (paths.RUNS_DIR / rid).is_dir():
            return rid
    candidates = sorted(
        (d for d in paths.RUNS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    return candidates[-1].name if candidates else None


def list_runs() -> list[dict]:
    rows: list[dict] = []
    if not paths.RUNS_DIR.is_dir():
        return rows
    for d in sorted(paths.RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        stages = {}
        for sd in sorted(d.iterdir()):
            if sd.is_dir():
                done = sd / "_done.txt"
                n_done = len(DoneSet(done)) if done.is_file() else 0
                status = read_json(sd / "status.json", {}) or {}
                stages[sd.name] = {"units_done": n_done, **status}
        rows.append(
            {
                "run_id": d.name,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.stat().st_mtime)),
                "stages": stages,
            }
        )
    return rows


@dataclass
class RunContext:
    """Handle for one (run_id, stage) pair."""

    run_id: str
    stage: str
    overwrite: bool = False
    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dir = paths.run_dir(self.run_id, self.stage)
        # take the lock before touching anything in the directory
        self.lock = StageLock(self.dir / ".lock")
        self.lock.acquire()
        self.records_path = self.dir / "records.jsonl"
        self.done = DoneSet(self.dir / "_done.txt")
        if self.overwrite:
            log.warning("--overwrite: clearing previous state in %s", paths.relpath(self.dir))
            self.done.clear()
            if self.records_path.is_file():
                bak = self.records_path.with_suffix(f".jsonl.bak-{timestamp()}")
                os.replace(self.records_path, bak)
        write_json(self.dir / "config.json", self.config)
        (paths.RUNS_DIR / "LATEST").write_text(self.run_id, encoding="utf-8")
        self._writer: JsonlWriter | None = None
        self.t0 = time.time()
        log.info(
            "run=%s stage=%s dir=%s resumable_units_done=%d",
            self.run_id,
            self.stage,
            paths.relpath(self.dir),
            len(self.done),
        )

    # -- record io ----------------------------------------------------------
    @property
    def writer(self) -> JsonlWriter:
        if self._writer is None:
            self._writer = JsonlWriter(self.records_path)
        return self._writer

    def is_done(self, key: str) -> bool:
        return (not self.overwrite) and key in self.done

    def commit(self, key: str, record: dict) -> None:
        """Persist one finished unit (record + done-marker) immediately."""
        record = {"_unit": key, "_ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        self.writer.append(record)
        self.done.add(key)

    def records(self) -> list[dict]:
        return read_jsonl(self.records_path)

    def path(self, name: str) -> Path:
        return self.dir / name

    def set_status(self, **kw: Any) -> None:
        status = read_json(self.dir / "status.json", {}) or {}
        status.update(kw)
        status["elapsed_s"] = round(time.time() - self.t0, 1)
        status["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(self.dir / "status.json", status)

    def close(self, **status: Any) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.set_status(**status)
        self.lock.release()


def iter_pending(units: list[tuple[str, Any]], ctx: RunContext) -> Iterator[tuple[str, Any]]:
    """Yield only units not yet finished, logging how many were skipped."""
    pending = [(k, v) for k, v in units if not ctx.is_done(k)]
    skipped = len(units) - len(pending)
    if skipped:
        log.info("resume: skipping %d/%d already-finished units", skipped, len(units))
    yield from pending
