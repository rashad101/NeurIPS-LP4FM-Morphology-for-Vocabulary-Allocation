"""Part C evaluation data: MGSM (math CoT) and Belebele (reading comprehension).

Why both: MGSM — the task the European tokenizer-tax study used, so H3's replication
claim is like-for-like — exists for only 3 of our languages (en, bn, te) plus the
sw/de contrasts.  Belebele covers **all** 12 Indic languages with the identical
passage set, which is what the H3 regression needs to have enough (language ×
condition) cells to separate fertility from boundary-F1.

Everything is downloaded through the repo-local HF cache; parsed items are cached as
``data/processed/tasks/<task>/<lang>.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .download import hf_snapshot_file
from .languages import Language
from .logging_utils import get_logger
from .progress import pbar

log = get_logger("tasks")

MGSM_REPO = "juletxara/mgsm"
BELEBELE_REPO = "facebook/belebele"
TASK_DIR = paths.PROC_DIR / "tasks"


@dataclass
class Item:
    uid: str
    task: str
    lang: str
    question: str
    answer: str  # gold answer as a string ("18", "B", ...)
    context: str = ""  # passage (Belebele)
    choices: list[str] = field(default_factory=list)
    cot: str = ""  # exemplar rationale (MGSM train)

    def to_json(self) -> dict:
        return self.__dict__.copy()


def _cache_path(task: str, lang: Language, split: str) -> Path:
    return TASK_DIR / task / f"{lang.code}.{split}.json"


def _save(items: list[Item], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([i.to_json() for i in items], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def _load(path: Path) -> list[Item]:
    return [Item(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------
# MGSM
# ---------------------------------------------------------------------------
def load_mgsm(lang: Language, split: str = "test", force: bool = False) -> list[Item]:
    if not lang.mgsm:
        return []
    cache = _cache_path("mgsm", lang, split)
    if cache.is_file() and not force:
        return _load(cache)
    import pandas as pd

    fname = f"{lang.mgsm}/{split}-00000-of-00001.parquet"
    fp = hf_snapshot_file(MGSM_REPO, fname)
    df = pd.read_parquet(fp)
    items: list[Item] = []
    for i, row in enumerate(df.to_dict("records")):
        ans = row.get("answer_number")
        cot = row.get("answer") or ""
        if ans is None and cot:
            ans = str(cot).rsplit("####", 1)[-1].strip()
        items.append(
            Item(
                uid=f"mgsm-{lang.code}-{split}-{i:04d}",
                task="mgsm",
                lang=lang.code,
                question=str(row["question"]).strip(),
                answer=str(ans).strip(),
                cot=str(cot).strip(),
            )
        )
    _save(items, cache)
    log.info("mgsm %s/%s: %d items -> %s", lang.code, split, len(items), paths.relpath(cache))
    return items


# ---------------------------------------------------------------------------
# Belebele
# ---------------------------------------------------------------------------
LETTERS = ["A", "B", "C", "D"]


def load_belebele(lang: Language, force: bool = False) -> list[Item]:
    cache = _cache_path("belebele", lang, "test")
    if cache.is_file() and not force:
        return _load(cache)
    fp = hf_snapshot_file(BELEBELE_REPO, f"data/{lang.flores}.jsonl")
    rows = []
    with Path(fp).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    items: list[Item] = []
    for i, row in enumerate(rows):
        choices = [str(row[f"mc_answer{k}"]).strip() for k in range(1, 5)]
        gold_idx = int(row["correct_answer_num"]) - 1
        items.append(
            Item(
                uid=f"belebele-{lang.code}-{i:04d}",
                task="belebele",
                lang=lang.code,
                question=str(row["question"]).strip(),
                answer=LETTERS[gold_idx],
                context=str(row["flores_passage"]).strip(),
                choices=choices,
            )
        )
    _save(items, cache)
    log.info("belebele %s: %d items -> %s", lang.code, len(items), paths.relpath(cache))
    return items


# ---------------------------------------------------------------------------
def ensure_tasks(languages: list[Language], tasks: list[str], force: bool = False) -> dict:
    """Pre-download all Part C data (so the GPU stage never waits on the network)."""
    summary: dict[str, dict[str, int]] = {}
    for lang in pbar(languages, desc="fetch task data", unit="lang"):
        info: dict[str, int] = {}
        if "mgsm" in tasks and lang.mgsm:
            info["mgsm_test"] = len(load_mgsm(lang, "test", force=force))
            info["mgsm_train"] = len(load_mgsm(lang, "train", force=force))
        if "belebele" in tasks:
            try:
                info["belebele"] = len(load_belebele(lang, force=force))
            except Exception as exc:
                log.error("belebele %s failed: %s", lang.code, exc)
        summary[lang.code] = info
    return summary


def get_items(task: str, lang: Language, split: str = "test") -> list[Item]:
    if task == "mgsm":
        return load_mgsm(lang, split)
    if task == "belebele":
        return load_belebele(lang)
    raise ValueError(f"unknown task {task!r}")
