"""``morphtok verify`` — is this run correct, consistent and complete?

Checks that can catch a silently broken 8-hour run:

* **environment** — every cache still points inside the repo;
* **data** — FLORES splits, gold segmentations and coverage reports agree;
* **per stage** — every committed record has its artifact on disk, every artifact has
  a record, no duplicate units, counts in the record match the rows in the file;
* **Part C correctness** — accuracies are *recomputed from the saved predictions* and
  compared against the committed record, which also proves the predictions are
  complete enough to re-analyse without the GPU;
* **outputs** — analysis tables and figures exist and are non-empty.

Findings are ERROR (something is wrong), WARN (probably fine, worth knowing) or OK.
Exit status is non-zero if any ERROR was found.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .logging_utils import get_logger
from .state import DoneSet, read_json, read_jsonl

log = get_logger("verify")

STAGES = ("part_a", "part_b", "part_c", "part_d")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok: list[str] = field(default_factory=list)

    def error(self, msg: str, *a: object) -> None:
        self.errors.append(msg % a if a else msg)
        log.error("FAIL  %s", msg % a if a else msg)

    def warn(self, msg: str, *a: object) -> None:
        self.warnings.append(msg % a if a else msg)
        log.warning("WARN  %s", msg % a if a else msg)

    def good(self, msg: str, *a: object) -> None:
        self.ok.append(msg % a if a else msg)
        log.info("OK    %s", msg % a if a else msg)

    def summary(self) -> str:
        return (f"{len(self.ok)} checks passed, {len(self.warnings)} warnings, "
                f"{len(self.errors)} errors")


# ---------------------------------------------------------------------------
def check_environment(rep: Report) -> None:
    root = str(paths.ROOT)
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TIKTOKEN_CACHE_DIR",
                "TORCH_HOME", "VLLM_CACHE_ROOT", "MPLCONFIGDIR", "TMPDIR",
                "XDG_CACHE_HOME"):
        val = os.environ.get(key)
        if not val:
            rep.warn("%s is unset", key)
        elif not val.startswith(root):
            rep.error("%s points outside the repo: %s", key, val)
    if not [e for e in rep.errors if "points outside" in e]:
        rep.good("all caches are repo-local")

    home = Path.home()
    strays = [p for p in (home / ".cache" / "huggingface", home / ".cache" / "vllm")
              if p.exists()]
    if strays:
        rep.warn("pre-existing caches in $HOME (not written by this project): %s",
                 ", ".join(str(p) for p in strays))


def check_data(rep: Report) -> None:
    from .data_flores import PROC_ROOT
    from .languages import LANGUAGES

    missing = []
    for lang in LANGUAGES:
        for split in ("dev", "devtest"):
            if not (PROC_ROOT / split / f"{lang.flores}.txt").is_file():
                missing.append(f"{lang.code}/{split}")
    if missing:
        rep.warn("FLORES text missing for %d language-splits (%s%s)", len(missing),
                 ", ".join(missing[:6]), " …" if len(missing) > 6 else "")
    else:
        rep.good("FLORES dev+devtest present for all %d languages", len(LANGUAGES))

    gold_files = sorted(paths.GOLD_DIR.glob("*.gold.jsonl"))
    if not gold_files:
        rep.error("no gold segmentations — run `morphtok gold`")
        return
    empty, mismatched = [], []
    for gf in gold_files:
        code = gf.name.split(".")[0]
        n = sum(1 for line in gf.open(encoding="utf-8") if line.strip())
        cov = read_json(paths.GOLD_DIR / f"{code}.coverage.json", {}) or {}
        if n == 0:
            empty.append(code)
        elif cov.get("n_gold") not in (None, n):
            mismatched.append(f"{code} ({n} lines vs n_gold={cov.get('n_gold')})")
    if mismatched:
        rep.error("gold files disagree with their coverage report: %s",
                  ", ".join(mismatched))
    if empty:
        rep.warn("languages with an empty gold set (excluded from Part B): %s",
                 ", ".join(empty))
    rep.good("%d gold segmentation files, %d non-empty",
             len(gold_files), len(gold_files) - len(empty))


# ---------------------------------------------------------------------------
def _records(run_dir: Path, stage: str) -> list[dict]:
    return read_jsonl(run_dir / stage / "records.jsonl")


def check_stage_generic(rep: Report, run_dir: Path, stage: str) -> list[dict]:
    d = run_dir / stage
    recs = _records(run_dir, stage)
    if not recs:
        rep.warn("%s: no records", stage)
        return []
    done = DoneSet(d / "_done.txt")
    units = [r.get("_unit") for r in recs]
    dupes = {u for u in units if units.count(u) > 1}
    if dupes:
        rep.warn("%s: %d unit(s) committed more than once (last wins): %s",
                 stage, len(dupes), ", ".join(sorted(dupes)[:4]))
    missing_marker = [u for u in set(units) if u not in done]
    if missing_marker:
        rep.error("%s: %d record(s) without a done-marker — resume would redo them",
                  stage, len(missing_marker))
    orphan = [u for u in done.keys if u not in set(units)]
    if orphan:
        rep.error("%s: %d done-marker(s) without a record", stage, len(orphan))
    if not missing_marker and not orphan:
        rep.good("%s: %d records match %d done-markers", stage, len(set(units)),
                 len(done))
    return recs


def check_part_a(rep: Report, run_dir: Path) -> None:
    import pandas as pd

    recs = check_stage_generic(rep, run_dir, "part_a")
    if not recs:
        return
    bad = 0
    for r in recs:
        p = paths.ROOT / str(r.get("parquet", ""))
        if not p.is_file():
            rep.error("part_a: missing sentence parquet for %s/%s",
                      r["tokenizer"], r["language"])
            bad += 1
            continue
        n = len(pd.read_parquet(p, columns=["sent_id"]))
        if n != r["n_sentences"]:
            rep.error("part_a: %s/%s parquet has %d rows, record says %d",
                      r["tokenizer"], r["language"], n, r["n_sentences"])
            bad += 1
    if not bad:
        rep.good("part_a: all %d per-sentence parquets match their records", len(recs))
    words = list((run_dir / "part_a" / "words").glob("*.parquet"))
    if words:
        rep.good("part_a: %d per-word-type artifacts saved", len(words))
    else:
        rep.warn("part_a: no per-word-type artifacts (--no-save-word-stats?)")
    for name in ("indic_audit.csv", "fertility_matrix.csv"):
        if not (run_dir / "part_a" / name).is_file():
            rep.error("part_a: %s missing", name)


def check_part_b(rep: Report, run_dir: Path) -> None:
    import pandas as pd

    recs = check_stage_generic(rep, run_dir, "part_b")
    if not recs:
        return
    wdir = run_dir / "part_b" / "words"
    bad = 0
    for r in recs:
        p = wdir / f"{r['tokenizer']}__{r['language']}.parquet"
        if not p.is_file():
            rep.warn("part_b: no per-word parquet for %s/%s (--no-save-words?)",
                     r["tokenizer"], r["language"])
            continue
        n = len(pd.read_parquet(p, columns=["word"]))
        if n != r["n_gold_words"]:
            rep.error("part_b: %s/%s parquet has %d words, record says %d",
                      r["tokenizer"], r["language"], n, r["n_gold_words"])
            bad += 1
    if not bad:
        rep.good("part_b: per-word artifacts consistent with %d records", len(recs))
    if not (run_dir / "part_b" / "qualitative_examples.json").is_file():
        rep.warn("part_b: qualitative_examples.json missing (Figure 3 source)")


def check_part_c(rep: Report, run_dir: Path, rescore: bool = True) -> None:
    from .part_c_capability import score_prediction

    recs = check_stage_generic(rep, run_dir, "part_c")
    if not recs:
        return
    bad = mismatched = 0
    n_preds = 0
    for r in recs:
        if not r.get("n"):
            continue
        p = paths.ROOT / str(r.get("predictions", ""))
        if not p.is_file():
            rep.error("part_c: predictions missing for %s/%s/%s",
                      r.get("model", "?"), r["language"], r["condition"])
            bad += 1
            continue
        rows = {x["uid"]: x for x in read_jsonl(p)}.values()
        n_preds += len(rows)
        if len(rows) != r["n"]:
            rep.error("part_c: %s/%s/%s has %d predictions, record says %d",
                      r.get("model", "?"), r["language"], r["condition"],
                      len(rows), r["n"])
            bad += 1
            continue
        if not rescore:
            continue
        # recompute accuracy straight from the saved generations
        acc = sum(score_prediction(r["task"], x["gold"], x["output"])
                  for x in rows) / max(1, len(rows))
        if abs(acc - float(r["accuracy"])) > 1e-6:
            rep.error("part_c: %s/%s/%s recomputed accuracy %.4f != recorded %.4f",
                      r.get("model", "?"), r["language"], r["condition"], acc,
                      r["accuracy"])
            mismatched += 1
    if not bad and not mismatched:
        rep.good("part_c: %d predictions on disk; every accuracy re-derives from them",
                 n_preds)
    models = sorted({r.get("model", "") for r in recs if r.get("model")})
    if len(models) > 1:
        rep.good("part_c: %d models evaluated (%s)", len(models), ", ".join(models))
    elif models:
        rep.warn("part_c: only one model (%s) — H3's boundary-F1 term stays "
                 "underpowered until a second is run", models[0])


def check_part_d(rep: Report, run_dir: Path) -> None:
    recs = check_stage_generic(rep, run_dir, "part_d")
    if not recs:
        return
    langs = {r["language"] for r in recs}
    for lang in langs:
        arms = {r["arm"] for r in recs if r["language"] == lang}
        if not {"baseline", "morph_train", "morph_full"} <= arms:
            rep.error("part_d: %s is missing arms %s", lang,
                      {"baseline", "morph_train", "morph_full"} - arms)
        vocabs = {r["arm"]: r.get("vocab_realised") for r in recs
                  if r["language"] == lang}
        if vocabs.get("baseline") and vocabs.get("morph_train"):
            if vocabs["baseline"] != vocabs["morph_train"]:
                rep.error("part_d: %s vocabulary budgets not matched (%s)", lang, vocabs)
    tok = list((run_dir / "part_d" / "tokenizers").rglob("*.json"))
    if tok:
        rep.good("part_d: %d trained tokenizers saved for %d languages",
                 len(tok), len(langs))
    words = list((run_dir / "part_d" / "words").glob("*.parquet"))
    if words:
        rep.good("part_d: %d per-word artifacts saved", len(words))


def check_outputs(rep: Report, run_dir: Path) -> None:
    tables = sorted(paths.TABLES_DIR.glob("*.csv"))
    empties = [t.name for t in tables if t.stat().st_size < 40]
    if not tables:
        rep.warn("no analysis tables — run `morphtok analyze`")
    else:
        rep.good("%d analysis tables in %s", len(tables),
                 paths.relpath(paths.TABLES_DIR))
    if empties:
        rep.warn("empty tables: %s", ", ".join(empties))
    figs = sorted(paths.FIGURES_DIR.glob("*.png"))
    if not figs:
        rep.warn("no figures — run `morphtok figures`")
    else:
        rep.good("%d figures rendered", len(figs))
    snap = run_dir / "analysis"
    if snap.is_dir():
        rep.good("analysis snapshot kept with the run (%d files)",
                 len(list(snap.glob("*"))))


# ---------------------------------------------------------------------------
def verify_run(run_id: str, rescore: bool = True) -> Report:
    rep = Report()
    run_dir = paths.RUNS_DIR / run_id
    if not run_dir.is_dir():
        rep.error("run %s does not exist", run_id)
        return rep
    log.info("verifying run %s", run_id)
    check_environment(rep)
    check_data(rep)
    present = [s for s in STAGES if (run_dir / s / "records.jsonl").is_file()]
    if not present:
        rep.error("run %s has no stage records at all", run_id)
    for stage in present:
        {"part_a": check_part_a, "part_b": check_part_b,
         "part_c": lambda r, d: check_part_c(r, d, rescore),
         "part_d": check_part_d}[stage](rep, run_dir)
    missing = [s for s in STAGES if s not in present]
    if missing:
        rep.warn("stages not run: %s", ", ".join(missing))
    check_outputs(rep, run_dir)
    log.info("-" * 80)
    log.info("VERIFY %s: %s", run_id, rep.summary())
    return rep
