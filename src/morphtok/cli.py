"""``morphtok`` command line interface.

Stages are separate commands so runs can be split by time or by algorithm:

    morphtok fetch-data                     # downloads everything into ./data, ./models
    morphtok part-a  --tag audit            # CPU  ~1-4h
    morphtok part-b  --tag align            # CPU  ~2-6h
    morphtok part-c  --tag caps             # GPU  ~4-8h
    morphtok analyze --run-id latest
    morphtok figures --run-id latest
    morphtok all                            # everything, one run id

Any command can be resumed by passing the same ``--run-id`` (or ``latest``);
finished work units are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import paths

# cache redirection must happen before transformers/datasets/torch are imported
paths.setup_env()

from .logging_utils import Timer, get_logger, log_env_summary, setup_logging  # noqa: E402
from .state import RunContext, list_runs, resolve_run_id  # noqa: E402

log = get_logger("cli")

MODEL_TOKENIZER_HINTS = {
    "qwen3": "qwen3",
    "qwen": "qwen3",
    "gemma-3": "gemma3",
    "gemma3": "gemma3",
    "llama-3.1": "llama31",
    "llama3.1": "llama31",
    "sarvam": "sarvam_m",
    "mistral-nemo": "mistral_nemo",
    "aya": "aya_expanse",
    "deepseek": "deepseek_v3",
}

DEFAULT_MODEL = "Qwen/Qwen3-8B"

#: Part C evaluates **two** models, not one, because H3 asks whether boundary-F1
#: predicts budgeted-accuracy loss *beyond* fertility — and a single model gives that
#: term no variation to explain beyond the language axis.  The two are chosen for a
#: tokenizer contrast rather than a capability contrast: on Part A's Malayalam
#: numbers Qwen3 devotes 0.01 % of its 151k byte-level BPE vocabulary to the script
#: and Gemma-3 0.75 % of its 262k SentencePiece one, so pooling them adds a
#: within-language, between-tokenizer comparison at matched prompts.
#:
#: Gemma-3 is gated: with a token in ``.hf_token`` (or ``HF_TOKEN``) the canonical
#: ``google/gemma-3-12b-it`` is used; without one, fall back to the ungated
#: ``unsloth/gemma-3-12b-it`` mirror, which ships the identical vocabulary and
#: weights, so the default pipeline runs either way.  Both models fit an 80 GB card
#: comfortably and are loaded one at a time regardless.
GEMMA_CANONICAL = "google/gemma-3-12b-it"
GEMMA_MIRROR = "unsloth/gemma-3-12b-it"


def _gemma_repo() -> str:
    return GEMMA_CANONICAL if paths._hf_token() else GEMMA_MIRROR


DEFAULT_MODELS = (DEFAULT_MODEL, _gemma_repo())


def infer_model_tokenizer_key(model: str) -> str | None:
    low = model.lower()
    for hint, key in MODEL_TOKENIZER_HINTS.items():
        if hint in low:
            return key
    return None


def resolve_part_c_models(args: argparse.Namespace) -> list[str]:
    """``--models`` wins, then an explicit ``--model``, else the two-model default."""
    if getattr(args, "models", None):
        return [m.strip() for m in str(args.models).split(",") if m.strip()]
    if getattr(args, "model", None):
        return [str(args.model)]
    return list(DEFAULT_MODELS)


# ---------------------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------------------
def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-id", default=None,
                   help="reuse/resume a run id ('latest' for the most recent); "
                        "default: new timestamped id")
    p.add_argument("--tag", default=None, help="suffix for a new run id, e.g. --tag ablation")
    p.add_argument("--overwrite", action="store_true",
                   help="discard previous results for this stage instead of resuming")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _add_selection_args(p: argparse.ArgumentParser, default_langs: str) -> None:
    p.add_argument("--tokenizers", default=None,
                   help="comma list of zoo keys, or all|default|mainstream|specialized|byte")
    p.add_argument("--langs", default=default_langs,
                   help="comma list of ISO-639-3 codes, or all|indic|core|morph|mgsm")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="morphtok",
        description="Morphology, not frequency, should govern vocabulary allocation — "
                    "experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # -- env-info -----------------------------------------------------------
    q = sub.add_parser("env-info", help="show repo-local cache paths and probe tokenizers")
    q.add_argument("--probe-tokenizers", action="store_true",
                   help="try loading every tokenizer and report availability")
    q.add_argument("--log-level", default="INFO")

    # -- fetch-data ---------------------------------------------------------
    d = sub.add_parser("fetch-data", help="download all corpora, analyzers, fonts")
    _add_selection_args(d, "all")
    d.add_argument("--tasks", default="mgsm,belebele", help="Part C tasks to pre-download")
    d.add_argument("--gold-types", type=int, default=2000,
                   help="gold word types per language")
    d.add_argument("--flores-dir", default=None,
                   help="use an existing local flores200_dataset/ directory")
    d.add_argument("--skip-tasks", action="store_true", help="skip Part C data")
    d.add_argument("--train-morfessor", action="store_true",
                   help="train a fallback Morfessor model where no pretrained one exists")
    d.add_argument("--force", action="store_true", help="re-download / rebuild everything")
    d.add_argument("--log-level", default="INFO")

    # -- gold ---------------------------------------------------------------
    g = sub.add_parser("gold", help="(re)build gold morpheme segmentations only")
    _add_selection_args(g, "morph")
    g.add_argument("--gold-types", type=int, default=2000)
    g.add_argument("--train-morfessor", action="store_true")
    g.add_argument("--force", action="store_true")
    g.add_argument("--log-level", default="INFO")

    # -- part-a -------------------------------------------------------------
    a = sub.add_parser("part-a", help="Part A: Indic fertility audit (CPU)")
    _add_selection_args(a, "all")
    a.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    a.add_argument("--max-sentences", type=int, default=None,
                   help="cap sentences per language (smoke tests)")
    a.add_argument("--no-save-word-stats", action="store_true",
                   help="skip the per-word-type parquet artefacts")
    _add_run_args(a)

    # -- part-b -------------------------------------------------------------
    b = sub.add_parser("part-b", help="Part B: morpheme-boundary alignment (CPU)")
    _add_selection_args(b, "morph")
    b.add_argument("--tolerance", type=int, default=1,
                   help="character tolerance for the fuzzy boundary F1")
    b.add_argument("--no-save-words", action="store_true",
                   help="do not write per-word alignment parquets")
    _add_run_args(b)

    # -- part-c -------------------------------------------------------------
    c = sub.add_parser("part-c", help="Part C: fixed-budget capability study (GPU)")
    c.add_argument("--langs", default="mgsm,indic,eng",
                   help="languages to evaluate (Belebele covers all; MGSM subsets)")
    c.add_argument("--model", default=None,
                   help="evaluate this single model instead of the default sweep ("
                        + ", ".join(DEFAULT_MODELS) + ")")
    c.add_argument("--models", default=None,
                   help="comma list of models to evaluate SEQUENTIALLY; each is fully "
                        "unloaded (GPU freed) before the next is loaded. Default: "
                        + ",".join(DEFAULT_MODELS))
    c.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    c.add_argument("--tasks", default="mgsm,belebele")
    c.add_argument("--conditions", default="budget1024,budget2048,unconstrained")
    c.add_argument("--n-items", type=int, default=250, help="eval items per language")
    c.add_argument("--n-shots", type=int, default=5)
    c.add_argument("--batch-size", type=int, default=32,
                   help="generations per checkpoint flush")
    c.add_argument("--max-model-len", type=int, default=8192,
                   help="high-fertility languages need room: unconstrained k=5 "
                        "Tamil prompts exceed 5k tokens")
    c.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    c.add_argument("--tensor-parallel-size", type=int, default=1)
    c.add_argument("--no-chat-template", action="store_true")
    c.add_argument("--dry-run", action="store_true",
                   help="build prompts and report budget/shot statistics without "
                        "loading a model (CPU-only check before booking the GPU)")
    c.add_argument("--save-prompts", action="store_true",
                   help="also write every rendered prompt to prompts/*.jsonl "
                        "(large; the sha1 is always recorded regardless)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--model-tokenizer-key", default=None,
                   help="zoo key of the model's tokenizer (auto-inferred by default)")
    _add_run_args(c)

    # -- part-d -------------------------------------------------------------
    pd_ = sub.add_parser("part-d",
                         help="Part D: morpheme-constrained vs frequency BPE prototype")
    _add_selection_args(pd_, "core")
    pd_.add_argument("--vocab-size", type=int, default=16000,
                     help="identical budget for both arms")
    pd_.add_argument("--max-types", type=int, default=200000,
                     help="training word types drawn from the analyzer lexicon")
    pd_.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    _add_run_args(pd_)

    # -- analyze / figures --------------------------------------------------
    an = sub.add_parser("analyze", help="regressions and tables (H1-H4)")
    an.add_argument("--run-id", default="latest")
    an.add_argument("--include-byte", action="store_true",
                    help="keep byte-level tokenizers in the regressions")
    an.add_argument("--model-tokenizer-key", default=None)
    an.add_argument("--no-pool-part-c", action="store_true",
                    help="use only this run's Part C results instead of pooling every "
                         "run (pooling adds model-level replication to H3)")
    an.add_argument("--log-level", default="INFO")

    f = sub.add_parser("figures", help="render the paper's figures")
    f.add_argument("--run-id", default="latest")
    f.add_argument("--log-level", default="INFO")

    # -- all ----------------------------------------------------------------
    al = sub.add_parser("all", help="fetch-data + A + B + C + analyze + figures")
    _add_selection_args(al, "all")
    al.add_argument("--morph-langs", default="morph")
    al.add_argument("--part-c-langs", default="mgsm,indic,eng")
    al.add_argument("--skip-part-c", action="store_true")
    al.add_argument("--skip-part-d", action="store_true")
    al.add_argument("--vocab-size", type=int, default=16000,
                    help="Part D prototype vocabulary budget (both arms)")
    al.add_argument("--skip-fetch", action="store_true")
    al.add_argument("--model", default=None,
                    help="single Part C model instead of the default sweep")
    al.add_argument("--models", default=None,
                    help="comma list of Part C models, run sequentially. Default: "
                         + ",".join(DEFAULT_MODELS))
    al.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    al.add_argument("--tasks", default="mgsm,belebele")
    al.add_argument("--n-items", type=int, default=250)
    al.add_argument("--max-sentences", type=int, default=None)
    al.add_argument("--gold-types", type=int, default=2000)
    al.add_argument("--max-model-len", type=int, default=8192)
    al.add_argument("--batch-size", type=int, default=32)
    al.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    al.add_argument("--tensor-parallel-size", type=int, default=1)
    _add_run_args(al)

    v = sub.add_parser("verify",
                       help="check a run is correct, consistent and complete")
    v.add_argument("--run-id", default="latest")
    v.add_argument("--no-rescore", action="store_true",
                   help="skip re-deriving Part C accuracies from the saved predictions")
    v.add_argument("--log-level", default="INFO")

    sub.add_parser("runs", help="list runs and their stage status")
    return p


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_env_info(args: argparse.Namespace) -> None:
    setup_logging("env-info", level=args.log_level)
    log_env_summary()
    for name, d in [("data", paths.DATA_DIR), ("models", paths.MODELS_DIR),
                    ("runs", paths.RUNS_DIR), ("results", paths.RESULTS_DIR),
                    ("logs", paths.LOGS_DIR), ("cache", paths.CACHE_DIR)]:
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
        log.info("%-8s %-46s %6.2f GB", name, paths.relpath(d), size)
    if args.probe_tokenizers:
        from .tokenizer_zoo import ZOO, load_many

        loaded, skipped = load_many(list(ZOO))
        for key, lt in loaded.items():
            log.info("OK    %-14s %-44s vocab=%d", key, lt.source, lt.n_vocab)
        for key, why in skipped.items():
            log.warning("MISS  %-14s %s", key, why[:120])


def cmd_fetch_data(args: argparse.Namespace) -> None:
    setup_logging("fetch-data", level=args.log_level)
    log_env_summary()
    from .data_flores import ensure_flores
    from .data_morph import build_gold, fetch_resources
    from .data_share import fetch_wiki_sizes
    from .data_tasks import ensure_tasks
    from .fonts import ensure_fonts
    from .languages import resolve
    from .progress import pbar

    langs = resolve(args.langs, [])
    with Timer("fetch FLORES-200", log):
        ensure_flores(langs, explicit_dir=args.flores_dir, force=args.force)
    with Timer("fetch morphology analyzers", log):
        fetch_resources(langs, force=args.force)
    with Timer("build gold segmentations", log):
        for lang in pbar(langs, desc="gold segmentations", unit="lang"):
            build_gold(lang, n_types=args.gold_types,
                       train_fallback=args.train_morfessor, force=args.force)
    with Timer("fetch wikipedia size proxy", log):
        fetch_wiki_sizes(force=args.force)
    with Timer("fetch fonts", log):
        ensure_fonts(force=args.force)
    if not args.skip_tasks:
        with Timer("fetch Part C task data", log):
            tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
            summary = ensure_tasks(langs, tasks, force=args.force)
            log.info("task item counts: %s", json.dumps(summary))
    _write_gold_index()
    log.info("fetch-data complete — everything is under %s", paths.relpath(paths.DATA_DIR))


def _write_gold_index() -> None:
    import pandas as pd

    rows = []
    for p in sorted(paths.GOLD_DIR.glob("*.coverage.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if rows:
        df = pd.json_normalize(rows)
        out = paths.GOLD_DIR / "coverage.csv"
        df.to_csv(out, index=False)
        log.info("gold coverage report -> %s", paths.relpath(out))


def cmd_gold(args: argparse.Namespace) -> None:
    setup_logging("gold", level=args.log_level)
    from .data_morph import build_gold, fetch_resources
    from .languages import resolve
    from .progress import pbar

    langs = resolve(args.langs, [])
    fetch_resources(langs, force=args.force)
    for lang in pbar(langs, desc="gold segmentations", unit="lang"):
        build_gold(lang, n_types=args.gold_types, train_fallback=args.train_morfessor,
                   force=args.force)
    _write_gold_index()


def _prepare(args: argparse.Namespace, stage: str) -> tuple[RunContext, dict]:
    run_id = resolve_run_id(args.run_id, args.tag)
    setup_logging(stage, run_id=run_id, level=args.log_level)
    log_env_summary()
    cfg = {k: v for k, v in vars(args).items() if k not in ("func",)}
    cfg["stage"] = stage
    cfg["run_id"] = run_id
    ctx = RunContext(run_id=run_id, stage=stage, overwrite=args.overwrite, config=cfg)
    return ctx, cfg


def cmd_part_a(args: argparse.Namespace) -> None:
    ctx, _ = _prepare(args, "part_a")
    from .data_flores import ensure_flores
    from .languages import resolve
    from .part_a_audit import run_part_a
    from .tokenizer_zoo import resolve_specs

    langs = resolve(args.langs, [])
    specs = resolve_specs(args.tokenizers)
    ensure_flores(langs)
    ctx.set_status(tokenizers_requested=[s.key for s in specs])
    with Timer("PART A — Indic fertility audit", log):
        df = run_part_a(specs, langs, ctx, split=args.split,
                        max_sentences=args.max_sentences,
                        save_word_stats=not args.no_save_word_stats)
    ctx.close(done=True, rows=len(df))
    if not df.empty:
        log.info("\n%s", df[["tokenizer", "language", "fertility",
                            "premium_corpus" if "premium_corpus" in df.columns
                            else "fertility_uni"]].to_string(index=False))


def cmd_part_b(args: argparse.Namespace) -> None:
    ctx, _ = _prepare(args, "part_b")
    from .languages import resolve
    from .part_b_alignment import run_part_b
    from .tokenizer_zoo import resolve_specs

    langs = resolve(args.langs, [])
    specs = resolve_specs(args.tokenizers)
    ctx.set_status(tokenizers_requested=[s.key for s in specs])
    with Timer("PART B — morpheme boundary alignment", log):
        df = run_part_b(specs, langs, ctx, tolerance=args.tolerance,
                        save_words=not args.no_save_words)
    ctx.close(done=True, rows=len(df))
    if not df.empty:
        cols = ["tokenizer", "language", "boundary_f1", "morpheme_integrity",
                "tokens_per_gold_word"]
        log.info("\n%s", df[cols].to_string(index=False))


def cmd_part_c(args: argparse.Namespace) -> None:
    models = resolve_part_c_models(args)
    tok_keys = {m: (args.model_tokenizer_key or infer_model_tokenizer_key(m))
                for m in models}
    args.model_tokenizer_key = tok_keys[models[0]]
    args.model = models[0]  # config.json records a concrete id, never None
    args.models_resolved = models
    dry = getattr(args, "dry_run", False)
    ctx, _ = _prepare(args, "part_c_dryrun" if dry else "part_c")
    from .data_tasks import ensure_tasks
    from .languages import resolve
    from .part_c_capability import run_part_c_models

    langs = resolve(args.langs, [])
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    ensure_tasks(langs, list(tasks))
    log.info("models=%s tokenizer_keys=%s tasks=%s conditions=%s",
             models, tok_keys, tasks, conditions)
    with Timer("PART C — fixed-budget capability", log):
        df = run_part_c_models(
            models, langs, ctx, model_tokenizer_keys=tok_keys,
            backend=args.backend, tasks=tasks,
            conditions=conditions, n_items=args.n_items, n_shots=args.n_shots,
            batch_size=args.batch_size, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            use_chat_template=not args.no_chat_template, seed=args.seed,
            dry_run=dry, save_prompts=args.save_prompts,
        )
    ctx.close(done=True, rows=len(df))
    if not df.empty:
        cols = [c for c in ["model", "task", "language", "condition", "accuracy",
                            "mean_shots", "mean_prompt_tokens"] if c in df.columns]
        log.info("\n%s", df[cols].to_string(index=False))


def cmd_part_d(args: argparse.Namespace) -> None:
    ctx, _ = _prepare(args, "part_d")
    from .languages import resolve
    from .part_d_prototype import run_part_d

    langs = resolve(args.langs, [])
    with Timer("PART D — morphology-governed allocator prototype", log):
        df = run_part_d(langs, ctx, vocab_size=args.vocab_size,
                        max_types=args.max_types, split=args.split)
    ctx.close(done=True, rows=len(df))
    if not df.empty:
        cols = [c for c in ["language", "fertility_baseline", "fertility_morph",
                            "fertility_delta", "boundary_f1_baseline",
                            "boundary_f1_morph", "boundary_f1_delta"] if c in df.columns]
        log.info("\n%s", df[cols].to_string(index=False))


def cmd_analyze(args: argparse.Namespace) -> None:
    run_id = None if args.run_id in ("latest", None) else args.run_id
    setup_logging("analyze", run_id=run_id, level=args.log_level)
    from .analysis import run_analysis

    with Timer("ANALYSIS — H1-H4", log):
        run_analysis(run_id=run_id, include_byte=args.include_byte,
                     model_tokenizer=args.model_tokenizer_key,
                     pool_part_c=not args.no_pool_part_c)


def cmd_figures(args: argparse.Namespace) -> None:
    run_id = None if args.run_id in ("latest", None) else args.run_id
    setup_logging("figures", run_id=run_id, level=args.log_level)
    from .figures import make_all

    with Timer("FIGURES", log):
        make_all(run_id=run_id)


def cmd_all(args: argparse.Namespace) -> None:
    run_id = resolve_run_id(args.run_id, args.tag or "full")
    log.info("full pipeline under run id %s", run_id)

    base = dict(vars(args))
    base["run_id"] = run_id

    if not args.skip_fetch:
        fetch = argparse.Namespace(
            langs=args.langs, tokenizers=args.tokenizers, tasks=args.tasks,
            gold_types=args.gold_types, flores_dir=None,
            skip_tasks=args.skip_part_c, train_morfessor=False, force=False,
            log_level=args.log_level,
        )
        cmd_fetch_data(fetch)

    cmd_part_a(argparse.Namespace(
        langs=args.langs, tokenizers=args.tokenizers, split="devtest",
        max_sentences=args.max_sentences, no_save_word_stats=False,
        run_id=run_id, tag=None,
        overwrite=args.overwrite, log_level=args.log_level))

    cmd_part_b(argparse.Namespace(
        langs=args.morph_langs, tokenizers=args.tokenizers, tolerance=1,
        no_save_words=False, run_id=run_id, tag=None, overwrite=args.overwrite,
        log_level=args.log_level))

    if not args.skip_part_c:
        cmd_part_c(argparse.Namespace(
            langs=args.part_c_langs, model=args.model, models=args.models,
            save_prompts=False, backend=args.backend,
            tasks=args.tasks, conditions="budget1024,budget2048,unconstrained",
            n_items=args.n_items, n_shots=5, batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size, no_chat_template=False,
            dry_run=False, seed=0, model_tokenizer_key=None, run_id=run_id, tag=None,
            overwrite=args.overwrite, log_level=args.log_level))
    else:
        log.warning("--skip-part-c: no GPU stage in this run")

    if not args.skip_part_d:
        cmd_part_d(argparse.Namespace(
            langs=args.morph_langs, tokenizers=args.tokenizers,
            vocab_size=args.vocab_size, max_types=200000, split="devtest",
            run_id=run_id, tag=None, overwrite=args.overwrite,
            log_level=args.log_level))

    cmd_analyze(argparse.Namespace(run_id=run_id, include_byte=False,
                                   model_tokenizer_key=None, no_pool_part_c=False,
                                   log_level=args.log_level))
    cmd_figures(argparse.Namespace(run_id=run_id, log_level=args.log_level))
    try:  # a pipeline that finishes should also say whether it finished *correctly*
        cmd_verify(argparse.Namespace(run_id=run_id, no_rescore=False,
                                      log_level=args.log_level))
    except SystemExit as exc:
        log.error("%s", exc)
    log.info("PIPELINE COMPLETE — run_id=%s", run_id)
    log.info("tables: %s", paths.relpath(paths.TABLES_DIR))
    log.info("figures: %s", paths.relpath(paths.FIGURES_DIR))


def cmd_verify(args: argparse.Namespace) -> int:
    run_id = resolve_run_id(args.run_id if args.run_id != "latest" else "latest")
    setup_logging("verify", run_id=run_id, level=args.log_level)
    from .verify import verify_run

    rep = verify_run(run_id, rescore=not args.no_rescore)
    if rep.errors:
        raise SystemExit(f"verification FAILED: {rep.summary()}")
    log.info("verification passed: %s", rep.summary())
    return 0


def cmd_runs(_args: argparse.Namespace) -> None:
    setup_logging("runs")
    rows = list_runs()
    if not rows:
        log.info("no runs yet")
        return
    for r in rows:
        log.info("%-34s %s", r["run_id"], r["mtime"])
        for stage, info in r["stages"].items():
            log.info("    %-10s %s", stage, json.dumps(info))


COMMANDS = {
    "env-info": cmd_env_info,
    "fetch-data": cmd_fetch_data,
    "gold": cmd_gold,
    "part-a": cmd_part_a,
    "part-b": cmd_part_b,
    "part-c": cmd_part_c,
    "part-d": cmd_part_d,
    "analyze": cmd_analyze,
    "figures": cmd_figures,
    "verify": cmd_verify,
    "all": cmd_all,
    "runs": cmd_runs,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        COMMANDS[args.cmd](args)
    except KeyboardInterrupt:
        log.warning("interrupted — progress is saved; rerun with --run-id latest to resume")
        return 130
    except SystemExit as exc:
        # ``raise SystemExit("message")`` is the house style for a user-facing abort
        # (unknown tokenizer, stage already locked, ...), so exc.code is usually a
        # string, not a number — int() on it would bury the message under a
        # ValueError traceback.
        if exc.code:
            log.error("%s", exc)
        if isinstance(exc.code, int):
            return exc.code
        return 0 if exc.code is None else 1
    except Exception:
        log.exception("stage failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
