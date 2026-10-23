"""Part C — capability under a fixed token budget (GPU, inference only).

Mechanism under test: a fixed *prompt* budget buys fewer few-shot exemplars (and,
past a point, a truncated question) in a high-fertility language.  We therefore run
each (task × language) at ``budget=1024``, ``budget=2048`` and unconstrained ``k=5``,
and later regress the budgeted-accuracy loss on fertility **and** boundary-F1 of the
evaluated model's own tokenizer (H3).

Every generation is kept.  Predictions are appended to
``predictions/<model>__<task>__<lang>__<cond>.jsonl`` and fsynced after every batch, so
a crashed 8-hour run resumes at the item and the full output text, the extracted
answer, the exemplars used and the prompt hash remain available for re-analysis
without the GPU (``morphtok verify`` re-derives every accuracy from them).

Models are evaluated one at a time: ``run_part_c_models`` loads a model, runs all of
its units, then tears it down — vLLM's KV cache and model-parallel state are released
explicitly before the next model is constructed, so a multi-model sweep does not OOM.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd

from . import paths
from .data_tasks import Item, get_items
from .languages import Language
from .logging_utils import Timer, get_logger
from .progress import pbar
from .state import DoneSet, JsonlWriter, RunContext, read_jsonl

log = get_logger("part_c")

PRED_DIR = "predictions"

# Indic digit blocks -> ASCII (MGSM answers are written in native digits)
_DIGIT_MAP = {}
for _base in (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66,
              0x06F0, 0x0660):
    for _d in range(10):
        _DIGIT_MAP[chr(_base + _d)] = str(_d)
_DIGIT_TABLE = str.maketrans(_DIGIT_MAP)

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
# Multiple-choice extraction, tried in this order.  The loose ``[ABCD]`` search that
# an earlier version used as a fallback is a trap: uppercased "The answer is 2"
# contains the "A" of ANSWER, so every such generation scored as option A.
LEAD_RE = re.compile(r"^\(?([ABCD])(?:[).:,\s]|$)")
ANSWER_MARK_RE = re.compile(r"(?:ANSWER|OPTION|CHOICE)\D{0,10}?([ABCD1-4])\b")
LETTER_RE = re.compile(r"\b([ABCD])\b")
DIGIT_RE = re.compile(r"\b([1-4])\b")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
@dataclass
class PromptSpec:
    text: str
    n_shots: int
    n_prompt_tokens: int
    truncated: bool


def _mgsm_block(item: Item, with_answer: bool) -> str:
    if with_answer:
        answer = item.cot or f"The answer is {item.answer}."
        return f"Question: {item.question}\nAnswer: {answer}"
    return f"Question: {item.question}\nAnswer:"


def _belebele_block(item: Item, with_answer: bool) -> str:
    opts = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(item.choices))
    head = f"Passage: {item.context}\nQuestion: {item.question}\n{opts}\nAnswer:"
    return f"{head} {item.answer}" if with_answer else head


BLOCKS = {"mgsm": _mgsm_block, "belebele": _belebele_block}
INSTRUCTIONS = {
    "mgsm": "Solve the math word problem. End your reply with the final number.",
    "belebele": "Answer the multiple-choice question with a single letter (A, B, C or D).",
}


def build_prompt(
    task: str,
    item: Item,
    exemplars: list[Item],
    tokenizer,
    budget: int | None,
    max_shots: int = 5,
    use_chat_template: bool = True,
    hard_cap: int | None = None,
) -> PromptSpec:
    """Greedily fit exemplars into ``budget`` prompt tokens; truncate if the question
    alone overflows (context erosion, the effect we are measuring)."""
    block = BLOCKS[task]
    instruction = INSTRUCTIONS[task]
    tail = block(item, False)

    def render(shots: list[Item]) -> str:
        parts = [instruction] + [block(s, True) for s in shots] + [tail]
        body = "\n\n".join(parts)
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": body}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": body}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return body

    def n_tok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def clip(text: str, cap: int) -> tuple[str, int, bool]:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= cap:
            return text, len(ids), False
        # keep the tail: the question (and the generation prompt) must survive
        return tokenizer.decode(ids[-cap:], skip_special_tokens=True), cap, True

    shots = list(exemplars[:max_shots])
    if budget is None:
        text = render(shots)
        n = n_tok(text)
        if hard_cap and n > hard_cap:
            text, n, trunc = clip(text, hard_cap)
            return PromptSpec(text, len(shots), n, trunc)
        return PromptSpec(text, len(shots), n, False)

    chosen: list[Item] = []
    for ex in shots:
        trial = chosen + [ex]
        if n_tok(render(trial)) <= budget:
            chosen = trial
        else:
            break
    text = render(chosen)
    cap = min(budget, hard_cap) if hard_cap else budget
    text, n, truncated = clip(text, cap)
    return PromptSpec(text, len(chosen), n, truncated)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _norm_number(s: str) -> str | None:
    s = s.translate(_DIGIT_TABLE)
    hits = NUM_RE.findall(s)
    if not hits:
        return None
    raw = hits[-1].replace(",", "").rstrip(".")
    try:
        val = float(raw)
    except ValueError:
        return None
    return str(int(val)) if val == int(val) else f"{val:.4f}".rstrip("0")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _extracted_answer(task: str, output: str) -> str | None:
    """The answer the scorer actually read out of the generation (stored for audit).

    Multiple choice is read in decreasing order of evidence: a leading option letter,
    then an explicit "answer/option/choice … X" marker, then a standalone letter, then
    a standalone 1-4 mapped to A-D.  Letters embedded in words never count.
    """
    if task == "mgsm":
        return _norm_number(output)
    text = output.strip().upper()
    for rx in (LEAD_RE, ANSWER_MARK_RE, LETTER_RE, DIGIT_RE):
        m = rx.search(text)
        if m:
            g = m.group(1)
            return g if g in "ABCD" else "ABCD"[int(g) - 1]
    return None


def score_prediction(task: str, gold: str, output: str) -> int:
    """Score a generation *through* :func:`_extracted_answer`, so the ``extracted``
    field saved with every prediction is exactly what the score was computed from."""
    pred = _extracted_answer(task, output)
    if pred is None:
        return 0
    if task == "mgsm":
        ref = _norm_number(gold)
        return int(ref is not None and pred == ref)
    return int(pred == gold.strip().upper()[:1])


# ---------------------------------------------------------------------------
# generation backends
# ---------------------------------------------------------------------------
class DryGenerator:
    """Tokenizer-only stand-in: builds prompts and measures budgets, generates nothing.

    ``morphtok part-c --dry-run`` uses this to verify on a laptop how many few-shot
    exemplars each language actually fits into 1,024 / 2,048 tokens before any GPU
    time is spent.
    """

    def __init__(self, model: str, **_: object):
        from transformers import AutoTokenizer

        self.model = model
        self.backend = "dry"
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    def generate(self, prompts: list[str], max_tokens: int,
                 stop: list[str] | None = None) -> list[str]:
        return [""] * len(prompts)

    def close(self) -> None:
        return None


class Generator:
    """Thin wrapper so Part C runs on vLLM (GPU) or transformers (smoke test)."""

    def __init__(self, model: str, backend: str = "vllm", max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.90, dtype: str = "auto",
                 tensor_parallel_size: int = 1, seed: int = 0):
        self.model = model
        self.backend = backend
        self.max_model_len = max_model_len
        self._llm = None
        self._hf = None
        self.seed = seed
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if backend == "vllm":
            # vLLM binds an IPC socket at $TMPDIR/<uuid4>; the repo-local TMPDIR is
            # past the 107-char Unix-socket limit, so make sure it has been
            # shortened even when Part C is driven from something other than the CLI
            ipc_dir = paths.setup_short_tmpdir()
            log.info("vLLM IPC base path: %s", ipc_dir)

            from vllm import LLM

            log.info("initialising vLLM (model=%s, max_model_len=%d, tp=%d)",
                     model, max_model_len, tensor_parallel_size)
            self._llm = LLM(
                model=model,
                download_dir=str(paths.MODELS_DIR / "vllm"),
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=True,
            )
        else:
            import torch
            from transformers import AutoModelForCausalLM

            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
            log.info("initialising transformers backend on %s (model=%s)", device, model)
            self._hf = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype="auto", device_map=device if device != "mps" else None,
            )
            if device == "mps":
                self._hf = self._hf.to("mps")
            self._hf.eval()
            self._device = device
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def close(self) -> None:
        """Free the GPU before the next model is loaded.

        vLLM holds the KV cache and the NCCL/model-parallel state, so a second
        ``LLM(...)`` in the same process OOMs unless the first is torn down
        explicitly — deleting the Python reference is not enough.
        """
        import gc

        if self._llm is not None:
            try:
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment, destroy_model_parallel)

                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception as exc:  # pragma: no cover - vLLM version differences
                log.debug("vLLM teardown helpers unavailable: %s", exc)
            try:
                del self._llm.llm_engine
            except Exception:
                pass
        self._llm = None
        self._hf = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free, total = torch.cuda.mem_get_info()
                log.info("GPU released: %.1f / %.1f GB free", free / 1e9, total / 1e9)
        except Exception:  # pragma: no cover - CPU-only environments
            pass
        log.info("model %s unloaded", self.model)

    def __enter__(self) -> "Generator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def generate(self, prompts: list[str], max_tokens: int,
                 stop: list[str] | None = None) -> list[str]:
        if self._llm is not None:
            from vllm import SamplingParams

            params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                                    stop=stop, seed=self.seed)
            outs = self._llm.generate(prompts, params, use_tqdm=False)
            return [o.outputs[0].text for o in outs]
        import torch

        results = []
        for prompt in prompts:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self._hf.device)
            with torch.no_grad():
                out = self._hf.generate(
                    **enc, max_new_tokens=max_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            text = self.tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
            )
            for s in stop or []:
                text = text.split(s)[0]
            results.append(text)
        return results


# ---------------------------------------------------------------------------
# stage driver
# ---------------------------------------------------------------------------
CONDITIONS_DEFAULT = ["budget1024", "budget2048", "unconstrained"]
MAX_NEW_TOKENS = {"mgsm": 384, "belebele": 8}
STOPS = {"mgsm": ["\nQuestion:", "\n\nQuestion:"], "belebele": ["\n"]}


def _cond_budget(cond: str) -> int | None:
    if cond == "unconstrained":
        return None
    m = re.search(r"(\d+)", cond)
    return int(m.group(1)) if m else None


def _unit_key(task: str, lang: str, cond: str, model: str = "") -> str:
    tag = _model_tag(model)
    return f"{tag}::{task}::{lang}::{cond}" if tag else f"{task}::{lang}::{cond}"


def _model_tag(model: str) -> str:
    """Filesystem-safe short name for a model id (``Qwen/Qwen3-8B`` -> ``Qwen3-8B``)."""
    return model.split("/")[-1].replace(" ", "-") if model else ""


def run_part_c_models(
    models: list[str],
    languages: list[Language],
    ctx: RunContext,
    model_tokenizer_keys: dict[str, str | None] | None = None,
    **kw: object,
) -> pd.DataFrame:
    """Evaluate several models **sequentially**, with the GPU freed between them.

    Only one model is ever resident: each is loaded, run to completion over all its
    units, then torn down (``Generator.close``) before the next is constructed.  A
    crash mid-model still leaves every finished prediction on disk, and re-running
    resumes at the item.

    One model failing (weights unavailable, OOM, an unsupported architecture) does
    **not** abandon the others: the failure is logged, recorded in the run status and
    the sweep continues, because a second model exists to give H3 replication and
    losing it should not cost the first model's results too.  If *every* model fails
    the stage still raises.
    """
    failures: list[tuple[str, str]] = []
    for i, model in enumerate(models, 1):
        log.info("=" * 70)
        log.info("MODEL %d/%d: %s", i, len(models), model)
        ctx.set_status(current_model=model, model_index=i, models_total=len(models))
        keys = model_tokenizer_keys or {}
        try:
            run_part_c(languages, ctx, model=model,
                       model_tokenizer_key=keys.get(model), **kw)  # type: ignore[arg-type]
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.exception("MODEL %s FAILED (%s: %s)", model, type(exc).__name__, exc)
            failures.append((model, f"{type(exc).__name__}: {exc}"))
            ctx.set_status(failed_models=[m for m, _ in failures])

    if failures:
        for model, why in failures:
            log.error("Part C model failed: %s -> %s", model, why)
        if len(failures) == len(models):
            raise RuntimeError(
                "every Part C model failed: "
                + "; ".join(f"{m} ({why})" for m, why in failures))
        log.warning("continuing with %d/%d model(s); analysis will pool what ran",
                    len(models) - len(failures), len(models))
    # aggregate over everything on disk, not just the last model's units
    return aggregate_part_c(ctx)


def run_part_c(
    languages: list[Language],
    ctx: RunContext,
    model: str,
    backend: str = "vllm",
    tasks: tuple[str, ...] = ("mgsm", "belebele"),
    conditions: list[str] | None = None,
    n_items: int = 250,
    n_shots: int = 5,
    batch_size: int = 32,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    use_chat_template: bool = True,
    seed: int = 0,
    dry_run: bool = False,
    save_prompts: bool = False,
    model_tokenizer_key: str | None = None,
) -> pd.DataFrame:
    conditions = conditions or CONDITIONS_DEFAULT
    pred_dir = ctx.path(PRED_DIR)
    pred_dir.mkdir(parents=True, exist_ok=True)

    units: list[tuple[str, tuple[str, Language, str]]] = []
    for task in tasks:
        for lang in languages:
            if task == "mgsm" and not lang.mgsm:
                continue
            for cond in conditions:
                units.append((_unit_key(task, lang.code, cond, model),
                              (task, lang, cond)))
    log.info("Part C: %d units (%s) model=%s backend=%s",
             len(units), ", ".join(tasks), model, backend)
    ctx.set_status(stage="part_c", units_total=len(units), model=model, backend=backend)

    if dry_run:
        log.warning("--dry-run: building prompts only, no model is loaded")
        gen: DryGenerator | Generator = DryGenerator(model)
    else:
        gen = Generator(
            model, backend=backend, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size, seed=seed,
        )

    tok_key = model_tokenizer_key or ctx.config.get("model_tokenizer_key")
    bar = pbar(units, desc=f"Part C {_model_tag(model)}", unit="unit")
    try:
        for key, (task, lang, cond) in bar:
            bar.set_postfix_str(f"{task}/{lang.code}/{cond}")
            # unit keys gained a model prefix when multi-model runs were added; accept
            # the legacy key too so a run started before that still resumes
            if ctx.is_done(key) or ctx.is_done(_unit_key(task, lang.code, cond)):
                continue
            with Timer(f"part C {key}", log):
                _run_unit(gen, ctx, task, lang, cond, n_items, n_shots, batch_size,
                          use_chat_template,
                          hard_cap=max(256, max_model_len - MAX_NEW_TOKENS[task] - 8),
                          model=model, model_tokenizer_key=tok_key,
                          save_prompts=save_prompts)
            ctx.set_status(units_done=len(ctx.done))
    finally:
        bar.close()
        gen.close()  # always release the GPU, including on Ctrl-C or a crash
    return aggregate_part_c(ctx)


def _exemplars(task: str, lang: Language, n_shots: int, eval_items: list[Item]) -> list[Item]:
    if task == "mgsm":
        try:
            train = get_items("mgsm", lang, split="train")
        except Exception as exc:
            log.warning("mgsm train exemplars unavailable for %s (%s)", lang.code, exc)
            train = []
        if train:
            return train[:n_shots]
        return eval_items[:n_shots]
    return eval_items[:n_shots]


def _run_unit(gen: Generator | DryGenerator, ctx: RunContext, task: str,
              lang: Language, cond: str, n_items: int, n_shots: int, batch_size: int,
              use_chat_template: bool, hard_cap: int | None = None,
              model: str = "", model_tokenizer_key: str | None = None,
              save_prompts: bool = False) -> None:
    items = get_items(task, lang)
    tag = _model_tag(model)
    if not items:
        log.warning("no items for %s/%s — skipping", task, lang.code)
        ctx.commit(_unit_key(task, lang.code, cond, model),
                   {"model": model, "task": task, "language": lang.code,
                    "condition": cond, "n": 0})
        return
    shots = _exemplars(task, lang, n_shots, items)
    shot_uids = {s.uid for s in shots}
    eval_items = [i for i in items if i.uid not in shot_uids][:n_items]

    stem = f"{tag}__{task}__{lang.code}__{cond}" if tag else f"{task}__{lang.code}__{cond}"
    pred_path = ctx.path(PRED_DIR) / f"{stem}.jsonl"
    legacy = ctx.path(PRED_DIR) / f"{task}__{lang.code}__{cond}.jsonl"
    if legacy.is_file() and not pred_path.is_file():
        # predictions written before per-model filenames: adopt them rather than
        # regenerate (they belong to whichever model that run used)
        legacy.rename(pred_path)
        if legacy.with_suffix(".done").is_file():
            legacy.with_suffix(".done").rename(pred_path.with_suffix(".done"))
        log.info("adopted legacy prediction file %s -> %s", legacy.name, pred_path.name)
    done = DoneSet(pred_path.with_suffix(".done"))
    existing = read_jsonl(pred_path)
    have = {r["uid"] for r in existing} | done.keys
    todo = [i for i in eval_items if i.uid not in have]
    if existing:
        log.info("%s/%s/%s: resuming with %d/%d predictions on disk",
                 task, lang.code, cond, len(have), len(eval_items))

    budget = _cond_budget(cond)
    writer = JsonlWriter(pred_path)
    prompt_writer = (JsonlWriter(ctx.path("prompts") / f"{stem}.jsonl")
                     if save_prompts else None)
    try:
        batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
        for batch in pbar(batches, desc=f"{task}/{lang.code}/{cond}", unit="batch",
                          leave=False):
            specs = [
                build_prompt(task, it, shots, gen.tokenizer, budget, n_shots,
                             use_chat_template, hard_cap=hard_cap)
                for it in batch
            ]
            outs = gen.generate([s.text for s in specs], MAX_NEW_TOKENS[task],
                                STOPS[task])
            for it, spec, out in zip(batch, specs, outs):
                correct = score_prediction(task, it.answer, out)
                # everything needed to re-score, re-analyse or audit this prediction
                # later without re-running the model
                writer.append(
                    {
                        "uid": it.uid,
                        "model": model,
                        "model_tokenizer_key": model_tokenizer_key,
                        "task": task,
                        "language": lang.code,
                        "condition": cond,
                        "budget": budget,
                        "n_shots": spec.n_shots,
                        "exemplar_uids": [s.uid for s in shots[: spec.n_shots]],
                        "n_prompt_tokens": spec.n_prompt_tokens,
                        "prompt_truncated": spec.truncated,
                        "prompt_sha1": _sha1(spec.text),
                        "gold": it.answer,
                        "output": out,
                        "extracted": _extracted_answer(task, out),
                        "n_output_chars": len(out),
                        "correct": correct,
                    }
                )
                if prompt_writer is not None:
                    prompt_writer.append({"uid": it.uid, "prompt_sha1": _sha1(spec.text),
                                          "prompt": spec.text})
                done.add(it.uid)
            writer.flush()
            if prompt_writer is not None:
                prompt_writer.flush()
    finally:
        writer.close()
        if prompt_writer is not None:
            prompt_writer.close()

    rows = read_jsonl(pred_path)
    rows = list({r["uid"]: r for r in rows}.values())
    n = len(rows)
    record = {
        "model": model or getattr(gen, "model", ""),
        "model_tokenizer_key": model_tokenizer_key,
        "task": task,
        "language": lang.code,
        "language_name": lang.name,
        "family": lang.family,
        "morph_type": lang.morph_type,
        "agglutinative": lang.agglutinative,
        "condition": cond,
        "budget": budget,
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / max(1, n),
        "mean_shots": sum(r["n_shots"] for r in rows) / max(1, n),
        "mean_prompt_tokens": sum(r["n_prompt_tokens"] for r in rows) / max(1, n),
        "truncated_rate": sum(bool(r["prompt_truncated"]) for r in rows) / max(1, n),
        "predictions": paths.relpath(pred_path),
    }
    ctx.commit(_unit_key(task, lang.code, cond, model), record)
    log.info("%s/%s/%s: acc=%.3f shots=%.2f prompt_tokens=%.0f n=%d",
             task, lang.code, cond, record["accuracy"], record["mean_shots"],
             record["mean_prompt_tokens"], n)


def aggregate_part_c(ctx: RunContext) -> pd.DataFrame:
    records = [r for r in ctx.records() if r.get("n")]
    if not records:
        log.warning("Part C: no records to aggregate")
        return pd.DataFrame()
    df = pd.DataFrame(records).drop(columns=["_unit", "_ts"], errors="ignore")
    keys = [c for c in ("model", "task", "language", "condition") if c in df.columns]
    df = df.drop_duplicates(subset=keys, keep="last")
    out_csv = ctx.path("capability_increment.csv")
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(ctx.path("capability_increment.parquet"), index=False)
    except Exception as exc:  # pragma: no cover
        log.warning("parquet export failed: %s", exc)

    # budget-loss table: unconstrained accuracy minus budgeted accuracy
    idx = [c for c in ("model", "task", "language") if c in df.columns]
    base = (df[df.condition == "unconstrained"]
            .set_index(idx)["accuracy"].rename("accuracy_unconstrained"))
    loss = df[df.condition != "unconstrained"].join(base, on=idx)
    if not loss.empty:
        loss["accuracy_loss"] = loss.accuracy_unconstrained - loss.accuracy
        loss["relative_loss"] = loss.accuracy_loss / loss.accuracy_unconstrained.clip(lower=1e-6)
        loss.to_csv(ctx.path("budget_loss.csv"), index=False)
    log.info("Part C summary -> %s (%d rows)", paths.relpath(out_csv), len(df))
    return df
