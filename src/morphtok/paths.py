"""Repo-local paths and cache redirection.

Every byte this project downloads (datasets, tokenizers, model weights, matplotlib
font cache, torch/triton/vllm caches) lands under the repository root.  Nothing is
written to ``$HOME``.

``setup_env()`` MUST be called before ``transformers`` / ``datasets`` / ``tiktoken`` /
``torch`` / ``vllm`` are imported, because those libraries read their cache
locations at import time.  ``morphtok.cli`` does this as its first action.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# repo root
# ---------------------------------------------------------------------------
# .../<repo>/src/morphtok/paths.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_root() -> Path:
    override = os.environ.get("MORPHTOK_ROOT")
    return Path(override).resolve() if override else REPO_ROOT


ROOT = _env_root()

# top-level layout -----------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # untouched downloads
PROC_DIR = DATA_DIR / "processed"  # parsed/derived corpora
GOLD_DIR = DATA_DIR / "gold"  # released gold morpheme segmentations
RESOURCES_DIR = DATA_DIR / "resources"  # morfessor models, unimorph dumps

MODELS_DIR = ROOT / "models"  # HF hub cache + vLLM download dir
CACHE_DIR = ROOT / ".cache"  # every other library cache
RUNS_DIR = ROOT / "runs"  # per-run outputs (resumable)
RESULTS_DIR = ROOT / "results"  # aggregated tables + figures
LOGS_DIR = ROOT / "logs"
TMP_DIR = CACHE_DIR / "tmp"

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

ALL_DIRS = [
    DATA_DIR,
    RAW_DIR,
    PROC_DIR,
    GOLD_DIR,
    RESOURCES_DIR,
    MODELS_DIR,
    CACHE_DIR,
    RUNS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    TMP_DIR,
    TABLES_DIR,
    FIGURES_DIR,
]

_ENV_READY = False


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# a socket-safe TMPDIR
# ---------------------------------------------------------------------------
# A Unix-domain socket address may not exceed ``sizeof(sockaddr_un.sun_path)`` =
# 108 bytes (107 usable).  vLLM's engine-core handshake binds an IPC socket at
# ``$TMPDIR/<uuid4>``, so pointing TMPDIR into the repository — which is what the
# rest of this module exists to do — only works while the repository path is short.
# This repository's path is not: ``<root>/.cache/tmp`` is 92 characters, leaving 15
# for a 36-character uuid, and every vLLM launch died with
#
#     ZMQError: ipc path "...cache/tmp/981c90e7-..." is longer than 107 characters
#
# before a single token was generated.  The fix keeps the invariant: a *symlink* to
# the repo-local tmp directory is created in a short location and TMPDIR points at
# the link, so the socket files themselves still live inside the repository and only
# a zero-byte pointer sits outside it.
IPC_PATH_MAX = 107
#: uuid4 (36) + separator + slack for vLLM's own suffixes (``_input_socket`` etc.).
IPC_NAME_BUDGET = 48

#: Set by :func:`setup_short_tmpdir`; logged by ``logging_utils.log_env_summary`` so
#: the redirection is visible in every stage log rather than silent.
TMPDIR_NOTE: str | None = None


def _tmpdir_fits(path: Path | str) -> bool:
    return len(str(path)) + IPC_NAME_BUDGET <= IPC_PATH_MAX


def _ipc_link_bases() -> list[str]:
    bases = [os.environ.get("MORPHTOK_IPC_ROOT") or ""]
    uid = getattr(os, "getuid", lambda: None)()
    if uid is not None:
        bases.append(f"/run/user/{uid}")
    bases += [tempfile.gettempdir(), "/tmp", "/var/tmp"]
    return [b for b in dict.fromkeys(bases) if b]


def setup_short_tmpdir() -> str:
    """Point ``TMPDIR`` at a directory short enough to hold a socket path.

    Returns the directory now in use.  When the repo-local ``.cache/tmp`` already
    fits, nothing is changed and it is returned unmodified.  Otherwise a symlink to
    it is created under ``$MORPHTOK_IPC_ROOT``, ``/run/user/<uid>``, ``/tmp`` or
    ``/var/tmp`` (first one that both works and is short enough), and ``TMPDIR``,
    ``TMP``, ``TEMP`` and ``VLLM_RPC_BASE_PATH`` all point at the link.

    Idempotent: the link name is derived from the repository path, so repeated calls
    and concurrent runs of *this* repo reuse one link, while a different checkout
    gets its own.
    """
    global TMPDIR_NOTE
    ensure_dirs()
    target = TMP_DIR
    chosen: Path | None = target if _tmpdir_fits(target) else None

    if chosen is None:
        name = "mtok-" + hashlib.sha1(str(ROOT).encode("utf-8")).hexdigest()[:8]
        attempts: list[str] = []
        for base in _ipc_link_bases():
            link = Path(base) / name
            if not _tmpdir_fits(link):
                attempts.append(f"{link}: still too long")
                continue
            try:
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    if Path(os.readlink(link)).resolve() != target.resolve():
                        link.unlink()
                        os.symlink(target, link)
                elif link.exists():  # a real file or directory squats on the name
                    attempts.append(f"{link}: exists and is not our symlink")
                    continue
                else:
                    os.symlink(target, link)
                if link.resolve() != target.resolve():
                    attempts.append(f"{link}: resolves elsewhere")
                    continue
                # prove it before trusting it: the socket, not the symlink, is what
                # has to work
                probe = link / ".ipc-probe"
                probe.write_text("", encoding="utf-8")
                probe.unlink()
            except OSError as exc:
                attempts.append(f"{link}: {type(exc).__name__}: {exc}")
                continue
            chosen = link
            TMPDIR_NOTE = (f"TMPDIR -> {link} (symlink to {relpath(target)}; the repo "
                           f"path is {len(str(target))} chars, over the {IPC_PATH_MAX}-"
                           "char Unix-socket limit)")
            break
        if chosen is None:
            chosen = target
            TMPDIR_NOTE = ("could not shorten TMPDIR (" + "; ".join(attempts) + ") — "
                           "vLLM will fail to bind its IPC socket; set "
                           "MORPHTOK_IPC_ROOT to a short writable directory")

    value = str(chosen)
    for key in ("TMPDIR", "TMP", "TEMP", "VLLM_RPC_BASE_PATH"):
        os.environ[key] = value
    # ``tempfile`` caches gettempdir() on first use, so the env var alone is not
    # enough once anything in-process has already asked for a temp file.
    tempfile.tempdir = value
    return value


def _hf_token() -> str | None:
    """Token from the environment or a gitignored ``.hf_token`` file in the repo."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    token_file = ROOT / ".hf_token"
    if token_file.is_file():
        val = token_file.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


def setup_env(quiet: bool = True) -> dict[str, str]:
    """Point all third-party caches into the repo. Idempotent."""
    global _ENV_READY
    ensure_dirs()

    hub_cache = MODELS_DIR / "hf-hub"
    env = {
        # --- huggingface -----------------------------------------------------
        "HF_HOME": str(MODELS_DIR / "hf-home"),
        "HUGGINGFACE_HUB_CACHE": str(hub_cache),
        "HF_HUB_CACHE": str(hub_cache),
        "TRANSFORMERS_CACHE": str(hub_cache),
        "HF_DATASETS_CACHE": str(CACHE_DIR / "hf-datasets"),
        "HF_MODULES_CACHE": str(CACHE_DIR / "hf-modules"),
        "HF_METRICS_CACHE": str(CACHE_DIR / "hf-metrics"),
        "HF_ASSETS_CACHE": str(CACHE_DIR / "hf-assets"),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        # --- tokenizers / tiktoken ------------------------------------------
        "TIKTOKEN_CACHE_DIR": str(CACHE_DIR / "tiktoken"),
        "DATA_GYM_CACHE_DIR": str(CACHE_DIR / "tiktoken"),
        "TOKENIZERS_PARALLELISM": "false",
        # --- torch / vllm / triton ------------------------------------------
        "TORCH_HOME": str(CACHE_DIR / "torch"),
        "TORCHINDUCTOR_CACHE_DIR": str(CACHE_DIR / "torchinductor"),
        "TRITON_CACHE_DIR": str(CACHE_DIR / "triton"),
        "VLLM_CACHE_ROOT": str(CACHE_DIR / "vllm"),
        "VLLM_CONFIG_ROOT": str(CACHE_DIR / "vllm-config"),
        "VLLM_ASSETS_CACHE": str(CACHE_DIR / "vllm-assets"),
        "OUTLINES_CACHE_DIR": str(CACHE_DIR / "outlines"),
        "XDG_CACHE_HOME": str(CACHE_DIR / "xdg"),
        "XDG_CONFIG_HOME": str(CACHE_DIR / "xdg-config"),
        "XDG_DATA_HOME": str(CACHE_DIR / "xdg-data"),
        # --- misc -------------------------------------------------------------
        "MPLCONFIGDIR": str(CACHE_DIR / "matplotlib"),
        "NLTK_DATA": str(RESOURCES_DIR / "nltk"),
        "TMPDIR": str(TMP_DIR),
        "UV_CACHE_DIR": str(CACHE_DIR / "uv"),
        "UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv"),
    }
    for key, value in env.items():
        os.environ[key] = value
        if key.endswith(("CACHE", "CACHE_DIR", "HOME", "TMPDIR", "NLTK_DATA", "ROOT")):
            try:
                Path(value).mkdir(parents=True, exist_ok=True)
            except OSError:  # pragma: no cover - defensive
                pass

    # must run after the loop above: it overrides TMPDIR with a socket-safe path
    env["TMPDIR"] = setup_short_tmpdir()
    env["VLLM_RPC_BASE_PATH"] = os.environ["VLLM_RPC_BASE_PATH"]

    token = _hf_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    _ENV_READY = True
    if not quiet:  # pragma: no cover - debugging aid
        for key in sorted(env):
            print(f"{key}={env[key]}")
    return env


def env_is_ready() -> bool:
    return _ENV_READY


# ---------------------------------------------------------------------------
# run directories
# ---------------------------------------------------------------------------
def run_dir(run_id: str, stage: str | None = None) -> Path:
    """``runs/<run_id>[/<stage>]``, created on demand."""
    d = RUNS_DIR / run_id
    if stage:
        d = d / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def relpath(p: Path | str) -> str:
    """Path relative to the repo root when possible (for tidy logs)."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(p)
