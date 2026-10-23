#!/usr/bin/env bash
# =============================================================================
#  Morphology, Not Frequency, Should Govern Vocabulary Allocation
#  One entry point: environment setup (uv) + every experiment stage.
#
#  Nothing is ever written outside this repository: uv, its cache, the virtual
#  environment, HuggingFace models, datasets, torch/vLLM caches and logs all live
#  under ./.tools, ./.venv, ./.cache, ./models, ./data, ./runs, ./logs.
#
#  Usage:
#     ./run.sh                       # full pipeline (fetch + A + B + C + D + analysis)
#     ./run.sh setup                 # only create the environment
#     ./run.sh fetch                 # only download data/models/analyzers
#     ./run.sh part-a  [-- args]     # single stage; extra args go to morphtok
#     ./run.sh part-b
#     ./run.sh part-c  --model Qwen/Qwen3-8B
#     ./run.sh part-d                # morpheme-constrained vs frequency BPE prototype
#     ./run.sh analyze | figures | runs | env-info
#     ./run.sh verify --run-id latest    # correctness/consistency/completeness audit
#     ./run.sh smoke                 # 10-minute CPU end-to-end check
#     ./run.sh all --tag ablation1   # separable run: new timestamped run id + tag
#     ./run.sh resume part-c         # resume the most recent run's stage
#
#  Environment knobs:
#     MORPHTOK_EXTRAS=gpu|cpu-infer|none   dependency group (default: auto-detect)
#     MORPHTOK_MODEL=Qwen/Qwen3-8B         one Part C model instead of the sweep
#     MORPHTOK_MODELS=a,b                  explicit Part C model list (sequential)
#     MORPHTOK_IPC_ROOT=/tmp               where the short TMPDIR symlink is made
#     HF_TOKEN=hf_xxx                      needed only for gated tokenizers
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- repo-local env
export MORPHTOK_ROOT="$REPO_ROOT"
export UV_CACHE_DIR="$REPO_ROOT/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.tools/python"
export UV_TOOL_DIR="$REPO_ROOT/.tools/uv-tools"
export UV_INSTALL_DIR="$REPO_ROOT/.tools/bin"
export XDG_CACHE_HOME="$REPO_ROOT/.cache/xdg"
export XDG_DATA_HOME="$REPO_ROOT/.cache/xdg-data"
export XDG_CONFIG_HOME="$REPO_ROOT/.cache/xdg-config"
export HF_HOME="$REPO_ROOT/models/hf-home"
export HF_HUB_CACHE="$REPO_ROOT/models/hf-hub"
export HUGGINGFACE_HUB_CACHE="$REPO_ROOT/models/hf-hub"
export HF_DATASETS_CACHE="$REPO_ROOT/.cache/hf-datasets"
export TIKTOKEN_CACHE_DIR="$REPO_ROOT/.cache/tiktoken"
export TORCH_HOME="$REPO_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache/triton"
export VLLM_CACHE_ROOT="$REPO_ROOT/.cache/vllm"
export VLLM_CONFIG_ROOT="$REPO_ROOT/.cache/vllm-config"
export MPLCONFIGDIR="$REPO_ROOT/.cache/matplotlib"
export TMPDIR="$REPO_ROOT/.cache/tmp"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

mkdir -p "$REPO_ROOT"/{data,models,runs,results,logs,.cache/tmp,.tools/bin}

if [[ -f "$REPO_ROOT/.hf_token" && -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$REPO_ROOT/.hf_token")"
  export HF_TOKEN
fi

TS="$(date +%Y%m%d-%H%M%S)"
SESSION_LOG="$REPO_ROOT/logs/run_${TS}.log"

log() { printf '\033[1;34m[run.sh %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\033[1;31m[run.sh ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- uv bootstrap
ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
  elif [[ -x "$REPO_ROOT/.tools/bin/uv" ]]; then
    UV="$REPO_ROOT/.tools/bin/uv"
  else
    log "uv not found — installing into $REPO_ROOT/.tools/bin (not \$HOME)"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$REPO_ROOT/.tools/bin" sh
    UV="$REPO_ROOT/.tools/bin/uv"
  fi
  [[ -x "$UV" ]] || die "uv installation failed"
  log "uv: $UV ($($UV --version))"
}

detect_extras() {
  if [[ -n "${MORPHTOK_EXTRAS:-}" ]]; then
    echo "$MORPHTOK_EXTRAS"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "gpu"
  else
    echo "none"
  fi
}

setup_env() {
  ensure_uv
  local extras; extras="$(detect_extras)"
  log "syncing dependencies (extras=$extras) into $UV_PROJECT_ENVIRONMENT"
  if [[ "$extras" == "none" ]]; then
    "$UV" sync
  else
    "$UV" sync --extra "$extras"
  fi
  log "environment ready: $($UV run python -c 'import sys; print(sys.version.split()[0])')"
  if [[ "$extras" == "gpu" ]]; then
    "$UV" run python - <<'PY' || true
import torch
print(f"[run.sh] torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[run.sh]   GPU{i}: {p.name} {p.total_memory/1e9:.1f} GB")
PY
  fi
}

mt() {  # run the CLI inside the project environment
  ensure_uv
  "$UV" run --no-sync morphtok "$@"
}

# ---------------------------------------------------------------- stages
# Part C evaluates the CLI's default two-model sweep (Qwen3-8B + Gemma-3-12B) unless
# MORPHTOK_MODEL (one model) or MORPHTOK_MODELS (a comma list) overrides it.  Nothing
# is passed when both are unset, so the default lives in one place: morphtok.cli.
model_args() {
  if [[ -n "${MORPHTOK_MODELS:-}" ]]; then
    printf '%s\n' --models "$MORPHTOK_MODELS"
  elif [[ -n "${MORPHTOK_MODEL:-}" ]]; then
    printf '%s\n' --model "$MORPHTOK_MODEL"
  fi
}

stage_fetch()    { mt fetch-data "$@"; }
stage_part_a()   { mt part-a "$@"; }
stage_part_b()   { mt part-b "$@"; }
stage_part_c()   { local m=(); mapfile -t m < <(model_args); mt part-c "${m[@]}" "$@"; }
stage_analyze()  { mt analyze "$@"; }
stage_figures()  { mt figures "$@"; }

stage_all() {
  local m=(); mapfile -t m < <(model_args)
  mt all "${m[@]}" "$@"
}

stage_smoke() {
  log "SMOKE TEST — 3 tokenizers, 3 languages, 40 sentences, 120 gold types, no GPU"
  local rid="smoke-${TS}"
  mt fetch-data --langs eng,hin,tam --tasks belebele --gold-types 120
  mt part-a --run-id "$rid" --tokenizers o200k,qwen3,sarvam1 --langs eng,hin,tam \
            --max-sentences 40
  mt part-b --run-id "$rid" --tokenizers o200k,qwen3,sarvam1 --langs hin,tam
  mt part-d --run-id "$rid" --langs hin --vocab-size 4000 --max-types 40000
  mt analyze --run-id "$rid"
  mt figures --run-id "$rid"
  mt verify  --run-id "$rid"
  log "smoke test finished — run id $rid"
  log "tables:  results/tables/   figures: results/figures/"
}

usage() { sed -n '2,32p' "$BASH_SOURCE"; }

main() {
  local cmd="${1:-all}"; shift || true
  case "$cmd" in
    setup)      setup_env ;;
    fetch)      setup_env; stage_fetch "$@" ;;
    part-a|a)   setup_env; stage_part_a "$@" ;;
    part-b|b)   setup_env; stage_part_b "$@" ;;
    part-c|c)   setup_env; stage_part_c "$@" ;;
    part-d|d)   setup_env; mt part-d "$@" ;;
    analyze)    setup_env; stage_analyze "$@" ;;
    figures)    setup_env; stage_figures "$@" ;;
    gold)       setup_env; mt gold "$@" ;;
    runs)       setup_env; mt runs ;;
    verify)     setup_env; mt verify "$@" ;;
    env-info)   setup_env; mt env-info --probe-tokenizers ;;
    smoke)      setup_env; stage_smoke ;;
    resume)     local st="${1:-}"; shift || true
                [[ -n "$st" ]] || die "usage: ./run.sh resume <part-a|part-b|part-c>"
                setup_env; mt "$st" --run-id latest "$@" ;;
    all)        setup_env; stage_all "$@" ;;
    -h|--help|help) usage ;;
    *)          die "unknown command '$cmd' (try ./run.sh --help)" ;;
  esac
}

log "session log: logs/$(basename "$SESSION_LOG")"
main "$@" 2>&1 | tee -a "$SESSION_LOG"
