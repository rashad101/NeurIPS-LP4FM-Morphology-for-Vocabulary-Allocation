# Position: Morphology, Not Frequency, Should Govern Vocabulary Allocation

PyTorch code for The Neurips Workshop - Linguistic Principles for Foundation Models (LP4FM) 2026 position paper: Position: Morphology, Not Frequency, Should Govern Vocabulary Allocation


| Part | What it produces | Hardware | Wall time (full) |
|------|------------------|----------|------------------|
| **A** — fertility audit + vocabulary-allocation audit | `indic_audit.csv`, premium/fertility matrices | CPU | ~40 min (12 tokenizers × 21 languages × 1 012 sentences) |
| **B** — morpheme-boundary alignment (**the core**) | `morph_alignment.csv`, per-word parquets, gold segmentations | CPU | ~45 min (12 × 19 × 2 000 word types) |
| **C** — fixed-budget capability | `capability_increment.parquet`, per-item predictions | 1× GPU (24–48 GB) | 3–8 h (~17k generations) |
| **D** — morphology-governed allocator prototype | `prototype_comparison.csv`, trained tokenizers | CPU | ~1 min per language |
| analysis + figures | H1–H4 regressions across 3 robustness scopes, 6 figures | CPU | < 3 min |

Everything — datasets, tokenizers, model weights, caches, logs — is written **inside
this repository**. Nothing touches `$HOME`.

---

## 1. Quick start

```bash
git clone <this repo> && cd Position-Morphology-*
./run.sh setup            # installs uv (repo-local if missing) + the environment
./run.sh smoke            # ~10 min CPU end-to-end check, writes results/
./run.sh all              # full pipeline: fetch → A → B → C → analyze → figures
```

On the 48 GB GPU box:

```bash
./run.sh all --tag paper-v1             # Part C sweeps Qwen3-8B + Gemma-3-12B
# MORPHTOK_MODEL=Qwen/Qwen3-8B ./run.sh all   # one Part C model instead of the sweep
```

`run.sh` auto-detects CUDA (`nvidia-smi`) and installs the `gpu` extra (torch + vLLM);
without a GPU it installs the CPU-only set and Part C can still be dry-run.

### The environment

* **uv** manages everything. If `uv` is missing, `run.sh` installs it into
  `.tools/bin` — *not* `~/.local/bin`.
* Python 3.11 (pinned in `.python-version`), venv at `./.venv`.
* Dependency groups: default (CPU: tokenizers, morphology, stats, figures),
  `--extra gpu` (torch + vLLM), `--extra cpu-infer` (torch only, for smoke-testing
  Part C without CUDA). Override detection with `MORPHTOK_EXTRAS=gpu|cpu-infer|none`.

### Everything stays in the repo

`run.sh` and `morphtok.paths.setup_env()` redirect every cache:

| Variable | Location |
|---|---|
| `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` | `models/hf-home`, `models/hf-hub` |
| vLLM weights (`download_dir`) | `models/vllm` |
| `HF_DATASETS_CACHE`, `TIKTOKEN_CACHE_DIR`, `TORCH_HOME`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `MPLCONFIGDIR`, `XDG_*`, `TMPDIR` | `.cache/…` |
| `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_INSTALL_DIR` | `.cache/uv`, `.tools/…` |

Verify at any time: `./run.sh env-info` (add `--probe-tokenizers` to test every
tokenizer download).

**One deliberate exception: a symlink.** A Unix-domain socket address cannot exceed
107 characters, and vLLM binds its engine-core handshake socket at `$TMPDIR/<uuid4>`.
This repository's path is 80 characters, so a repo-local `TMPDIR` leaves no room for a
36-character uuid and every vLLM launch died before loading a single weight:

```
ZMQError: ipc path ".../.cache/tmp/981c90e7-..." is longer than 107 characters
```

`paths.setup_short_tmpdir()` therefore creates a symlink — `/run/user/<uid>/mtok-<hash
of the repo path>`, falling back to `/tmp` or `/var/tmp` — pointing at `.cache/tmp`,
and points `TMPDIR`/`TMP`/`TEMP`/`VLLM_RPC_BASE_PATH` at the link. The socket files
are still created **inside** the repository; only a zero-byte pointer lives outside
it, it is reused across runs of the same checkout, and the redirection is printed in
every stage log (`TMPDIR note = ...`). Set `MORPHTOK_IPC_ROOT` to choose a different
short directory. Nothing happens at all when the repo path is short enough to fit.

### HuggingFace token (optional)

The pipeline runs without one: 11 of 13 tokenizers are ungated, gated repos resolve
to ungated community mirrors **of the same vocabulary**, and every fallback is logged.

```bash
echo "hf_xxx" > .hf_token        # gitignored; or export HF_TOKEN=hf_xxx
```

A token is read from `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`
or the gitignored `.hf_token` file, in that order (`paths._hf_token()`). Never commit
it and never inline it in source — `.hf_token` exists precisely so it stays out of
version control. Providing one changes three things:

| | without a token | with a token |
|---|---|---|
| `aya_expanse` | **skipped** — both repos gated, audit has 11 tokenizers | included (`CohereLabs/aya-expanse-8b`, 255 000) — 12 tokenizers |
| `llama31`, `gemma3` | ungated mirrors (`NousResearch/…`, `unsloth/…`) | canonical `meta-llama/…`, `google/…` |
| Part C Gemma model | `unsloth/gemma-3-12b-it` | `google/gemma-3-12b-it` |

`tokenizer_info.csv` records the repo actually used for every row, so a table built
either way states its own provenance.

---

## 2. Command reference

Every stage is a separate subcommand, so runs are separable **by time** (each run gets
a timestamped `run_id`) and **by algorithm** (`--tokenizers`, `--langs`, `--tasks`,
`--conditions`, `--model`).

```bash
./run.sh <stage> [args]          # wrapper: sets env, syncs deps, tees a session log
uv run morphtok <stage> [args]   # same thing without the wrapper
```

| Stage | Purpose |
|---|---|
| `env-info` | print repo-local cache paths, disk usage, tokenizer availability |
| `fetch-data` | download corpora, analyzers, fonts, task data; build gold segmentations |
| `gold` | rebuild gold morpheme segmentations only |
| `part-a` | Indic fertility audit |
| `part-b` | morpheme-boundary alignment |
| `part-c` | fixed-budget capability study (GPU) |
| `part-d` | morpheme-constrained vs frequency BPE prototype |
| `analyze` | H1–H4 regressions and tables (3 robustness scopes) |
| `verify` | audit a run: correctness, consistency, completeness |
| `figures` | the paper's six figures |
| `all` | fetch → A → B → C → D → analyze → figures under one run id |
| `runs` | list runs and per-stage status |

### 2.1 Data

```bash
./run.sh fetch                                   # all 21 languages, both Part C tasks
./run.sh fetch --langs indic --skip-tasks        # corpora + morphology only
./run.sh gold --langs morph --gold-types 2000 --force
./run.sh gold --langs ory,swh --train-morfessor  # fallback analyzer (see §4)
```

Language selectors: ISO-639-3 codes (`hin,tam`), FLORES codes (`hin_Deva`), or the
groups `all`, `indic`, `core` (hin, mar, tam, tel, kan), `morph` (any gold analyzer),
`annotated` (hand-curated gold only), `control` (the non-Indic typological controls),
`typology` (morph + control), `mgsm`, `belebele`.

### 2.2 Part A — Indic audit (CPU)

```bash
./run.sh part-a --tag audit                       # all tokenizers × all languages
./run.sh part-a --tokenizers specialized --langs indic
./run.sh part-a --run-id latest                   # resume; finished pairs are skipped
./run.sh part-a --max-sentences 120               # fast pilot
```

Outputs in `runs/<run_id>/part_a/`: `records.jsonl` (one line per finished pair),
`indic_audit.csv`, `premium_matrix.csv`, `fertility_matrix.csv`, `tokenizer_info.csv`,
`tokenizer_duplicates.csv`, `sentences/<tok>__<lang>.parquet` (per-sentence token
counts, used for the aligned-sentence premium) and `words/<tok>__<lang>.parquet` (every
word type with its count, token count and mid-character splits; `--no-save-word-stats`
to skip).

The loop is **tokenizer-major and evicts each tokenizer before loading the next**, so
peak memory is one vocabulary rather than thirteen — Parts A/B can share a box with a
running vLLM job. Evictions are logged (`unloaded qwen3; resident tokenizers: []`).

Key columns: `fertility` (tokens/word, corpus-level), `fertility_macro`,
`premium_corpus` / `premium_median` (tokens relative to the aligned English sentence),
`effective_context_ratio`, `fragmentation_rate` (share of word occurrences split into
>1 token), `single_token_word_rate`, `mid_character_split_rate` (byte-level splits
*inside* a single Indic character), `byte_fallback_rate`, `chars_per_token`.

### 2.3 Part B — boundary alignment (CPU, the core analysis)

```bash
./run.sh part-b --tag align                        # gold languages × all tokenizers
./run.sh part-b --langs core --tokenizers mainstream,specialized
./run.sh part-b --tolerance 0                      # exact-only F1 (±1 is the default)
```

Outputs in `runs/<run_id>/part_b/`: `morph_alignment.csv`, `boundary_f1_matrix.csv`,
`qualitative_examples.json` (Figure 3 source), `words/<tok>__<lang>.parquet` (every
per-word alignment decision — part of the released artifact).

Key columns: `boundary_precision/recall/f1` (type-weighted) and `*_freq`
(frequency-weighted), `boundary_f1_tol1`, `morpheme_integrity` (share of gold morphemes
wholly inside one token), `token_purity`, `mono_split_rate` (monomorphemic words split
anyway), `oversegmentation_ratio`, `tokens_per_gold_word`, `boundary_f1_tier{A,B,C,D}`
(analyzer-noise robustness), `unalignable_rate`.

### 2.4 Part C — capability under a fixed budget (GPU)

```bash
# ALWAYS first — needs no GPU, shows how many exemplars each language actually fits
./run.sh part-c --dry-run --tasks belebele,mgsm --langs mgsm,indic,eng

# the real run (48 GB+): the default two-model sweep, 250 items per lang per condition
./run.sh part-c --tag caps --n-items 250

# alternatives
./run.sh part-c --model Qwen/Qwen3-8B                              # one model only
./run.sh part-c --models Qwen/Qwen3-8B,CohereLabs/aya-expanse-8b   # a different pair
./run.sh part-c --backend hf --model Qwen/Qwen3-0.6B --n-items 8   # CPU smoke test
./run.sh resume part-c                                             # resume last run
```

**Two models by default.** Part C evaluates `Qwen/Qwen3-8B` **and**
`unsloth/gemma-3-12b-it` (the ungated mirror of `google/gemma-3-12b-it`; Part A audits
the same vocabulary). This is a *tokenizer* contrast, not a capability contrast, and
H3 needs it: with one model the boundary-F1 term has no variation to explain except
the language axis itself. The two vocabularies differ sharply on exactly the languages
at issue — Qwen3 gives Malayalam 0.01 % of its 151k byte-level BPE vocabulary, Gemma-3
gives it 0.75 % of its 262k SentencePiece one — and the effect is visible before a
single token is generated. From `--dry-run` at `budget1024` on Belebele:

| model | eng | hin | tam | tel |
|---|---|---|---|---|
| Qwen3-8B | 4 shots | **0** | **0** | **0** |
| Gemma-3-12B | 4 shots | 3 | 3 | 2 |

Under an identical 1 024-token budget one model still gets few-shot prompting in
Indic languages and the other is reduced to zero-shot. `--models` overrides the pair;
`--model` runs a single one. If one model fails to load, the sweep logs it, records it
in `status.json` and continues with the rest — losing the second model should not cost
the first model's results.

Conditions: `budget1024`, `budget2048`, `unconstrained` (k=5). Under a budget,
exemplars are added greedily while the prompt fits; if the question alone overflows,
the prompt is truncated from the left and `prompt_truncated` is recorded — that *is*
the context-erosion mechanism being measured.

**Several models, one at a time.** The default sweep (and any `--models` list) runs
them sequentially and tears each one down before the next is built — vLLM's KV cache and model-parallel state are
released explicitly (`destroy_model_parallel`, `empty_cache`), because dropping the
Python reference alone leaves the GPU allocated and the second model OOMs:

```bash
./run.sh part-c --tag caps --models Qwen/Qwen3-8B,unsloth/gemma-3-12b-it
```

Free GPU memory is logged after each teardown, and the teardown also runs on Ctrl-C or
a crash. Unit keys and prediction filenames are model-scoped, so all models can share
one run id and `analyze` pools them with a model fixed effect.

Outputs in `runs/<run_id>/part_c/`:
`predictions/<model>__<task>__<lang>__<cond>.jsonl` (appended and fsynced after every
batch), `capability_increment.csv/.parquet`, `budget_loss.csv`, and — with
`--save-prompts` — `prompts/*.jsonl` with every rendered prompt.

Each prediction row carries everything needed to re-analyse or audit it later without
the GPU: `uid`, `model`, `model_tokenizer_key`, `task`, `language`, `condition`,
`budget`, `n_shots`, `exemplar_uids`, `n_prompt_tokens`, `prompt_truncated`,
`prompt_sha1`, `gold`, the **full** `output`, the `extracted` answer the scorer read,
and `correct`. `morphtok verify` re-derives every accuracy from these files.

### 2.5 Part D — the prototype allocator (CPU)

Only one language's tokenizers are resident at a time; the trained arms and the
Morfessor Viterbi model are dropped before the next language.


```bash
./run.sh part-d --langs core --vocab-size 16000     # 5 languages, ~5 min
./run.sh part-d --langs morph --vocab-size 32000
```

Three arms share a corpus, training stream, trainer and vocabulary budget (the
constrained arm saturates first and the baseline is trained to *its* realised size, so
the budget is matched by construction):

| arm | what governs allocation |
|---|---|
| `baseline` | frequency alone (ordinary byte-level BPE) |
| `morph_train` | merges *learned* only within morphemes; encoding still frequency-driven |
| `morph_full` | same vocabulary + morphology-aware pre-tokenization at encoding time |

Outputs in `runs/<run_id>/part_d/`: `prototype_arms.csv`, `prototype_comparison.csv`
(per-arm deltas), and the trained tokenizers under `tokenizers/<lang>/*.json`.

### 2.6 Analysis and figures

```bash
./run.sh analyze --run-id latest        # or a specific run id
./run.sh analyze --include-byte         # keep ByT5/UTF-8 in the regressions
./run.sh figures --run-id latest
```

Tables land in `results/tables/` (CSV + Markdown), figures in `results/figures/`
(PNG 300 dpi + PDF). Analysis can mix stages from different runs: if the requested run
lacks a stage, the most recent run that has it is used (and logged).

---

## 3. What is computed

### H1 — boundary misalignment and fertility, with typological structure

`results/tables/h1_fertility_regressions.csv` (OLS, SEs clustered by language):

```
M1  fertility ~ boundary_f1
M2  fertility ~ boundary_f1 + log10_wiki_articles
M3  fertility ~ boundary_f1 + log10_wiki_articles + C(family)
M4  fertility ~ boundary_f1 * agglutinative + log10_wiki_articles      ← headline
M5  M4 + C(tokenizer)         (tokenizer fixed effects)
M6  fertility ~ morpheme_integrity * agglutinative + log10_wiki_articles
M7  M4 + log10_script_vocab   ← the vocabulary-budget control
M8  M7 + C(tokenizer)
```

Every model is estimated in **three scopes**, because the two obvious attacks on H1
are "your gold is just another unsupervised segmenter" and "this is an Indic
artifact":

| scope | sample | answers |
|---|---|---|
| `indic` | Indo-Aryan + Dravidian only | the paper's primary claim |
| `all` | + Hungarian, Turkish, Mongolian, Czech, Russian, Spanish, German, Swahili | is the agglutinative slope Indic-specific? |
| `annotated` | `boundary_f1_annotated` — gold from SIGMORPHON/UniMorph only, never Morfessor; controls for gold source | is the effect an artifact of the analyzer? |

`h1_family_slopes.csv` gives simple slopes per morphological type per scope, and
`h1_gold_provenance.csv` states, per language, how much of the gold is annotated
rather than analyzer-derived. The resource control is Wikipedia article count (live
from the Wikimedia API, cached in `data/processed/wiki_sizes.json`) — a proxy, and
labelled as one. Byte-level tokenizers are excluded unless `--include-byte`.

**M4 vs M7 is itself a finding**: the vocabulary a tokenizer spends on a
script explains fertility on its own, and boundary alignment adds little to it. Do
not report M4 without M7.

### H2 — specialized tokenizers mark the attainable frontier

`h2_specialized_vs_mainstream.csv` (Welch t + Mann-Whitney + Cohen's d on Indic rows),
`h2_frontier_by_tokenizer.csv`, `h2_fertility_by_group_language.csv`.

**Budget control** (`h2_budget_control.csv`, `h2_vocabulary_allocation.csv`) — the
standard objection is that specialized tokenizers simply spend more of a fixed
allowance on Indic scripts. Part A therefore measures that allowance directly:
`script_vocab_count` / `script_vocab_share` count the vocabulary entries actually
written in each language's script. The models ask whether specialization buys
*alignment* beyond compression and budget:

```
B1  morpheme_integrity ~ is_specialized
B2  morpheme_integrity ~ fertility + is_specialized
B3  morpheme_integrity ~ fertility + log10_script_vocab + is_specialized   ← headline
B4  boundary_f1        ~ fertility + log10_script_vocab + is_specialized
B5  fertility ~ log10_script_vocab + log10_wiki_articles
B6  B5 + boundary_f1
```

### H3 — capability beyond token count

`h3_budget_loss_joined.csv`, `h3_capability_regressions.csv`:

```
L1  accuracy_loss ~ fertility + C(budget)
L2  accuracy_loss ~ fertility + boundary_f1 + C(budget)    ← the novel increment
L3  mean_shots ~ fertility                                  (the mechanism)
L4  accuracy ~ fertility + boundary_f1 + C(budget)
```

Fertility and boundary-F1 are those of **the evaluated model's own tokenizer**
(inferred from the model name; override with `--model-tokenizer-key`).

H3 is the weakest-powered analysis here: within one model, fertility and boundary-F1
vary only across languages, and H1 says they are collinear. Running a second model is
the only real fix, so Part C **evaluates two by default** (§2.4) — their tokenizers
disagree most on precisely the Indic languages under test. Part C results are also
**pooled across runs by default**, each row joined to its own model's tokenizer
metrics, with `C(model)` entering the regression whenever more than one model is
present (`--no-pool-part-c` to disable):

```bash
./run.sh part-c --tag caps           # both default models, one run id
./run.sh part-c --tag aya --models CohereLabs/aya-expanse-8b   # a third, with a token
./run.sh analyze --run-id latest     # pools every run, adds a model fixed effect
```

The observed `corr(fertility, boundary_f1)` and implied VIF are logged every run.

### H4 — what specialization costs English

`h4_english_cost.csv`, `h4_tradeoff.csv`, Figure 5.

### Figures

1. `fig1_boundary_f1_vs_fertility` — thesis figure: scatter coloured by morphological
   type, specialized tokenizers as diamonds, per-type slopes in the legend.
2. `fig2_premium_heatmap` — tokenizer × language premium, specialized rows separated.
3. `fig3_morpheme_overlay` — token boxes with gold morpheme boundaries overlaid.
   **Also written as `.html`**: matplotlib cannot shape Indic conjuncts, so open the
   HTML in a browser and screenshot it for the camera-ready.
4. `fig4_capability_vs_fertility` — budget loss vs fertility plus an added-variable
   plot for boundary-F1.
5. `fig5_english_cost` — English vs mean-Indic fertility per tokenizer.
6. `fig6_prototype_allocator` — Part D: boundary alignment and its fertility cost for
   the three allocator arms at matched vocabulary budget.

---

## 4. Data and models

| Resource | Source | Needs token | Local path |
|---|---|---|---|
| FLORES-200 (dev + devtest) | `dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` | no | `data/raw/flores200/` → `data/processed/flores/` |
| FLORES+ (fallback) | HF `openlanguagedata/flores_plus` | yes | HF cache |
| Morfessor analyzers (unsupervised) | `indic_nlp_resources` (hi, mr, bn, gu, pa, ta, te, kn, ml) | no | `data/resources/morfessor/` |
| UniMorph tables (annotated) | `github.com/unimorph/<iso>` (eng, deu, hin, ben, guj, kan, tel, urd, asm, hun, tur, ces, rus, spa …) | no | `data/resources/unimorph/` |
| SIGMORPHON-2022 segmentations (annotated) | `sigmorphon/2022SegmentationST` (eng, hun, ces, rus, spa, mon) | no | `data/resources/sigmorphon/` |
| Belebele (all 15 languages) | HF `facebook/belebele` | no | `data/processed/tasks/belebele/` |
| MGSM (en, de, sw, bn, te) | HF `juletxara/mgsm` | no | `data/processed/tasks/mgsm/` |
| Noto fonts | `notofonts.github.io` | no | `data/resources/fonts/` |

### Tokenizer zoo (`--tokenizers`)

| key | group | resolved source (no token needed) | vocab |
|---|---|---|---|
| `o200k` | mainstream | tiktoken `o200k_base` (GPT-4o) | 200 019 |
| `llama31` | mainstream | `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror | 128 000 |
| `gemma3` | mainstream | `unsloth/gemma-3-12b-it` mirror | 262 144 |
| `qwen3` | mainstream | `Qwen/Qwen3-8B` | 151 643 |
| `mistral_nemo` | mainstream | `mistralai/Mistral-Nemo-Instruct-2407` | 131 072 |
| `deepseek_v3` | mainstream | `deepseek-ai/DeepSeek-V3` | 128 000 |
| `sarvam_m` | mainstream¹ | `sarvamai/sarvam-m` | 131 072 |
| `aya_expanse` | mainstream | `CohereLabs/aya-expanse-8b` (**needs `HF_TOKEN`**) | 255 000 |
| `sarvam1` | specialized | `sarvamai/sarvam-1` | 68 096 |
| `muril` | specialized | `google/muril-base-cased` | 197 285 |
| `indicbert_v2` | specialized | `ai4bharat/IndicBERTv2-MLM-only` | 250 000 |
| `byt5` | byte | `google/byt5-small` | 256 |
| `utf8` | byte | analytic UTF-8 reference (no download) | 256 |

¹ Sarvam-M is an Indic-adapted *model* shipping the **unchanged Mistral vocabulary**
(verified: identical encodings on every FLORES sentence; only 6 special tokens
renamed). It is audited as mainstream and is the cleanest evidence in the paper that
adapting a model is not adapting its vocabulary. `part-a` detects such pairs
automatically and writes `tokenizer_duplicates.csv`.

### Gold morpheme segmentations (`data/gold/`, the released artifact)

`<lang>.gold.jsonl` — one JSON object per word type: `word`, `freq`, `morphs`,
`boundaries` (character offsets), `tier`, `sources`. Confidence tiers:

| tier | meaning |
|---|---|
| A | Morfessor **and** UniMorph agree on ≥ 1 boundary |
| B | UniMorph-derived stem/affix boundary (longest common prefix of lemma and form) |
| C | Morfessor multi-morph segmentation (all morphs ≥ 2 chars) |
| D | Morfessor says monomorphemic — 0 gold boundaries, capped at 25 % of the set; these carry the boundary-*precision* signal |

Word types come from FLORES dev+devtest (frequency-ranked, script-filtered, ≥ 4 chars).
`data/gold/coverage.csv` reports analyzer coverage per language — quote it in the paper.

Typical coverage at 2 000 types/language: hin .86, tam .83, kan .82, guj .78, tel .77,
mar .73, ben .72, pan .53, mal .53.

## 5. Reproducing the paper

```bash
# 0. environment + all data (~10 min, ~700 MB)
./run.sh setup
./run.sh fetch --langs all --gold-types 2000

# 1. Parts A and B — one run id, CPU only (~1 h)
./run.sh part-a --run-id paper-v1 --tokenizers all --langs all
./run.sh part-b --run-id paper-v1 --tokenizers all --langs morph,eng,deu

# 2. Part C — check budgets first, then the GPU run (both models, 6-16 h)
./run.sh part-c --run-id paper-v1 --dry-run
nohup ./run.sh part-c --run-id paper-v1 --n-items 250 > logs/partc.out 2>&1 &

# 3. tables + figures
./run.sh analyze --run-id paper-v1
./run.sh figures --run-id paper-v1

# optional robustness runs, cleanly separated
./run.sh part-b --tag tol0 --tolerance 0
./run.sh part-c --tag qwen-only --model Qwen/Qwen3-8B
```
