# llm-scfg

`llm-scfg` generates paired synthetic languages with synchronous context-free grammars, samples aligned sentence pairs from them, and builds prompt/batch inputs for LLM translation experiments. The intended use case is controlled language-learning evaluation: give a model a compact grammar "textbook", then test whether it can translate from language A to language B.

## Overview

The codebase supports:

- generating paired grammars over synthetic languages
- sampling parallel sentences at controlled recursion depth
- varying structural and morphological properties of the language pair
- rendering grammars into prompt-friendly displays
- building JSONL batch requests for model evaluation
- replaying those JSONL batch requests against local OpenAI-compatible servers such as vLLM

Current experimental axes:

- `word order`: head direction and specifier order
- `orthography`: writing system and phonotactics
- `agreement`: person/number agreement, plus latent gender with optional overt realization
- `size`: lexical inventory size
- `complexity`: grammar size and embedding depth

## Repository layout

- [main.py](/Users/jacksonpetty/Development/llm-scfg/main.py): main CLI for grammar generation, sample generation, and batch creation
- [open_weights.py](/Users/jacksonpetty/Development/llm-scfg/open_weights.py): run existing batch JSONL files against local OpenAI-compatible servers and emit analysis-compatible output JSONL
- [scfg/scfg.py](/Users/jacksonpetty/Development/llm-scfg/scfg/scfg.py): grammar parameterization, lexicon/paradigm generation, rule construction, sampling, and display formatting
- [scfg/agreement.py](/Users/jacksonpetty/Development/llm-scfg/scfg/agreement.py): feature bundles, inventories, and unification helpers
- [scfg/prompt.py](/Users/jacksonpetty/Development/llm-scfg/scfg/prompt.py): prompt templates
- [scripts/preview_grammar.py](/Users/jacksonpetty/Development/llm-scfg/scripts/preview_grammar.py): quick grammar/sample preview tool
- [scripts/run_vllm_eval.sh](/Users/jacksonpetty/Development/llm-scfg/scripts/run_vllm_eval.sh): launch a local vLLM server and run one input file or directory of batch files through it
- [scripts/slurm_vllm_eval.sbatch](/Users/jacksonpetty/Development/llm-scfg/scripts/slurm_vllm_eval.sbatch): Slurm template for cluster execution
- [tests](/Users/jacksonpetty/Development/llm-scfg/tests): `unittest` suite
- [data](/Users/jacksonpetty/Development/llm-scfg/data): saved grammars and samples
- [batches](/Users/jacksonpetty/Development/llm-scfg/batches): generated batch JSONL files
- [docs/agreement-roadmap.md](/Users/jacksonpetty/Development/llm-scfg/docs/agreement-roadmap.md): agreement implementation notes

## Setup

The repo expects Python 3.12+ and uses `uv`.

Install dependencies:

```bash
uv sync
```

Install the cluster runner environment, including `wandb` and Linux-only `vllm`:

```bash
uv sync --group cluster
```

The `vllm` dependency is declared with a Linux platform marker in
[pyproject.toml](/Users/jacksonpetty/Development/llm-scfg/pyproject.toml), so
this group resolves cleanly on macOS while still producing a complete GPU
environment on Linux cluster nodes.

All CLI entrypoints load `.env` from the repo root if present, so cluster-only
credentials such as `WANDB_API_KEY` can live there.

Example `.env`:

```bash
WANDB_API_KEY=...
WANDB_PROJECT=llm-scfg-vllm
WANDB_ENTITY=...
HF_TOKEN=...
HUGGINGFACE_HUB_TOKEN=...
```

The cluster launch script [scripts/run_vllm_eval.sh](/Users/jacksonpetty/Development/llm-scfg/scripts/run_vllm_eval.sh) also loads `.env` before starting `vllm serve`, so gated Hugging Face models such as Gemma 3 can authenticate via either `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`.

The open-weight runner in [open_weights.py](/Users/jacksonpetty/Development/llm-scfg/open_weights.py) will also log batch progress to W&B when `WANDB_API_KEY` is present. By default it logs to project `llm-scfg-vllm`; override with `WANDB_PROJECT`, and optionally set `WANDB_ENTITY`, `WANDB_RUN_GROUP`, or `WANDB_RUN_NAME`.

To verify Hugging Face authentication before launching a job:

```bash
uv run python scripts/check_hf_auth.py google/gemma-3-12b-it
```

That script loads `.env` from the repo root and checks authenticated access to
the target repo through the Hugging Face API.

Run the test suite:

```bash
uv run python -m unittest discover -s tests -v
```

## Quick start

Create one grammar:

```bash
uv run python main.py create_grammar --rng_seed=42
```

This writes `data/grammar_<id>.json`.

Generate samples for that grammar:

```bash
uv run python main.py gen_samples \
  --grammar_name=<grammar_id> \
  --min_depth=0 \
  --max_depth=3 \
  --n_samples_per_depth=5
```

This writes `data/samples_<grammar_id>.jsonl`.

Generate a batchfile for one grammar:

```bash
uv run python main.py gen_batchfile \
  --grammar_name=<grammar_id> \
  --model=gpt-5-nano
```

This writes a JSONL input file under [batches](/Users/jacksonpetty/Development/llm-scfg/batches).

Run that same JSONL through a local OpenAI-compatible server:

```bash
uv run python open_weights.py run_batch_file \
  --input_file=batches/inputs_<grammar_id>_gpt-5-nano.jsonl \
  --base_url=http://127.0.0.1:8000/v1 \
  --model_override=google/gemma-3-12b-it \
  --api_key=EMPTY
```

This writes a sibling `*_output.jsonl` file in the same OpenAI-like shape that the notebook analysis already consumes.

## Previewing a grammar

Use the preview script when you want to inspect the grammar display and a few sample pairs without running a whole experiment.

Preview a generated in-memory grammar:

```bash
uv run python scripts/preview_grammar.py --n_samples=3 --max_depth=1
```

Preview a saved grammar:

```bash
uv run python scripts/preview_grammar.py --grammar=data/agreement_exp/grammar_<id>.json
```

Preview an asymmetric gender system:

```bash
uv run python scripts/preview_grammar.py \
  --latent_gender=True \
  --realize_gender_a=False \
  --realize_gender_b=True \
  --n_samples=3 \
  --max_depth=0
```

The preview output includes:

- the compact grammar display used for prompting
- agreement metadata when agreement is enabled
- sampled `left` / `right` sentence pairs
- sampled agreement features for the subject and verb

## Main CLI commands

The CLI is exposed through `python main.py <command> ...` using `fire`.

Core commands:

- `create_grammar`: generate and save one grammar
- `gen_samples`: sample sentences from one saved grammar
- `gen_batchfile`: create one batch JSONL file for one grammar
- `gen_exp_batchfile`: create one or more batch JSONL files for an experiment directory
- `demo`, `demo_random`: print example grammars

Experiment commands:

- `exp_wordorder`
- `exp_wordorder_large`
- `exp_orthography`
- `exp_orthography_large`
- `exp_agreement`
- `exp_size`
- `exp_complexity`
- `exp_large_complexity`

Open-weight execution:

- `uv run python open_weights.py run_batch_file ...`
- `uv run python open_weights.py run_batch_dir ...`

These commands accept the same `inputs_*.jsonl` files produced for OpenAI-style batch APIs. Use `model_override` when replaying a GPT-targeted input file against a local model such as Gemma 3.

## Open-weight pipeline

The open-weight path is deliberately contract-compatible with the existing analysis flow:

- input: the existing `inputs_*.jsonl` chat-completions batch files
- execution: a local OpenAI-compatible endpoint, typically `vllm serve`
- output: `*_output.jsonl` files with `custom_id` and `response.body` fields shaped like provider batch results

That means the current analysis in [notebooks/error_analysis.py](/Users/jacksonpetty/Development/llm-scfg/notebooks/error_analysis.py) can ingest the outputs without a separate conversion step.

### Local vLLM run

Generate a standard experiment batch file first:

```bash
uv run python main.py gen_exp_batchfile --exp=wordorder_large --model=google/gemma-3-12b-it
```

Replay the resulting files against Gemma 3 via vLLM:

```bash
MODEL_NAME=google/gemma-3-12b-it bash scripts/run_vllm_eval.sh batches/wordorder_large_exp
```

The script starts `vllm serve`, waits for the server to answer on `/v1/models`, and then runs the batch files whose filename model suffix matches `MODEL_NAME` through [open_weights.py](/Users/jacksonpetty/Development/llm-scfg/open_weights.py). By default, a directory run against `google/gemma-3-12b-it` looks for files matching `inputs_*_google_gemma-3-12b-it_*.jsonl`.

If you need to override the filename matching for legacy batch files, set:

```bash
INPUT_GLOB='inputs_*_gemma-3-12b-it_*.jsonl' bash scripts/run_vllm_eval.sh batches/wordorder_large_exp
```

### Scratch venv bootstrap on NYU HPC

On the NYU cluster, the supported path is to build the project environment on
`/scratch` and run jobs from that venv instead of depending on a shared Python
install.

Do this on a compute node, not the login node. A GPU node is the safest default
because the bootstrap path uses `apptainer exec --nv` and installs the same
stack you will use for `vllm` inference. If you already know your cluster image
and CUDA stack work on a CPU-only compute node, that can also work, but the
recommended path is an interactive GPU allocation.

Example interactive allocation:

```bash
srun --account=torch_pr_287_general --partition=<gpu-partition> --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
```

Then move to scratch and clone the repo there if you want both the checkout and
the venv on scratch:

```bash
cd /scratch/$USER
git clone <repo-url> llm-scfg
cd llm-scfg
```

You do not need `uv` preinstalled on the cluster. The bootstrap script installs
it for you under `/scratch/$USER/uv`.

Run the bootstrap script from the repo root:

```bash
bash scripts/bootstrap_scratch_cluster_venv.sh
```

That script will:

- install `uv` under `/scratch/$USER/uv`
- install Python `3.12` under `/scratch/$USER/uv-python`
- create `/scratch/$USER/venvs/llm-scfg`
- install this repo with the `cluster` dependency group
- verify that `openai`, `wandb`, and `vllm` are importable/executable

The bootstrap script pins Python `3.12` on purpose. This matches the Python
version used in your other cluster setup and is the safer target for the GPU
runtime stack here. For this `vllm` path, do not separately install upstream
`flash-attn`; `vllm` ships and selects its own compatible attention backend
stack.

After bootstrap, the default runtime paths are:

```text
/scratch/$USER/venvs/llm-scfg/bin/python
/scratch/$USER/venvs/llm-scfg/bin/vllm
```

The Slurm wrapper in [scripts/slurm_vllm_eval.sbatch](/Users/jacksonpetty/Development/llm-scfg/scripts/slurm_vllm_eval.sbatch) and the launch script in [scripts/run_vllm_eval.sh](/Users/jacksonpetty/Development/llm-scfg/scripts/run_vllm_eval.sh) both default to that scratch venv, so no extra activation step is needed once bootstrap has finished.

### Cluster notes

- `scripts/slurm_vllm_eval.sbatch` is a minimal Slurm template for this workflow.
- The Slurm script includes `#SBATCH --account=torch_pr_287_general`, matching the account used in your other NYU cluster scripts.
- `google/gemma-3-12b-it` is the safest initial target for L40S nodes; larger Gemma 3 variants will usually want more GPU memory or tensor parallelism across multiple GPUs.
- To build the cluster environment from this repo, run `uv sync --group cluster` on the Linux node or in the cluster image build.
- The bootstrap script defaults to `/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif`; override `IMAGE=...` if your partition uses a different Apptainer image.
- `vllm` auto-selects an attention backend by default. If you want to force FlashAttention for a run, pass `--export=ATTENTION_BACKEND=FLASH_ATTN` at submit time.
- Submit a job after bootstrap with:

```bash
scripts/submit_vllm_eval.sh batches/wordorder_large_exp --export=MODEL_NAME=google/gemma-3-12b-it
```

If you want to bypass the wrapper, the direct form is:

```bash
sbatch --export=ALL,INPUT_PATH=batches/wordorder_large_exp,MODEL_NAME=google/gemma-3-12b-it scripts/slurm_vllm_eval.sbatch
```

## Experimental setups

### Word order

Run:

```bash
uv run python main.py exp_wordorder
uv run python main.py exp_wordorder_large
```

This experiment varies:

- head directionality
- specifier order

It keeps orthography and agreement mostly fixed. Outputs live in [data/wordorder_exp](/Users/jacksonpetty/Development/llm-scfg/data/wordorder_exp) and [data/wordorder_large_exp](/Users/jacksonpetty/Development/llm-scfg/data/wordorder_large_exp).

### Orthography

Run:

```bash
uv run python main.py exp_orthography
uv run python main.py exp_orthography_large
```

This experiment varies the target-side orthography and related surface-form properties while holding syntax mostly fixed. The large variant uses `hebrew` and `hebrew_unpointed` as separate target-script conditions. Outputs live in [data/orthography_exp](/Users/jacksonpetty/Development/llm-scfg/data/orthography_exp) and [data/orthography_large_exp](/Users/jacksonpetty/Development/llm-scfg/data/orthography_large_exp).

### Agreement

Run:

```bash
uv run python main.py exp_agreement --grammar_sizes='[100,1000]'
```

Current agreement support includes:

- subject-verb agreement in person and number
- latent gender on nouns and proper nouns
- optional overt gender realization on verbs per language side
- asymmetric systems where one side overtly marks a feature and the other does not

Useful `create_grammar` flags:

- `agreement_enabled_a`, `agreement_enabled_b`
- `latent_gender_a`, `latent_gender_b`
- `realize_gender_a`, `realize_gender_b`

Example: both languages share latent gender, but only side `b` marks it overtly:

```bash
uv run python main.py create_grammar \
  --agreement_enabled_a=True \
  --agreement_enabled_b=True \
  --latent_gender_a=True \
  --latent_gender_b=True \
  --realize_gender_a=False \
  --realize_gender_b=True
```

Agreement experiment outputs live in [data/agreement_exp](/Users/jacksonpetty/Development/llm-scfg/data/agreement_exp).

### Size

Run:

```bash
uv run python main.py exp_size
```

This varies lexical inventory size while holding the rest of the grammar close to fixed. Outputs live in [data/size_exp](/Users/jacksonpetty/Development/llm-scfg/data/size_exp).

### Complexity

Run:

```bash
uv run python main.py exp_complexity
uv run python main.py exp_large_complexity
```

These experiments vary grammar size and recursion depth to test how grammatical and sentential complexity affect translation performance.

## Grammar display format

For agreement-enabled grammars, the prompt-facing display is intentionally more compact than the full internal lexicon/paradigm metadata.

Example:

```text
V1[lemma] -> <'stem_a', 'stem_b'>
V1[3.sg.fem] -> <'surface_a', 'surface_b'>
N2[lemma.masc] -> <'noun_a', 'noun_b'>
N2[pl.masc] -> <'plural_a', 'plural_b'>
PROPN1[3.sg.fem] -> <'name_a', 'name_b'>
PRON[2.sg] -> <'form_a', 'form_b'>
```

Interpretation:

- `V1`, `N2`, `PROPN1`, and similar tags are display-local IDs
- `lemma` lines show the reference lemma/stem pair
- bracketed feature tags indicate which surface pair realizes which feature bundle
- asymmetric systems can share latent features even when only one side marks them overtly

This compact display is meant for prompts and human inspection. The saved grammar JSON still contains richer metadata when needed for validation or analysis.

## Typical workflow

Generate an experiment:

```bash
uv run python main.py exp_agreement --grammar_sizes='[100]'
```

Build experiment batchfiles:

```bash
uv run python main.py gen_exp_batchfile --exp=agreement --model=gpt-5-nano
```

Then:

1. run the resulting batch JSONL against your model provider
2. collect model outputs
3. analyze results in notebooks or downstream scripts

The prompt construction logic lives in [scfg/prompt.py](/Users/jacksonpetty/Development/llm-scfg/scfg/prompt.py).

## Notes

- The displayed grammar is a compact presentation layer, not a full dump of all internal generation state.
- Generated languages are synthetic; they are designed to be controlled and verifiable rather than naturalistic.
- Agreement-enabled grammars expose both a compact display and richer `agreement_metadata` in the saved grammar JSON.
- Some tooling and notebooks in the repo still reflect older experiment setups; use the CLI and tests as the source of truth when in doubt.
