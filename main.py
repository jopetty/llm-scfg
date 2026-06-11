# Generating samples from scfgs

import json
import logging
import os
import pathlib
import random
import secrets
from functools import lru_cache
from hashlib import blake2b
from typing import Any, Dict, Union, cast

import fire
import numpy as np
import pandas as pd
import pyrootutils
import tiktoken
from dotenv import load_dotenv

from scfg.prompt import ChatCompletionResponse, basic_prompt
from scfg.scfg import SCFG, CFGParams, RuleBuilder, SCFGParams
from scfg.utils import get_logger, set_all_seeds

Path = pathlib.Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)
DATA_DIR = PROJECT_ROOT / "data"
BATCH_DIR = PROJECT_ROOT / "batches"
DEFAULT_HF_DATASET_REPO = "jowenpetty/scfg"

load_dotenv(PROJECT_ROOT / ".env")

GPT_MESSAGE_TOKEN_LIMIT = 272_000
DEFAULT_LARGE_GRAMMAR_SIZES = [25, 50, 100, 1_000, 5_000, 7_500, 10_000]
FEWSHOT_K_VALUES = [0, 1, 2, 4, 8, 16]

ORTHOGRAPHY_LABELS = {
    "latin": "Latin",
    "latin_diacritic": "Latin with diacritics",
    "cyrillic": "Cyrillic",
    "hebrew": "Hebrew",
    "yiddish": "Hebrew",
    "hebrew_unpointed": "Hebrew without nikkud",
}


def deterministic_seed(
    *parts: object, base_seed: int = 42, modulus: int = 10_000
) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return base_seed + (int.from_bytes(digest, "big") % modulus)


def experiment_dir(exp_name: str) -> Path:
    return DATA_DIR / f"{exp_name}_exp"


def _write_experiment_grammar_index(
    exp_dir: Path,
    exp_name: str,
    grammar_names: list[str],
) -> None:
    with open(exp_dir / f"{exp_name}_grammars.txt", "w") as handle:
        for name in grammar_names:
            handle.write(f"{name}\n")


def _load_first_sample(exp_dir: Path, grammar_name: str) -> dict[str, Any]:
    with open(exp_dir / f"samples_{grammar_name}.jsonl", "r") as handle:
        return json.loads(handle.readline())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def _resolve_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _resolve_hf_dataset_repo(hf_repo_id: str | None = None) -> str:
    return hf_repo_id or os.getenv("LLM_SCFG_HF_REPO_ID") or DEFAULT_HF_DATASET_REPO


def _experiment_config_name(exp: str) -> str:
    return exp if exp.endswith("_exp") else f"{exp}_exp"


def _strip_raw_json(row: dict[str, Any]) -> dict[str, Any]:
    raw_json = row.get("raw_json")
    if isinstance(raw_json, str) and raw_json:
        return cast(dict[str, Any], json.loads(raw_json))
    return row


@lru_cache(maxsize=32)
def _load_hf_split(
    repo_id: str,
    config_name: str,
    split: str,
) -> tuple[dict[str, Any], ...]:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Loading experiment data from Hugging Face requires pyarrow and "
            "huggingface-hub. "
            "Run `uv sync` to install the project dependencies."
        ) from exc

    token = _resolve_hf_token()
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{config_name}/{split}.parquet",
        token=token,
    )
    table = pq.read_table(path)
    rows = [_strip_raw_json(cast(dict[str, Any], row)) for row in table.to_pylist()]
    rows.sort(
        key=lambda row: (
            str(row.get("grammar_name") or row.get("name")),
            int(row.get("row_index") or 0),
        )
    )
    return tuple(rows)


def _load_hf_experiment_data(
    *,
    exp: str,
    hf_repo_id: str | None,
) -> dict[str, dict[str, Any]]:
    repo_id = _resolve_hf_dataset_repo(hf_repo_id)
    config_name = _experiment_config_name(exp)
    log.info("Loading %s from Hugging Face dataset %s", config_name, repo_id)

    grammar_rows = _load_hf_split(repo_id, config_name, "grammars")
    sample_rows = _load_hf_split(repo_id, config_name, "samples")
    shot_rows = _load_hf_split(repo_id, config_name, "shots")

    grammars: dict[str, dict[str, Any]] = {}
    samples_by_grammar: dict[str, list[dict[str, Any]]] = {}
    shots_by_grammar: dict[str, list[dict[str, Any]]] = {}

    for grammar in grammar_rows:
        name = str(grammar.get("name") or "")
        if name:
            grammars[name] = dict(grammar)
    for sample in sample_rows:
        grammar_name = str(sample.get("grammar_name") or "")
        if grammar_name:
            samples_by_grammar.setdefault(grammar_name, []).append(dict(sample))
    for shot in shot_rows:
        grammar_name = str(shot.get("grammar_name") or "")
        if grammar_name:
            shots_by_grammar.setdefault(grammar_name, []).append(dict(shot))

    grammar_names = [name for name in grammars if name in samples_by_grammar]
    if not grammar_names:
        raise ValueError(
            f"No HF samples found for experiment {config_name} in {repo_id}"
        )

    return {
        name: {
            "grammar": grammars[name],
            "samples": samples_by_grammar[name],
            "shots": shots_by_grammar.get(name),
        }
        for name in grammar_names
    }


def _load_local_experiment_data(exp: str) -> dict[str, dict[str, Any]]:
    exp_dir = DATA_DIR / _experiment_config_name(exp)
    grammar_list_filepath = exp_dir / f"{exp}_grammars.txt"

    grammar_names: list[str] = []
    with open(grammar_list_filepath, "r") as handle:
        for line in handle:
            if grammar_name := line.strip():
                grammar_names.append(grammar_name)

    experiment_data: dict[str, dict[str, Any]] = {}
    for grammar_name in grammar_names:
        samples = _load_jsonl(exp_dir / f"samples_{grammar_name}.jsonl")
        shots_path = exp_dir / f"shots_{grammar_name}.jsonl"
        shots = _load_jsonl(shots_path) if shots_path.exists() else None
        with open(exp_dir / f"grammar_{grammar_name}.json", "r") as handle:
            grammar = json.load(handle)
        experiment_data[grammar_name] = {
            "grammar": grammar,
            "samples": samples,
            "shots": shots,
        }
    return experiment_data


def _load_experiment_data(
    *,
    exp: str,
    data_source: str,
    hf_repo_id: str | None,
) -> dict[str, dict[str, Any]]:
    normalized_source = data_source.lower()
    exp_dir = DATA_DIR / _experiment_config_name(exp)
    local_index = exp_dir / f"{exp}_grammars.txt"
    if normalized_source == "auto":
        normalized_source = "local" if local_index.exists() else "hf"
    if normalized_source == "local":
        return _load_local_experiment_data(exp)
    if normalized_source == "hf":
        return _load_hf_experiment_data(exp=exp, hf_repo_id=hf_repo_id)
    raise ValueError("data_source must be one of: auto, local, hf")


def _load_single_grammar_data(
    *,
    grammar_name: str,
    exp: str | None,
    data_source: str,
    hf_repo_id: str | None,
) -> dict[str, Any]:
    normalized_source = data_source.lower()
    grammar_path = DATA_DIR / f"grammar_{grammar_name}.json"
    samples_path = DATA_DIR / f"samples_{grammar_name}.jsonl"
    if normalized_source == "auto":
        normalized_source = (
            "local" if grammar_path.exists() and samples_path.exists() else "hf"
        )

    if normalized_source == "local":
        samples = _load_jsonl(samples_path)
        shots_path = DATA_DIR / f"shots_{grammar_name}.jsonl"
        shots = _load_jsonl(shots_path) if shots_path.exists() else None
        with open(grammar_path, "r") as handle:
            grammar = json.load(handle)
        return {"grammar": grammar, "samples": samples, "shots": shots}

    if normalized_source == "hf":
        if exp is None:
            raise ValueError("exp is required when gen_batchfile loads from HF")
        experiment_data = _load_hf_experiment_data(exp=exp, hf_repo_id=hf_repo_id)
        if grammar_name not in experiment_data:
            raise ValueError(
                f"Grammar {grammar_name} not found in HF experiment "
                f"{_experiment_config_name(exp)}"
            )
        return experiment_data[grammar_name]

    raise ValueError("data_source must be one of: auto, local, hf")


def _orthography_label(orthography: str) -> str:
    return ORTHOGRAPHY_LABELS.get(orthography, orthography.replace("_", " ").title())


def _wordorder_label(head_initial: bool, spec_initial: bool) -> str:
    if head_initial and spec_initial:
        return "Target matches the source order"
    if not head_initial and spec_initial:
        return "Target is head-final, spec-initial"
    if not head_initial and not spec_initial:
        return "Target is head-final, spec-final"
    if head_initial and not spec_initial:
        return "Target is head-initial, spec-final"
    return f"target head_initial={head_initial}, spec_initial={spec_initial}"


def _write_experiment_readme(
    exp_name: str,
    title: str,
    overview: str,
    matrix_items: list[tuple[str, str]],
    condition_examples: list[dict[str, str]],
    regeneration_command: str,
) -> None:
    exp_dir = experiment_dir(exp_name)
    lines = [
        f"# {title}",
        "",
        overview,
        "",
        "## Experimental Matrix",
        "",
    ]
    for key, value in matrix_items:
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "```bash",
            regeneration_command,
            "```",
            "",
            "## Condition Examples",
            "",
            (
                "Examples use the `left_phonetic` and `right_phonetic` "
                "surface forms from the sample JSONL files."
            ),
            "",
        ]
    )

    for example in condition_examples:
        lines.extend(
            [
                f"### {example['title']}",
                "",
                f"- condition: {example['condition']}",
                f"- grammar: `{example['grammar_name']}`",
                f"- example left: `{example['left']}`",
                f"- example right: `{example['right']}`",
                "",
            ]
        )

    with open(exp_dir / "README.md", "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def _create_orthography_dataset(
    exp_name: str,
    title: str,
    overview: str,
    grammar_sizes: list[int],
    target_orthographies: list[str],
    max_depth: int,
    n_grammars_per_size: int,
    n_sentences_per_depth: int,
) -> None:
    syllable_structure = "C*VC"
    exp_dir = experiment_dir(exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    grammar_names: list[str] = []
    example_grammars: dict[str, tuple[str, int]] = {}

    for orthography in target_orthographies:
        for g_size in grammar_sizes:
            for index in range(n_grammars_per_size):
                g_seed = deterministic_seed(exp_name, orthography, g_size, index)
                grammar_name = create_grammar(
                    rng_seed=g_seed,
                    syllable_structure_a=syllable_structure,
                    syllable_structure_b=syllable_structure,
                    head_initial_a=True,
                    head_initial_b=True,
                    spec_initial_a=True,
                    spec_initial_b=True,
                    pro_drop_a=False,
                    pro_drop_b=False,
                    n_verbs=max(2, g_size // 5),
                    n_nouns=max(2, g_size // 5),
                    n_adjectives=max(2, g_size // 5),
                    n_propns=max(2, g_size // 5),
                    n_det_def=2,
                    n_det_indef=2,
                    n_prons=2,
                    n_comps=2,
                    orthography_b=orthography,
                    exp_name=exp_name,
                )
                log.info(
                    "Created grammar %s with orthography=%s, seed=%s",
                    grammar_name,
                    orthography,
                    g_seed,
                )
                generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=g_seed,
                    min_depth=0,
                    max_depth=max_depth,
                    n_samples_per_depth=n_sentences_per_depth,
                    exp_name=exp_name,
                )
                grammar_names.append(grammar_name)
                example_grammars.setdefault(orthography, (grammar_name, g_size))

    _write_experiment_grammar_index(exp_dir, exp_name, grammar_names)

    condition_examples = []
    for orthography in target_orthographies:
        grammar_name, g_size = example_grammars[orthography]
        sample = _load_first_sample(exp_dir, grammar_name)
        condition_examples.append(
            {
                "title": _orthography_label(orthography),
                "condition": (
                    f"target orthography=`{orthography}`, lexical size target={g_size}"
                ),
                "grammar_name": grammar_name,
                "left": sample["left_phonetic"],
                "right": sample["right_phonetic"],
            }
        )

    total_grammars = len(grammar_names)
    total_samples = total_grammars * (max_depth + 1) * n_sentences_per_depth
    _write_experiment_readme(
        exp_name=exp_name,
        title=title,
        overview=overview,
        matrix_items=[
            ("source orthography", "`latin`"),
            (
                "target orthographies",
                ", ".join(f"`{orthography}`" for orthography in target_orthographies),
            ),
            ("grammar sizes", str(grammar_sizes)),
            ("grammars per size and orthography", str(n_grammars_per_size)),
            ("depth range", f"`0..{max_depth}`"),
            ("samples per depth", str(n_sentences_per_depth)),
            ("total grammars", str(total_grammars)),
            ("total samples", str(total_samples)),
        ],
        condition_examples=condition_examples,
        regeneration_command=f"uv run python main.py exp_{exp_name}",
    )


def _create_wordorder_dataset(
    exp_name: str,
    title: str,
    overview: str,
    grammar_sizes: list[int],
    target_head_spec_params: list[tuple[bool, bool]],
    max_depth: int,
    n_grammars_per_size: int,
    n_sentences_per_depth: int,
) -> None:
    syllable_structure = "C*VC"
    source_head_spec_params: tuple[bool, bool] = (True, True)
    exp_dir = experiment_dir(exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    grammar_names: list[str] = []
    example_grammars: dict[tuple[bool, bool], tuple[str, int]] = {}

    for hi_b, si_b in target_head_spec_params:
        for g_size in grammar_sizes:
            for index in range(n_grammars_per_size):
                g_seed = deterministic_seed(exp_name, hi_b, si_b, g_size, index)
                grammar_name = create_grammar(
                    rng_seed=g_seed,
                    syllable_structure_a=syllable_structure,
                    syllable_structure_b=syllable_structure,
                    head_initial_a=source_head_spec_params[0],
                    head_initial_b=hi_b,
                    spec_initial_a=source_head_spec_params[1],
                    spec_initial_b=si_b,
                    pro_drop_a=False,
                    pro_drop_b=False,
                    n_verbs=max(2, g_size // 5),
                    n_nouns=max(2, g_size // 5),
                    n_adjectives=max(2, g_size // 5),
                    n_propns=max(2, g_size // 5),
                    n_det_def=2,
                    n_det_indef=2,
                    n_prons=2,
                    n_comps=2,
                    exp_name=exp_name,
                )
                log.info(
                    "Created grammar %s with hi_b=%s, si_b=%s, seed=%s",
                    grammar_name,
                    hi_b,
                    si_b,
                    g_seed,
                )
                generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=g_seed,
                    min_depth=0,
                    max_depth=max_depth,
                    n_samples_per_depth=n_sentences_per_depth,
                    exp_name=exp_name,
                )
                grammar_names.append(grammar_name)
                example_grammars.setdefault((hi_b, si_b), (grammar_name, g_size))

    _write_experiment_grammar_index(exp_dir, exp_name, grammar_names)

    condition_examples = []
    for hi_b, si_b in target_head_spec_params:
        grammar_name, g_size = example_grammars[(hi_b, si_b)]
        sample = _load_first_sample(exp_dir, grammar_name)
        condition_examples.append(
            {
                "title": _wordorder_label(hi_b, si_b),
                "condition": (
                    f"target head_initial={hi_b}, target spec_initial={si_b}, "
                    f"lexical size target={g_size}"
                ),
                "grammar_name": grammar_name,
                "left": sample["left_phonetic"],
                "right": sample["right_phonetic"],
            }
        )

    total_grammars = len(grammar_names)
    total_samples = total_grammars * (max_depth + 1) * n_sentences_per_depth
    _write_experiment_readme(
        exp_name=exp_name,
        title=title,
        overview=overview,
        matrix_items=[
            ("source word order", "`head_initial=True`, `spec_initial=True`"),
            (
                "target word orders",
                ", ".join(
                    f"`head_initial={hi_b}, spec_initial={si_b}`"
                    for hi_b, si_b in target_head_spec_params
                ),
            ),
            ("grammar sizes", str(grammar_sizes)),
            ("grammars per size and word-order condition", str(n_grammars_per_size)),
            ("depth range", f"`0..{max_depth}`"),
            ("samples per depth", str(n_sentences_per_depth)),
            ("total grammars", str(total_grammars)),
            ("total samples", str(total_samples)),
        ],
        condition_examples=condition_examples,
        regeneration_command=f"uv run python main.py exp_{exp_name}",
    )


def prompt_grammar_str(
    grammar: dict[str, object],
    sample: str,
    prompt_type: str,
) -> str:
    if prompt_type == "basic":
        return str(grammar["grammar_str"])
    if prompt_type == "compact":
        params = SCFGParams.from_dict(grammar)
        return RuleBuilder(params).build_compact_prompt_grammar(sample)
    raise ValueError(f"Unknown prompt type: {prompt_type}")


def normalize_k_shots(k_shots: int | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(k_shots, int):
        values = [k_shots]
    else:
        values = list(k_shots)
    if any(k < 0 for k in values):
        raise ValueError(f"k_shots must be non-negative; got {values}")
    return list(dict.fromkeys(values))


def estimate_prompt_tokens(prompt: str, model: str) -> int:
    normalized = model.lower()
    if "gemma" in normalized:
        tokenizer = gemma_tokenizer(model)
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(cast(list[int], tokens))

    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(prompt))


def add_prompt_token_estimates(df: pd.DataFrame, model: str) -> pd.DataFrame:
    df = df.copy()
    if "prompt" in df and "prompt_tokens" not in df:
        df["prompt_tokens"] = df["prompt"].apply(
            lambda prompt: estimate_prompt_tokens(prompt, model)
        )
    return df


@lru_cache(maxsize=16)
def gemma_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Generating Gemma batch files requires transformers. "
            "Install the cluster dependency group or add transformers locally."
        ) from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    return AutoTokenizer.from_pretrained(model, token=token)


def model_input_token_limit(model: str) -> int | None:
    normalized = model.lower()
    if "gemma-3-270m" in normalized or "gemma-3-1b" in normalized:
        return 32_000
    if any(size in normalized for size in ["gemma-3-4b", "gemma-3-12b", "gemma-3-27b"]):
        return 128_000
    if "gemma-2" in normalized:
        return 8_192
    return None


def drop_rows_over_model_context(
    df: pd.DataFrame,
    *,
    model: str,
    grammar_name: str,
) -> pd.DataFrame:
    context_limit = model_input_token_limit(model)
    if context_limit is None or "prompt" not in df:
        return df

    df = add_prompt_token_estimates(df, model)
    token_estimates = pd.to_numeric(df["prompt_tokens"], errors="coerce")
    over_limit_mask = token_estimates > context_limit
    dropped = int(over_limit_mask.sum())
    if dropped:
        max_tokens = int(token_estimates.max())
        log.warning(
            (
                "Dropping %s prompts for %s because they exceed the %s-token "
                "input window for %s; max estimate=%s"
            ),
            dropped,
            grammar_name,
            context_limit,
            model,
            max_tokens,
        )
    return df.loc[~over_limit_mask].copy()


def warn_for_large_prompts(
    df: pd.DataFrame,
    model: str,
    grammar_name: str,
    threshold: int = GPT_MESSAGE_TOKEN_LIMIT,
) -> None:
    if not model.startswith("gpt") or "prompt" not in df:
        return
    df = add_prompt_token_estimates(df, model)
    token_estimates = pd.to_numeric(df["prompt_tokens"], errors="coerce")
    max_tokens = int(token_estimates.max())
    over_limit = int((token_estimates > threshold).sum())
    near_limit = int((token_estimates > int(0.8 * threshold)).sum())
    if near_limit:
        log.warning(
            (
                "Grammar %s has %s prompts above 80%% of the %s-token limit "
                "for %s; max estimate=%s"
            ),
            grammar_name,
            near_limit,
            threshold,
            model,
            max_tokens,
        )
    if over_limit:
        log.warning(
            (
                "Grammar %s has %s prompts estimated above the %s-token limit "
                "for %s; max estimate=%s"
            ),
            grammar_name,
            over_limit,
            threshold,
            model,
            max_tokens,
        )


def sample_metadata(
    *,
    grammar_name: str,
    sample_id: str,
    row: pd.Series,
    grammar: dict[str, object],
    k_shots: int,
) -> dict[str, str]:
    lexical_frequency = grammar.get("lexical_frequency_metadata")
    source_frequency: object = {}
    if isinstance(lexical_frequency, dict):
        frequency_metadata = cast(dict[str, object], lexical_frequency)
        source_frequency = frequency_metadata.get("a", {})
    if not isinstance(source_frequency, dict):
        source_frequency = {}
    source_frequency_metadata = cast(dict[str, object], source_frequency)
    return {
        "grammar_name": grammar_name,
        "sample_id": sample_id,
        "depth": str(row["depth"]),
        "input_sentence": str(row.get("left_phonetic") or row.get("left") or ""),
        "output_sentence": str(row.get("right_phonetic") or row.get("right") or ""),
        "n_words": str(grammar.get("n_words", "")),
        "n_rules": str(grammar.get("n_rules", "")),
        "prompt_tokens": str(row.get("prompt_tokens", "")),
        "k_shots": str(k_shots),
        "lexical_frequency_profile": str(source_frequency_metadata.get("profile", "")),
        "lexical_frequency_exponent": str(
            source_frequency_metadata.get("exponent", "")
        ),
        "lexical_frequency_length_unit": str(
            source_frequency_metadata.get("length_unit", "")
        ),
    }


def prompt_agreement_metadata(grammar: dict[str, object]) -> dict | None:
    agreement_metadata = grammar.get("agreement_metadata")
    if isinstance(agreement_metadata, dict):
        return agreement_metadata
    return None


def shot_examples_from_samples(
    shot_samples: list[dict[str, Any]],
    *,
    k_shots: int,
    exclude_sample_id: int | None = None,
) -> list[dict[str, str]]:
    if k_shots == 0:
        return []

    examples: list[dict[str, str]] = []
    for index, sample in enumerate(shot_samples):
        if exclude_sample_id is not None and index == exclude_sample_id:
            continue
        examples.append(
            {
                "input": str(sample.get("left_phonetic") or sample.get("left") or ""),
                "output": str(
                    sample.get("right_phonetic") or sample.get("right") or ""
                ),
            }
        )
        if len(examples) >= k_shots:
            break
    if len(examples) < k_shots:
        raise ValueError(
            f"Requested {k_shots} few-shot examples but only found {len(examples)}"
        )
    return examples


def load_shot_samples(
    *,
    grammar_dir: Path,
    grammar_name: str,
    fallback_samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    shots_path = grammar_dir / f"shots_{grammar_name}.jsonl"
    if shots_path.exists():
        return _load_jsonl(shots_path), True
    return fallback_samples, False


def create_orthography_data(
    grammar_sizes: list[int] = DEFAULT_LARGE_GRAMMAR_SIZES,
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
    target_orthographies: list[str] = [
        "latin",
        "latin_diacritic",
        "cyrillic",
        "hebrew",
        "hebrew_unpointed",
    ],
):
    """
    Generates the paper-facing orthography dataset.
    """

    _create_orthography_dataset(
        exp_name="orthography",
        title="Orthography Experiment Data",
        overview=(
            "This dataset expands the orthography experiment with a larger "
            "grammar-size "
            "grid and two additional target-side writing systems."
        ),
        grammar_sizes=grammar_sizes,
        target_orthographies=target_orthographies,
        max_depth=max_depth,
        n_grammars_per_size=n_grammars_per_size,
        n_sentences_per_depth=n_sentences_per_depth,
    )


def create_wordorder_data(
    grammar_sizes: list[int] = DEFAULT_LARGE_GRAMMAR_SIZES,
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
    target_head_spec_params: list[tuple[bool, bool]] = [
        (True, True),
        (False, True),
        (False, False),
    ],
):
    """
    Generates the paper-facing word-order dataset.
    """

    _create_wordorder_dataset(
        exp_name="wordorder",
        title="Word Order Experiment Data",
        overview=(
            "This dataset expands the word-order experiment with more grammar sizes "
            "and more grammar replicates per condition."
        ),
        grammar_sizes=grammar_sizes,
        target_head_spec_params=target_head_spec_params,
        max_depth=max_depth,
        n_grammars_per_size=n_grammars_per_size,
        n_sentences_per_depth=n_sentences_per_depth,
    )


def create_fewshot_data(
    grammar_sizes: list[int] = [25, 50, 100, 1000],
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
    n_shot_examples: int = max(FEWSHOT_K_VALUES),
    target_head_spec_params: list[tuple[bool, bool]] = [
        (True, True),
        (False, True),
        (False, False),
    ],
) -> None:
    """
    Generates Latin-script word-order grammars with held-out few-shot pools.
    """
    syllable_structure = "C*VC"
    source_head_spec_params: tuple[bool, bool] = (True, True)
    exp_name = "fewshot"
    exp_dir = experiment_dir(exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    grammar_names: list[str] = []
    example_grammars: dict[tuple[bool, bool], tuple[str, int]] = {}
    n_shots_per_depth = max(1, (n_shot_examples + max_depth) // (max_depth + 1))

    for hi_b, si_b in target_head_spec_params:
        for g_size in grammar_sizes:
            for index in range(n_grammars_per_size):
                g_seed = deterministic_seed(exp_name, hi_b, si_b, g_size, index)
                grammar_name = create_grammar(
                    rng_seed=g_seed,
                    syllable_structure_a=syllable_structure,
                    syllable_structure_b=syllable_structure,
                    head_initial_a=source_head_spec_params[0],
                    head_initial_b=hi_b,
                    spec_initial_a=source_head_spec_params[1],
                    spec_initial_b=si_b,
                    pro_drop_a=False,
                    pro_drop_b=False,
                    n_verbs=max(2, g_size // 5),
                    n_nouns=max(2, g_size // 5),
                    n_adjectives=max(2, g_size // 5),
                    n_propns=max(2, g_size // 5),
                    n_det_def=2,
                    n_det_indef=2,
                    n_prons=2,
                    n_comps=2,
                    exp_name=exp_name,
                )
                log.info(
                    "Created few-shot grammar %s with hi_b=%s, si_b=%s, seed=%s",
                    grammar_name,
                    hi_b,
                    si_b,
                    g_seed,
                )
                generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=g_seed,
                    min_depth=0,
                    max_depth=max_depth,
                    n_samples_per_depth=n_sentences_per_depth,
                    exp_name=exp_name,
                )
                generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=deterministic_seed("fewshot_shots", grammar_name),
                    min_depth=0,
                    max_depth=max_depth,
                    n_samples_per_depth=n_shots_per_depth,
                    exp_name=exp_name,
                    output_prefix="shots",
                )
                grammar_names.append(grammar_name)
                example_grammars.setdefault((hi_b, si_b), (grammar_name, g_size))

    _write_experiment_grammar_index(exp_dir, exp_name, grammar_names)

    condition_examples = []
    for hi_b, si_b in target_head_spec_params:
        grammar_name, g_size = example_grammars[(hi_b, si_b)]
        sample = _load_first_sample(exp_dir, grammar_name)
        condition_examples.append(
            {
                "title": _wordorder_label(hi_b, si_b),
                "condition": (
                    f"target head_initial={hi_b}, target spec_initial={si_b}, "
                    f"lexical size target={g_size}"
                ),
                "grammar_name": grammar_name,
                "left": sample["left_phonetic"],
                "right": sample["right_phonetic"],
            }
        )

    total_grammars = len(grammar_names)
    total_eval_samples = total_grammars * (max_depth + 1) * n_sentences_per_depth
    total_shot_samples = total_grammars * (max_depth + 1) * n_shots_per_depth
    _write_experiment_readme(
        exp_name=exp_name,
        title="Few-Shot Word Order Experiment Data",
        overview=(
            "This dataset varies the number of in-context examples for Latin-script "
            "source and target grammars while using the word-order conditions from "
            "the main word-order experiment. Each grammar has a dedicated "
            "`shots_*.jsonl` pool held out from the evaluated `samples_*.jsonl` rows."
        ),
        matrix_items=[
            ("source word order", "`head_initial=True`, `spec_initial=True`"),
            (
                "target word orders",
                ", ".join(
                    f"`head_initial={hi_b}, spec_initial={si_b}`"
                    for hi_b, si_b in target_head_spec_params
                ),
            ),
            ("source orthography", "`latin`"),
            ("target orthography", "`latin`"),
            ("few-shot k values", str(FEWSHOT_K_VALUES)),
            ("grammar sizes", str(grammar_sizes)),
            ("grammars per size and word-order condition", str(n_grammars_per_size)),
            ("depth range", f"`0..{max_depth}`"),
            ("evaluation samples per depth", str(n_sentences_per_depth)),
            ("shot-pool samples per depth", str(n_shots_per_depth)),
            ("total grammars", str(total_grammars)),
            ("total evaluation samples", str(total_eval_samples)),
            ("total shot-pool samples", str(total_shot_samples)),
        ],
        condition_examples=condition_examples,
        regeneration_command="uv run python main.py exp_fewshot",
    )


def create_agreement_data(
    grammar_sizes: list[int] = [25, 50, 100, 1_000, 5_000, 7_500, 10_000],
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
):
    """
    Generates grammars which vary overt agreement marking.
    """

    syllable_structure: str = "CVC"
    exp_name = "agreement"
    exp_dir: Path = DATA_DIR / f"{exp_name}_exp"
    exp_dir.mkdir(parents=True, exist_ok=True)

    grammar_names: list[str] = []
    configurations = [
        {"agreement_enabled_a": False, "agreement_enabled_b": False},
        {"agreement_enabled_a": False, "agreement_enabled_b": True},
        {"agreement_enabled_a": True, "agreement_enabled_b": False},
        {"agreement_enabled_a": True, "agreement_enabled_b": True},
    ]

    for config in configurations:
        for g_size in grammar_sizes:
            for index in range(n_grammars_per_size):
                g_seed = deterministic_seed(
                    "agreement",
                    g_size,
                    index,
                    tuple(sorted(config.items())),
                )
                grammar_name = create_grammar(
                    rng_seed=g_seed,
                    syllable_structure_a=syllable_structure,
                    syllable_structure_b=syllable_structure,
                    head_initial_a=True,
                    head_initial_b=True,
                    spec_initial_a=True,
                    spec_initial_b=True,
                    pro_drop_a=False,
                    pro_drop_b=False,
                    n_verbs=max(2, g_size // 5),
                    n_nouns=max(2, g_size // 5),
                    n_adjectives=max(1, g_size // 10),
                    n_propns=max(2, g_size // 10),
                    n_det_def=2,
                    n_det_indef=2,
                    n_prons=6,
                    n_comps=2,
                    orthography_a="latin",
                    orthography_b="latin",
                    exp_name=exp_name,
                    agreement_enabled_a=config["agreement_enabled_a"],
                    agreement_enabled_b=config["agreement_enabled_b"],
                )
                log.info(
                    "Created grammar %s with agreement_a=%s agreement_b=%s seed=%s",
                    grammar_name,
                    config["agreement_enabled_a"],
                    config["agreement_enabled_b"],
                    g_seed,
                )
                generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=g_seed,
                    min_depth=0,
                    max_depth=max_depth,
                    n_samples_per_depth=n_sentences_per_depth,
                    exp_name=exp_name,
                )
                grammar_names.append(grammar_name)

    with open(exp_dir / f"{exp_name}_grammars.txt", "w") as f:
        for name in grammar_names:
            f.write(f"{name}\n")


def create_size_data(
    grammar_sizes: list[int] = [25, 50, 100, 1_000, 5_000, 7_500, 10_000],
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
    exp_name: str = "size",
):
    """
    Generates grammars which vary in size.
    """
    syllable_structure: str = "C*VC"
    target_head_initial: bool = False

    exp_dir: Path = DATA_DIR / (exp_name + "_exp")
    exp_dir.mkdir(parents=True, exist_ok=True)

    grammar_names: list[str] = []
    for g_size in grammar_sizes:
        for _ in range(n_grammars_per_size):
            g_seed = deterministic_seed(exp_name, g_size, _)
            grammar_name = create_grammar(
                rng_seed=g_seed,
                syllable_structure_a=syllable_structure,
                syllable_structure_b=syllable_structure,
                head_initial_a=True,
                head_initial_b=target_head_initial,
                spec_initial_a=True,
                spec_initial_b=True,
                pro_drop_a=False,
                pro_drop_b=False,
                n_verbs=g_size // 5,
                n_nouns=g_size // 5,
                n_adjectives=g_size // 5,
                n_propns=max(2, g_size // 5),
                n_det_def=2,
                n_det_indef=2,
                n_prons=2,
                n_comps=2,
                exp_name=exp_name,
            )
            log.info(
                f"Created grammar {grammar_name} with g_size={g_size}, seed={g_seed}"
            )
            generate_samples(
                grammar_name=grammar_name,
                rng_seed=g_seed,
                min_depth=0,
                max_depth=max_depth,
                n_samples_per_depth=n_sentences_per_depth,
                exp_name=exp_name,
            )
            grammar_names.append(grammar_name)

    with open(exp_dir / f"{exp_name}_grammars.txt", "w") as f:
        for name in grammar_names:
            f.write(f"{name}\n")


def create_complexity_data(
    grammar_sizes: list[int] = [25, 50, 100, 500, 1000],
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
):
    """
    Generates grammars to test how grammatical and sentential complexity
    affect transduction performance.

    Vary:
        - grammar size (number of words): [25, 50, 100, 500, 1000]
        - clause depth: [0, 1, 2, 3, 4, 5]

    Fixed:
        - all grammar hyperparameters (head-initial, spec-initial)
        - all lexical parameters (syllable structure)
    """

    syllable_structure = "CVC"
    head_initial = True
    spec_initial = True
    pro_drop = False

    grammar_names: list[str] = []

    for g_size in grammar_sizes:
        for _ in range(n_grammars_per_size):
            g_seed = deterministic_seed("complexity", g_size, _)
            grammar_name = create_grammar(
                rng_seed=g_seed,
                syllable_structure_a=syllable_structure,
                syllable_structure_b=syllable_structure,
                head_initial_a=head_initial,
                head_initial_b=head_initial,
                spec_initial_a=spec_initial,
                spec_initial_b=spec_initial,
                pro_drop_a=pro_drop,
                pro_drop_b=pro_drop,
                n_verbs=g_size // 5,
                n_nouns=g_size // 5,
                n_adjectives=g_size // 5,
                n_propns=max(2, g_size // 5),
                n_det_def=2,
                n_det_indef=2,
                n_prons=2,
                n_comps=2,
            )
            log.info(
                f"Created grammar {grammar_name} with g_size={g_size}, seed={g_seed}"
            )
            generate_samples(
                grammar_name=grammar_name,
                rng_seed=g_seed,
                min_depth=0,
                max_depth=max_depth,
                n_samples_per_depth=n_sentences_per_depth,
            )
            grammar_names.append(grammar_name)

    with open(DATA_DIR / "complexity_grammars.txt", "w") as f:
        for name in grammar_names:
            f.write(f"{name}\n")


def create_large_complexity_data(
    grammar_sizes: list[int] = [1500, 2500, 10000],
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
):
    """
    Generates grammars to test how grammatical and sentential complexity
    affect transduction performance.

    Vary:
        - grammar size (number of words): [1500, 2500, 10000]
        - clause depth: [0, 1, 2, 3, 4, 5]

    Fixed:
        - all grammar hyperparameters (head-initial, spec-initial)
        - all lexical parameters (syllable structure)
    """

    syllable_structure = "CVC"
    head_initial = True
    spec_initial = True
    pro_drop = False

    grammar_names: list[str] = []

    for g_size in grammar_sizes:
        for _ in range(n_grammars_per_size):
            g_seed = deterministic_seed("large_complexity", g_size, _)
            grammar_name = create_grammar(
                rng_seed=g_seed,
                syllable_structure_a=syllable_structure,
                syllable_structure_b=syllable_structure,
                head_initial_a=head_initial,
                head_initial_b=head_initial,
                spec_initial_a=spec_initial,
                spec_initial_b=spec_initial,
                pro_drop_a=pro_drop,
                pro_drop_b=pro_drop,
                n_verbs=g_size // 5,
                n_nouns=g_size // 5,
                n_adjectives=g_size // 5,
                n_propns=max(2, g_size // 5),
                n_det_def=2,
                n_det_indef=2,
                n_prons=2,
                n_comps=2,
            )
            log.info(
                f"Created grammar {grammar_name} with g_size={g_size}, seed={g_seed}"
            )
            generate_samples(
                grammar_name=grammar_name,
                rng_seed=g_seed,
                min_depth=0,
                max_depth=max_depth,
                n_samples_per_depth=n_sentences_per_depth,
            )
            grammar_names.append(grammar_name)

    with open(DATA_DIR / "complexity_grammars_large.txt", "w") as f:
        for name in grammar_names:
            f.write(f"{name}\n")


def create_grammar(
    rng_seed: int = 80,
    syllable_structure_a: str | None = None,
    syllable_structure_b: str | None = None,
    head_initial_a: bool = True,
    head_initial_b: bool = True,
    spec_initial_a: bool = True,
    spec_initial_b: bool = True,
    pro_drop_a: bool = False,
    pro_drop_b: bool = False,
    n_verbs: int = 10,
    n_nouns: int = 10,
    n_adjectives: int = 10,
    n_propns: int = 5,
    n_det_def: int = 2,
    n_det_indef: int = 2,
    n_prons: int = 2,
    n_comps: int = 2,
    orthography_a: str = "latin",
    orthography_b: str = "latin",
    exp_name: str | None = None,
    agreement_enabled_a: bool = False,
    agreement_enabled_b: bool = False,
    latent_gender_a: bool = False,
    latent_gender_b: bool = False,
    realize_gender_a: bool = False,
    realize_gender_b: bool = False,
    lexical_frequency_profile: str = "zipf_length",
    lexical_frequency_exponent: float = 1.0,
    lexical_frequency_length_unit: str = "chars",
) -> str:
    set_all_seeds(rng_seed)

    a_params = CFGParams(
        rng_seed=rng_seed,
        syllable_structure=syllable_structure_a,
        head_initial=head_initial_a,
        spec_initial=spec_initial_a,
        pro_drop=pro_drop_a,
        verbs=n_verbs,
        nouns=n_nouns,
        adjs=n_adjectives,
        propns=n_propns,
        det_def=n_det_def,
        det_indef=n_det_indef,
        prons=n_prons,
        comps=n_comps,
        orthography=orthography_a,
        agreement_enabled=agreement_enabled_a,
        latent_gender=latent_gender_a,
        realize_gender=realize_gender_a,
        lexical_frequency_profile=lexical_frequency_profile,
        lexical_frequency_exponent=lexical_frequency_exponent,
        lexical_frequency_length_unit=lexical_frequency_length_unit,
    )
    b_params = CFGParams(
        rng_seed=rng_seed + 1,
        syllable_structure=syllable_structure_b,
        head_initial=head_initial_b,
        spec_initial=spec_initial_b,
        pro_drop=pro_drop_b,
        verbs=n_verbs,
        nouns=n_nouns,
        adjs=n_adjectives,
        propns=n_propns,
        det_def=n_det_def,
        det_indef=n_det_indef,
        prons=n_prons,
        comps=n_comps,
        orthography=orthography_b,
        agreement_enabled=agreement_enabled_b,
        latent_gender=latent_gender_b,
        realize_gender=realize_gender_b,
        lexical_frequency_profile=lexical_frequency_profile,
        lexical_frequency_exponent=lexical_frequency_exponent,
        lexical_frequency_length_unit=lexical_frequency_length_unit,
    )
    params = SCFGParams(a=a_params, b=b_params)

    out_dir: Path = DATA_DIR
    if exp_name is not None:
        out_dir /= exp_name + "_exp"
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"grammar_{params.name}.json", "w") as f:
        json.dump(params.to_dict(), f, indent=2, ensure_ascii=False)

    log.info(f"Grammar saved to {out_dir / f'grammar_{params.name}.json'}")

    if params.name is None:
        raise ValueError("Grammar name is None")
    return str(params.name)


def generate_samples(
    grammar_name: str,
    rng_seed: int = 42,
    min_depth: int = 0,
    max_depth: int = 10,
    n_samples_per_depth: int = 2,
    exp_name: str | None = None,
    output_prefix: str = "samples",
):
    filepath = f"grammar_{grammar_name}.json"
    grammar_dir: Path = DATA_DIR
    if exp_name is not None:
        grammar_dir /= exp_name + "_exp"
    with open(grammar_dir / filepath, "r") as f:
        data = json.load(f)

    log.info(f"Loaded grammar from {grammar_dir / filepath}")

    params = SCFGParams.from_dict(data)
    scfg = SCFG(params)

    rng = random.Random(rng_seed)

    samples: list[Dict[str, Union[str, int]]] = []

    depths = list(range(min_depth, max_depth + 1))
    for d in depths:
        for _ in range(n_samples_per_depth):
            sample = scfg.sample(rng=rng, min_depth=d, max_depth=d)
            samples.append(sample)

    assert params.name is not None
    for s in samples:
        s["grammar_name"] = params.name
        s["min_depth"] = min_depth
        s["max_depth"] = max_depth
        s["rng_seed"] = rng_seed

    out_dir: Path = DATA_DIR
    if exp_name is not None:
        out_dir /= exp_name + "_exp"
        out_dir.mkdir(parents=True, exist_ok=True)

    # Save samples to a file
    samples_filepath = out_dir / f"{output_prefix}_{params.name}.jsonl"
    with open(samples_filepath, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def generate_batchfile(
    grammar_name: str,
    exp: str | None = None,
    prompt_type: str = "basic",
    model: str = "o4-mini",
    max_completion_tokens: int | None = None,
    n: int = 1,
    k_shots: int | list[int] = 0,
    data_source: str = "hf",
    hf_repo_id: str | None = None,
):
    data = _load_single_grammar_data(
        grammar_name=grammar_name,
        exp=exp,
        data_source=data_source,
        hf_repo_id=hf_repo_id,
    )
    samples = cast(list[dict[str, Any]], data["samples"])
    dedicated_shots = cast(list[dict[str, Any]] | None, data.get("shots"))
    shot_samples = dedicated_shots or samples
    has_dedicated_shots = dedicated_shots is not None
    grammar = cast(dict[str, object], data["grammar"])
    agreement_metadata = prompt_agreement_metadata(grammar)

    prompt_func = basic_prompt
    k_values = normalize_k_shots(k_shots)
    all_samples = []

    for k in k_values:
        df = pd.DataFrame(samples)
        df["prompt"] = df.apply(
            lambda row: prompt_func(
                grammar_str=prompt_grammar_str(
                    grammar, row["left_phonetic"], prompt_type
                ),
                sample=row["left_phonetic"],
                agreement_metadata=agreement_metadata,
                few_shot_examples=shot_examples_from_samples(
                    shot_samples,
                    k_shots=k,
                    exclude_sample_id=None if has_dedicated_shots else int(row.name),
                ),
            ),
            axis=1,
        )
        df = add_prompt_token_estimates(df, model)
        df = drop_rows_over_model_context(df, model=model, grammar_name=grammar_name)
        warn_for_large_prompts(df, model=model, grammar_name=grammar_name)
        df["json"] = df.apply(
            lambda row: ChatCompletionResponse(
                user_prompt=row["prompt"],
                max_completion_tokens=max_completion_tokens,
                n=n,
                metadata=sample_metadata(
                    grammar_name=grammar_name,
                    sample_id=str(row.name),
                    row=row,
                    grammar=grammar,
                    k_shots=k,
                ),
            ).to_openai_batched_json(
                model=model,
                custom_id=(
                    f"{grammar_name}-sample-{row.name}"
                    if k == 0 and len(k_values) == 1
                    else f"{grammar_name}-k{k}-sample-{row.name}"
                ),
            ),
            axis=1,
        )
        all_samples.append(df)

    all_df = pd.concat(all_samples, ignore_index=True)

    model_pathsafe_name = model.replace("/", "_")
    prompt_label = "" if prompt_type == "basic" else f"_{prompt_type}"
    shot_label = (
        ""
        if k_values == [0]
        else f"_k{k_values[0]}"
        if len(k_values) == 1
        else "_kshots"
    )
    batch_jsonl_filename = (
        f"inputs_{grammar_name}{prompt_label}{shot_label}_{model_pathsafe_name}.jsonl"
    )
    batch_jsonl_path = BATCH_DIR / batch_jsonl_filename
    log.info(f"Writing batch job to {batch_jsonl_path}")

    with open(batch_jsonl_path, "w") as f:
        for j in all_df["json"]:
            f.write(f"{j}\n")


def generate_experiment_batchfile(
    exp: str,
    prompt_type: str = "basic",
    model: str = "gpt-5-nano",
    max_completion_tokens: int | None = None,
    n: int = 1,
    max_filesize_mb: int = 200,
    k_shots: int | list[int] = 0,
    data_source: str = "auto",
    hf_repo_id: str | None = None,
):
    """
    Generates a single batchfile for multiple grammars.
    """
    exp_batch_dir = BATCH_DIR / (exp + "_exp")

    exp_batch_dir.mkdir(parents=True, exist_ok=True)

    k_values = normalize_k_shots(k_shots)
    all_samples = []
    experiment_data = _load_experiment_data(
        exp=exp,
        data_source=data_source,
        hf_repo_id=hf_repo_id,
    )
    for grammar_name, data in experiment_data.items():
        samples = cast(list[dict[str, Any]], data["samples"])
        dedicated_shots = cast(list[dict[str, Any]] | None, data.get("shots"))
        shot_samples = dedicated_shots or samples
        has_dedicated_shots = dedicated_shots is not None
        grammar = cast(dict[str, object], data["grammar"])
        agreement_metadata = prompt_agreement_metadata(grammar)

        log.info(
            "Building prompts for %s/%s with %s samples and %s dedicated shots",
            exp,
            grammar_name,
            len(samples),
            len(dedicated_shots or []),
        )

        for k in k_values:
            df = pd.DataFrame(samples)
            prompt_func = basic_prompt
            df["prompt"] = df.apply(
                lambda row: prompt_func(
                    grammar_str=prompt_grammar_str(
                        grammar, row["left_phonetic"], prompt_type
                    ),
                    sample=row["left_phonetic"],
                    agreement_metadata=agreement_metadata,
                    few_shot_examples=shot_examples_from_samples(
                        shot_samples,
                        k_shots=k,
                        exclude_sample_id=None
                        if has_dedicated_shots
                        else int(row.name),
                    ),
                ),
                axis=1,
            )
            df = add_prompt_token_estimates(df, model)
            df = drop_rows_over_model_context(
                df, model=model, grammar_name=grammar_name
            )
            warn_for_large_prompts(df, model=model, grammar_name=grammar_name)
            df["json"] = df.apply(
                lambda row: ChatCompletionResponse(
                    user_prompt=row["prompt"],
                    max_completion_tokens=max_completion_tokens,
                    n=n,
                    metadata=sample_metadata(
                        grammar_name=grammar_name,
                        sample_id=str(row.name),
                        row=row,
                        grammar=grammar,
                        k_shots=k,
                    ),
                ).to_openai_batched_json(
                    model=model,
                    custom_id=(
                        f"{grammar_name}-sample-{row.name}"
                        if k == 0 and len(k_values) == 1
                        else f"{grammar_name}-k{k}-sample-{row.name}"
                    ),
                ),
                axis=1,
            )

            all_samples.append(df)

    if not all_samples:
        raise ValueError(f"No samples found for experiment {exp}")
    all_df = pd.concat(all_samples, ignore_index=True)

    # if the all_df["json"] column entries as strings exceed max_filesize_mb,
    # split into multiple files
    total_size_bytes = (
        all_df["json"].apply(lambda x: len((x + "-aaaaaa\n").encode("utf-8"))).sum()
    )

    total_size_mb = total_size_bytes / (1024 * 1024) * 1.05
    num_files = max(1, int(total_size_mb // max_filesize_mb) + 1)
    partitioned_dfs = list(np.array_split(all_df, num_files))

    model_pathsafe_name: str = model.replace("/", "_")
    prompt_label = "" if prompt_type == "basic" else f"_{prompt_type}"
    shot_label = (
        ""
        if k_values == [0]
        else f"_k{k_values[0]}"
        if len(k_values) == 1
        else "_kshots"
    )

    for i, raw_part_df in enumerate(partitioned_dfs):
        part_df = cast(pd.DataFrame, raw_part_df)
        fname_hash: str = secrets.token_hex(3)

        # Add fname_hash to each custom_id so we can retrieve the input file
        # without metadata.
        part_df["json"] = part_df["json"].apply(json.loads)
        part_df["json"] = part_df["json"].apply(
            lambda x: {
                **x,
                "custom_id": f"{x['custom_id'].replace('-', f'-{fname_hash}-', 1)}",
            }
        )
        part_df["json"] = part_df["json"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

        base_fname: str = (
            f"inputs_{exp}{prompt_label}{shot_label}_{model_pathsafe_name}"
            f"_part{i + 1}_of_{num_files}"
        )
        fname: str = f"{base_fname}_{fname_hash}.jsonl"
        fpath: Path = exp_batch_dir / fname
        log.info(f"Writing batch job to {fpath}")

        with open(fpath, "w") as f:
            for j in part_df["json"]:
                f.write(f"{j}\n")


def demo():
    a_params = CFGParams.english()
    b_params = CFGParams.german()
    scfg_params = SCFGParams(a_params, b_params)
    scfg = SCFG(scfg_params)
    print(scfg.as_cfg)


def demo_random():
    a_params = CFGParams()
    b_params = CFGParams(orthography="cyrillic")
    scfg_params = SCFGParams(a_params, b_params)
    scfg = SCFG(scfg_params)
    print(scfg.as_cfg)


if __name__ == "__main__":
    fire.Fire(
        {
            # Core functionality
            "create_grammar": create_grammar,
            "gen_samples": generate_samples,
            "gen_batchfile": generate_batchfile,
            "gen_exp_batchfile": generate_experiment_batchfile,
            # Demos
            "demo": demo,
            "demo_random": demo_random,
            # Experiments
            "exp_complexity": create_complexity_data,
            "exp_large_complexity": create_large_complexity_data,
            "exp_wordorder": create_wordorder_data,
            "exp_orthography": create_orthography_data,
            "exp_agreement": create_agreement_data,
            "exp_fewshot": create_fewshot_data,
            "exp_size": create_size_data,
            # TODO: add hash id to input files, attach it to requiest custom id
            # to make grammar identificaiton easier for gemini
        }
    )
