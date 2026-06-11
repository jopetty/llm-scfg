from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, cast

DEFAULT_HF_DATASET_REPO = "jowenpetty/scfg"
DEFAULT_EXPERIMENTS = ("size", "wordorder", "orthography", "agreement", "fewshot")


def resolve_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def resolve_hf_dataset_repo(hf_repo_id: str | None = None) -> str:
    return hf_repo_id or os.getenv("LLM_SCFG_HF_REPO_ID") or DEFAULT_HF_DATASET_REPO


def experiment_config_name(exp: str) -> str:
    return exp if exp.endswith("_exp") else f"{exp}_exp"


def experiment_label(config_name: str) -> str:
    return config_name.removesuffix("_exp")


def _merge_raw_json(row: dict[str, Any]) -> dict[str, Any]:
    raw_json = row.get("raw_json")
    if not isinstance(raw_json, str) or not raw_json:
        return row
    parsed = cast(dict[str, Any], json.loads(raw_json))
    for key in ["experiment", "source_file", "row_index", "grammar_name"]:
        if key in row and key not in parsed:
            parsed[key] = row[key]
    if "name" in row and "name" not in parsed:
        parsed["name"] = row["name"]
    return parsed


@lru_cache(maxsize=64)
def load_hf_split(
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
            "huggingface-hub. Run `uv sync` to install the project dependencies."
        ) from exc

    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{config_name}/{split}.parquet",
        token=resolve_hf_token(),
    )
    rows = [
        _merge_raw_json(cast(dict[str, Any], row))
        for row in pq.read_table(path).to_pylist()
    ]
    rows.sort(
        key=lambda row: (
            str(row.get("grammar_name") or row.get("name") or ""),
            int(row.get("row_index") or 0),
        )
    )
    return tuple(rows)


def load_hf_experiment_data(
    *,
    exp: str,
    hf_repo_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    repo_id = resolve_hf_dataset_repo(hf_repo_id)
    config_name = experiment_config_name(exp)

    grammar_rows = load_hf_split(repo_id, config_name, "grammars")
    sample_rows = load_hf_split(repo_id, config_name, "samples")
    shot_rows = load_hf_split(repo_id, config_name, "shots")

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

    return {
        name: {
            "grammar": grammars[name],
            "samples": samples_by_grammar.get(name, []),
            "shots": shots_by_grammar.get(name, []),
        }
        for name in grammars
    }


def discover_hf_experiments(hf_repo_id: str | None = None) -> list[str]:
    repo_id = resolve_hf_dataset_repo(hf_repo_id)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return list(DEFAULT_EXPERIMENTS)

    api = HfApi(token=resolve_hf_token())
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    configs = sorted(
        path.split("/", 1)[0]
        for path in files
        if path.endswith("/grammars.parquet") and "/" in path
    )
    return [experiment_label(config) for config in configs]
