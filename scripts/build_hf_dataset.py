"""Build a Hugging Face-ready dataset from data/<experiment> artifacts.

Layout produced under --output-dir:

    <output-dir>/
        README.md                       (data card with YAML configs front-matter)
        <experiment>/
            grammars.parquet
            samples.parquet
            shots.parquet
        ...
        manifest.json                   (counts, orphan grammars, schema notes)

Each experiment becomes a Hugging Face config with three splits, ``grammars``,
``samples``, and ``shots``. Stable scalar fields are lifted to typed columns;
the full original JSON is preserved in a ``raw_json`` column for fidelity.

Usage:
    uv run python scripts/build_hf_dataset.py
    uv run python scripts/build_hf_dataset.py --output-dir build/hf_dataset
    uv run python scripts/build_hf_dataset.py --experiments=agreement_exp,size_exp
    uv run python scripts/build_hf_dataset.py --repo-id=<owner>/llm-scfg --upload
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import dotenv
import fire
import pyarrow as pa
import pyarrow.parquet as pq
import pyrootutils

PROJECT_ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

dotenv.load_dotenv(PROJECT_ROOT / ".env")


def _resolve_hf_token() -> str | None:
    """Resolve the Hugging Face token from the environment / `.env`.

    Mirrors `scripts/check_hf_auth.py`: prefer `HF_TOKEN`, fall back to
    `HUGGINGFACE_HUB_TOKEN`.
    """
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# Scalar grammar fields lifted from g["a"] / g["b"] into a_*/b_* columns. Any
# field not in this list (vocab lists, paradigms, agreement_suffixes, ...) stays
# in `raw_json` only.
_SIDE_SCALAR_FIELDS: dict[str, pa.DataType] = {
    "head_initial": pa.bool_(),
    "spec_initial": pa.bool_(),
    "pro_drop": pa.bool_(),
    "proper_with_det": pa.bool_(),
    "syllable_structure": pa.string(),
    "avg_syllables_per_word": pa.float64(),
    "max_consonants": pa.int64(),
    "rng_seed": pa.int64(),
    "orthography": pa.string(),
    "agreement_enabled": pa.bool_(),
    "agreement_strategy": pa.string(),
    "verb_agreement": pa.bool_(),
    "noun_number_marking": pa.bool_(),
    "pronouns_are_featured": pa.bool_(),
    "latent_gender": pa.bool_(),
    "realize_gender": pa.bool_(),
    "space_alpha": pa.float64(),
    "space_beta": pa.float64(),
    "syllable_max": pa.int64(),
    "lexical_frequency_profile": pa.string(),
    "lexical_frequency_exponent": pa.float64(),
    "lexical_frequency_length_unit": pa.string(),
}

_SIDE_LIST_FIELDS: tuple[str, ...] = (
    "verbs",
    "nouns",
    "propns",
    "prons",
    "adjs",
    "det_def",
    "det_indef",
    "comps",
    "tenses",
    "asps",
    "agreement_axes",
    "gender_values",
)


def _json_dumps(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _grammar_row(path: Path, experiment: str) -> dict:
    g = json.loads(path.read_text())
    row: dict = {
        "experiment": experiment,
        "name": g.get("name", path.stem.removeprefix("grammar_")),
        "grammar_str": g.get("grammar_str"),
        "n_rules": g.get("n_rules"),
        "n_words": g.get("n_words"),
        "has_agreement_metadata": "agreement_metadata" in g,
        "agreement_metadata_json": _json_dumps(g.get("agreement_metadata")),
        "lexical_frequency_metadata_json": _json_dumps(
            g.get("lexical_frequency_metadata")
        ),
    }
    for side in ("a", "b"):
        sub = g.get(side, {})
        for field in _SIDE_SCALAR_FIELDS:
            row[f"{side}_{field}"] = sub.get(field)
        for field in _SIDE_LIST_FIELDS:
            v = sub.get(field)
            row[f"{side}_{field}"] = list(v) if isinstance(v, list) else None
    row["raw_json"] = json.dumps(g, sort_keys=True, ensure_ascii=False)
    return row


def _example_row(
    line: str,
    experiment: str,
    source_file: str,
    row_index: int,
) -> dict:
    s = json.loads(line)
    subj = s.get("subject_features") or {}
    verb = s.get("verb_features") or {}
    return {
        "experiment": experiment,
        "source_file": source_file,
        "row_index": row_index,
        "grammar_name": s.get("grammar_name"),
        "left": s.get("left"),
        "right": s.get("right"),
        "left_phonetic": s.get("left_phonetic"),
        "right_phonetic": s.get("right_phonetic"),
        "left_tree": s.get("left_tree"),
        "right_tree": s.get("right_tree"),
        "depth": s.get("depth"),
        "min_depth": s.get("min_depth"),
        "max_depth": s.get("max_depth"),
        "rng_seed": s.get("rng_seed"),
        "subject_person": subj.get("person"),
        "subject_number": subj.get("number"),
        "verb_person": verb.get("person"),
        "verb_number": verb.get("number"),
        "agreement_ok": s.get("agreement_ok"),
        "agreement_trace": _json_dumps(s.get("agreement_trace")),
        "possible_right": s.get("possible_right"),
        "possible_right_phonetic": s.get("possible_right_phonetic"),
        "raw_json": json.dumps(s, sort_keys=True, ensure_ascii=False),
    }


_GRAMMARS_SCHEMA = pa.schema(
    [
        ("experiment", pa.string()),
        ("name", pa.string()),
        ("grammar_str", pa.string()),
        ("n_rules", pa.int64()),
        ("n_words", pa.int64()),
        ("has_agreement_metadata", pa.bool_()),
        ("agreement_metadata_json", pa.string()),
        ("lexical_frequency_metadata_json", pa.string()),
        *[
            (f"{side}_{field}", ty)
            for side in ("a", "b")
            for field, ty in _SIDE_SCALAR_FIELDS.items()
        ],
        *[
            (f"{side}_{field}", pa.list_(pa.string()))
            for side in ("a", "b")
            for field in _SIDE_LIST_FIELDS
        ],
        ("raw_json", pa.string()),
    ]
)

_EXAMPLES_SCHEMA = pa.schema(
    [
        ("experiment", pa.string()),
        ("source_file", pa.string()),
        ("row_index", pa.int64()),
        ("grammar_name", pa.string()),
        ("left", pa.string()),
        ("right", pa.string()),
        ("left_phonetic", pa.string()),
        ("right_phonetic", pa.string()),
        ("left_tree", pa.string()),
        ("right_tree", pa.string()),
        ("depth", pa.int64()),
        ("min_depth", pa.int64()),
        ("max_depth", pa.int64()),
        ("rng_seed", pa.int64()),
        ("subject_person", pa.string()),
        ("subject_number", pa.string()),
        ("verb_person", pa.string()),
        ("verb_number", pa.string()),
        ("agreement_ok", pa.bool_()),
        ("agreement_trace", pa.string()),
        ("possible_right", pa.list_(pa.string())),
        ("possible_right_phonetic", pa.list_(pa.string())),
        ("raw_json", pa.string()),
    ]
)


def _write_parquet(rows: list[dict], schema: pa.Schema, out_path: Path) -> None:
    if not rows:
        # Still emit an empty file so the config split is well-defined.
        table = schema.empty_table()
    else:
        table = pa.Table.from_pylist(rows, schema=schema)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")


def _readme(configs: list[str]) -> str:
    fm_configs = "\n".join(
        f"  - config_name: {c}\n"
        f"    data_files:\n"
        f"      - split: grammars\n"
        f"        path: {c}/grammars.parquet\n"
        f"      - split: samples\n"
        f"        path: {c}/samples.parquet\n"
        f"      - split: shots\n"
        f"        path: {c}/shots.parquet"
        for c in configs
    )
    return f"""---
language:
  - en
tags:
  - synthetic
  - scfg
  - grammar
  - translation
size_categories:
  - 10K<n<100K
configs:
{fm_configs}
---

# llm-scfg

Synthetic parallel-language data generated from paired synchronous
context-free grammars (SCFGs). Each *experiment* config holds three splits:

- `grammars` — one row per grammar, with the SCFG source, lexicons, and
  typological knobs for both language *a* and language *b*.
- `samples` — generated (left, right) sentence pairs joined to a grammar by
  `grammar_name`.
- `shots` — held-out example pools for few-shot prompting. This split is empty
  for experiments that do not define `shots_*.jsonl` files.

## Loading

```python
from datasets import load_dataset

grammars = load_dataset("<owner>/llm-scfg", "agreement_exp", split="grammars")
samples  = load_dataset("<owner>/llm-scfg", "agreement_exp", split="samples")
shots    = load_dataset("<owner>/llm-scfg", "fewshot_exp", split="shots")
```

Join `samples.grammar_name == grammars.name` for grammar-conditioned analysis.

## Configs

{chr(10).join(f"- `{c}`" for c in configs)}

## Schemas

### `grammars`

Stable scalar knobs are lifted to `a_*` / `b_*` columns; lexicons stay as list
columns. The full original grammar JSON (including paradigms, agreement
suffix tables, and any experiment-specific extras) is preserved verbatim in
`raw_json`.

- `experiment` (string): matches the config name.
- `name` (string): grammar hash, used as the join key.
- `grammar_str` (string): SCFG source.
- `n_rules`, `n_words` (int64): grammar stats.
- `has_agreement_metadata` (bool): true on configs that record
  agreement_metadata.
- `agreement_metadata_json` (string): full agreement metadata object where
  present.
- `lexical_frequency_metadata_json` (string): lexical-frequency settings for
  the paired grammar.
- `a_*`, `b_*` scalars (typed): head_initial, spec_initial,
  syllable_structure, rng_seed, lexical_frequency_profile,
  lexical_frequency_exponent, lexical_frequency_length_unit, etc. — see column
  names in the parquet schema.
- `a_*`, `b_*` lists (list&lt;string&gt;): verbs, nouns, propns, prons,
  adjs, det_def, det_indef, comps, tenses, asps, agreement_axes,
  gender_values.
- `raw_json` (string): full grammar JSON for round-tripping.

Fields not present on a given experiment (e.g. `a_orthography` on
`complexity_exp`) are null.

### `samples`

- `experiment` (string): matches the config name.
- `source_file` (string): original JSONL basename.
- `row_index` (int64): zero-based line index within `source_file`.
- `grammar_name` (string): join key into `grammars`.
- `left`, `right` (string): parallel surface strings.
- `left_phonetic`, `right_phonetic` (string): phonetic forms.
- `left_tree`, `right_tree` (string): parse trees.
- `depth`, `min_depth`, `max_depth` (int64): derivation-depth metadata.
- `rng_seed` (int64): per-sample seed.
- `subject_person`, `subject_number`, `verb_person`, `verb_number`
  (string): agreement features; null where not applicable.
- `agreement_ok` (bool): null where not applicable.
- `agreement_trace` (string): null where not applicable.
- `possible_right`, `possible_right_phonetic` (list&lt;string&gt;):
  alternates considered during generation; null where not recorded.
- `raw_json` (string): full original sample JSON for round-tripping.

### `shots`

Same schema as `samples`, but rows come from `shots_*.jsonl` files. These rows
are held-out few-shot example pools; they are not evaluation samples.

## Notes

- Experiments tagged with `has_agreement_metadata=true` were generated with
  the agreement-aware pipeline; samples on those configs populate the
  `subject_*` / `verb_*` / `agreement_*` columns.
- Some grammars in `agreement_exp` have no shipped samples file (the original
  artifacts were multi-GB and excluded from the public release). Those rows
  still appear in the `grammars` split; see `manifest.json` for the list.
"""


def _parse_experiments(
    experiments: str | tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | None:
    if experiments is None:
        return None
    if isinstance(experiments, str):
        return tuple(e.strip() for e in experiments.split(",") if e.strip())
    return tuple(str(e).strip() for e in experiments if str(e).strip())


def _load_jsonl_rows(path: Path, experiment: str) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(_example_row(line, experiment, path.name, row_index))
    return rows


def _push_to_hub(
    output_dir: Path,
    repo_id: str,
    *,
    private: bool,
    commit_message: str,
) -> None:
    from huggingface_hub import HfApi

    token = _resolve_hf_token()
    if token is None:
        raise SystemExit(
            "No Hugging Face token found. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN "
            "in .env (see scripts/check_hf_auth.py)."
        )

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        commit_message=commit_message,
    )


def build(
    data_dir: str | Path = PROJECT_ROOT / "data",
    output_dir: str | Path = PROJECT_ROOT / "build" / "hf_dataset",
    experiments: str | tuple[str, ...] | list[str] | None = None,
    repo_id: str | None = None,
    upload: bool = False,
    private: bool = False,
    commit_message: str = "Update llm-scfg dataset",
) -> None:
    """Build parquet shards and a data-card README under `output_dir`.

    Args:
        data_dir: Source `data/` directory containing per-experiment folders.
        output_dir: Where to write parquet shards, README.md, and manifest.json.
        experiments: Optional comma-separated list (or tuple from Fire) of
            experiment names to include; default is all subdirs of `data_dir`.
        repo_id: Optional Hugging Face dataset repo, e.g. `owner/llm-scfg`.
        upload: If true, upload `output_dir` to `repo_id` after building.
        private: Whether to create the dataset repo as private when uploading.
        commit_message: Commit message for direct uploads.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    experiments = _parse_experiments(experiments)

    discovered = sorted(
        p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    discovered = [name for name in discovered if name != "old"]
    selected = list(experiments) if experiments else discovered
    missing = [e for e in selected if e not in discovered]
    if missing:
        raise SystemExit(f"unknown experiments under {data_dir}: {missing}")

    manifest: dict = {"experiments": {}}
    for exp in selected:
        exp_dir = data_dir / exp
        grammar_paths = sorted(exp_dir.glob("grammar_*.json"))
        sample_paths = {
            p.stem.removeprefix("samples_"): p for p in exp_dir.glob("samples_*.jsonl")
        }
        shot_paths = {
            p.stem.removeprefix("shots_"): p for p in exp_dir.glob("shots_*.jsonl")
        }

        grammar_rows = [_grammar_row(p, exp) for p in grammar_paths]
        grammar_names = [r["name"] for r in grammar_rows]
        orphans = [n for n in grammar_names if n not in sample_paths]
        shot_orphans = sorted(n for n in shot_paths if n not in grammar_names)

        sample_rows: list[dict] = []
        for name in grammar_names:
            p = sample_paths.get(name)
            if p is None:
                continue
            sample_rows.extend(_load_jsonl_rows(p, exp))

        shot_rows: list[dict] = []
        for name in grammar_names:
            p = shot_paths.get(name)
            if p is None:
                continue
            shot_rows.extend(_load_jsonl_rows(p, exp))

        _write_parquet(
            grammar_rows, _GRAMMARS_SCHEMA, output_dir / exp / "grammars.parquet"
        )
        _write_parquet(
            sample_rows, _EXAMPLES_SCHEMA, output_dir / exp / "samples.parquet"
        )
        _write_parquet(shot_rows, _EXAMPLES_SCHEMA, output_dir / exp / "shots.parquet")

        manifest["experiments"][exp] = {
            "n_grammars": len(grammar_rows),
            "n_samples": len(sample_rows),
            "n_shots": len(shot_rows),
            "n_orphan_grammars": len(orphans),
            "orphan_grammars": orphans,
            "n_orphan_shot_files": len(shot_orphans),
            "orphan_shot_files": shot_orphans,
        }
        print(
            f"[{exp}] grammars={len(grammar_rows)} samples={len(sample_rows)} "
            f"shots={len(shot_rows)} orphan_grammars={len(orphans)} "
            f"orphan_shot_files={len(shot_orphans)}"
        )

    (output_dir / "README.md").write_text(_readme(selected))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {output_dir}/README.md and manifest.json")
    if upload:
        if not repo_id:
            raise SystemExit("--repo-id is required when --upload=True")
        _push_to_hub(
            output_dir,
            repo_id,
            private=private,
            commit_message=commit_message,
        )
        print(f"uploaded {output_dir} to https://huggingface.co/datasets/{repo_id}")
    else:
        target = repo_id or "<owner>/llm-scfg"
        print(
            f"upload with: huggingface-cli upload {target} "
            f"{output_dir} --repo-type=dataset"
        )


if __name__ == "__main__":
    fire.Fire(build)
