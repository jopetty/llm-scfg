"""Build a HuggingFace-ready dataset from data/<experiment>/{grammar,samples}_*.

Layout produced under --output-dir:

    <output-dir>/
        README.md                       (data card with YAML configs front-matter)
        <experiment>/
            grammars.parquet
            samples.parquet
        ...
        manifest.json                   (counts, orphan grammars, schema notes)

Each experiment becomes a HuggingFace config with two splits, ``grammars`` and
``samples``. Stable scalar fields are lifted to typed columns; the full
original JSON is preserved in a ``raw_json`` column for fidelity.

Usage:
    uv run python scripts/build_hf_dataset.py
    uv run python scripts/build_hf_dataset.py --output-dir build/hf_dataset
    uv run python scripts/build_hf_dataset.py --experiments=agreement_exp,size_exp
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import pyarrow as pa
import pyarrow.parquet as pq
import pyrootutils

PROJECT_ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def _grammar_row(path: Path, experiment: str) -> dict:
    g = json.loads(path.read_text())
    row: dict = {
        "experiment": experiment,
        "name": g.get("name", path.stem.removeprefix("grammar_")),
        "grammar_str": g.get("grammar_str"),
        "n_rules": g.get("n_rules"),
        "n_words": g.get("n_words"),
        "has_agreement_metadata": "agreement_metadata" in g,
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


def _sample_row(line: str, experiment: str) -> dict:
    s = json.loads(line)
    subj = s.get("subject_features") or {}
    verb = s.get("verb_features") or {}
    return {
        "experiment": experiment,
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
        "agreement_trace": s.get("agreement_trace"),
        "possible_right": s.get("possible_right"),
        "possible_right_phonetic": s.get("possible_right_phonetic"),
    }


_GRAMMARS_SCHEMA = pa.schema(
    [
        ("experiment", pa.string()),
        ("name", pa.string()),
        ("grammar_str", pa.string()),
        ("n_rules", pa.int64()),
        ("n_words", pa.int64()),
        ("has_agreement_metadata", pa.bool_()),
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

_SAMPLES_SCHEMA = pa.schema(
    [
        ("experiment", pa.string()),
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
        f"        path: {c}/samples.parquet"
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
context-free grammars (SCFGs). Each *experiment* config holds two splits:

- `grammars` — one row per grammar, with the SCFG source, lexicons, and
  typological knobs for both language *a* and language *b*.
- `samples` — generated (left, right) sentence pairs joined to a grammar by
  `grammar_name`.

## Loading

```python
from datasets import load_dataset

grammars = load_dataset("<owner>/llm-scfg", "agreement_exp", split="grammars")
samples  = load_dataset("<owner>/llm-scfg", "agreement_exp", split="samples")
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
- `a_*`, `b_*` scalars (typed): head_initial, spec_initial,
  syllable_structure, rng_seed, etc. — see column names in the parquet
  schema.
- `a_*`, `b_*` lists (list&lt;string&gt;): verbs, nouns, propns, prons,
  adjs, det_def, det_indef, comps, tenses, asps, agreement_axes,
  gender_values.
- `raw_json` (string): full grammar JSON for round-tripping.

Fields not present on a given experiment (e.g. `a_orthography` on
`complexity_exp`) are null.

### `samples`

- `experiment` (string): matches the config name.
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

## Notes

- Experiments tagged with `has_agreement_metadata=true` were generated with
  the agreement-aware pipeline; samples on those configs populate the
  `subject_*` / `verb_*` / `agreement_*` columns.
- Some grammars in `agreement_exp` have no shipped samples file (the original
  artifacts were multi-GB and excluded from the public release). Those rows
  still appear in the `grammars` split; see `manifest.json` for the list.
"""


def build(
    data_dir: str | Path = PROJECT_ROOT / "data",
    output_dir: str | Path = PROJECT_ROOT / "build" / "hf_dataset",
    experiments: str | tuple[str, ...] | None = None,
) -> None:
    """Build parquet shards and a data-card README under `output_dir`.

    Args:
        data_dir: Source `data/` directory containing per-experiment folders.
        output_dir: Where to write parquet shards, README.md, and manifest.json.
        experiments: Optional comma-separated list (or tuple from Fire) of
            experiment names to include; default is all subdirs of `data_dir`.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if isinstance(experiments, str):
        experiments = tuple(e.strip() for e in experiments.split(",") if e.strip())

    discovered = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
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

        grammar_rows = [_grammar_row(p, exp) for p in grammar_paths]
        grammar_names = [r["name"] for r in grammar_rows]
        orphans = [n for n in grammar_names if n not in sample_paths]

        sample_rows: list[dict] = []
        for name in grammar_names:
            p = sample_paths.get(name)
            if p is None:
                continue
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample_rows.append(_sample_row(line, exp))

        _write_parquet(
            grammar_rows, _GRAMMARS_SCHEMA, output_dir / exp / "grammars.parquet"
        )
        _write_parquet(
            sample_rows, _SAMPLES_SCHEMA, output_dir / exp / "samples.parquet"
        )

        manifest["experiments"][exp] = {
            "n_grammars": len(grammar_rows),
            "n_samples": len(sample_rows),
            "n_orphan_grammars": len(orphans),
            "orphan_grammars": orphans,
        }
        print(
            f"[{exp}] grammars={len(grammar_rows)} samples={len(sample_rows)} "
            f"orphan_grammars={len(orphans)}"
        )

    (output_dir / "README.md").write_text(_readme(selected))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {output_dir}/README.md and manifest.json")
    print(
        "upload with: huggingface-cli upload <owner>/llm-scfg "
        f"{output_dir} --repo-type=dataset"
    )


if __name__ == "__main__":
    fire.Fire(build)
