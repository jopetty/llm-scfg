from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyrootutils
import tiktoken

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scfg.hf_data import (  # noqa: E402
    discover_hf_experiments,
    experiment_label,
    load_hf_experiment_data,
    resolve_hf_dataset_repo,
)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "cache" / "lexical_frequency"


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranked = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranked[order[k]] = rank
        i = j + 1
    return ranked


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denominator = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if not x_denominator or not y_denominator:
        return None
    return numerator / (x_denominator * y_denominator)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(ranks(xs), ranks(ys))


def load_encoder(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def experiment_dirs(data_dir: Path, experiments: list[str] | None) -> list[Path]:
    if experiments:
        return [
            data_dir
            / (experiment if experiment.endswith("_exp") else f"{experiment}_exp")
            for experiment in experiments
        ]
    return sorted(path for path in data_dir.glob("*_exp") if path.is_dir())


def iter_local_experiment_data(
    data_dir: Path, experiments: list[str] | None
) -> list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]]:
    records: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] = []
    for exp_dir in experiment_dirs(data_dir, experiments):
        if not exp_dir.exists():
            continue
        experiment = exp_dir.name.removesuffix("_exp")
        for grammar_path in sorted(exp_dir.glob("grammar_*.json")):
            with open(grammar_path) as handle:
                grammar = json.load(handle)
            grammar_name = str(
                grammar.get("name") or grammar_path.stem.removeprefix("grammar_")
            )
            samples_path = exp_dir / f"samples_{grammar_name}.jsonl"
            if not samples_path.exists():
                continue
            with open(samples_path) as handle:
                samples = [json.loads(line) for line in handle]
            records.append((experiment, grammar_name, grammar, samples))
    return records


def iter_hf_experiment_data(
    hf_repo_id: str | None, experiments: list[str] | None
) -> list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]]:
    records: list[tuple[str, str, dict[str, Any], list[dict[str, Any]]]] = []
    selected_experiments = experiments or discover_hf_experiments(hf_repo_id)
    for experiment in selected_experiments:
        exp_data = load_hf_experiment_data(exp=experiment, hf_repo_id=hf_repo_id)
        for grammar_name, item in exp_data.items():
            records.append(
                (
                    experiment_label(
                        str(item["grammar"].get("experiment", experiment))
                    ),
                    grammar_name,
                    item["grammar"],
                    item["samples"],
                )
            )
    return records


def side_metadata(grammar: dict[str, Any], side: str) -> dict[str, str]:
    side_params = grammar.get("a" if side == "source" else "b", {})
    frequency = grammar.get("lexical_frequency_metadata", {})
    frequency_side = frequency.get("a" if side == "source" else "b", {})
    return {
        "orthography": str(side_params.get("orthography", "")),
        "word_order": (
            f"head_initial={side_params.get('head_initial', True)},"
            f"spec_initial={side_params.get('spec_initial', True)}"
        ),
        "lexical_frequency_profile": str(frequency_side.get("profile", "")),
        "lexical_frequency_exponent": str(frequency_side.get("exponent", "")),
        "lexical_frequency_length_unit": str(frequency_side.get("length_unit", "")),
    }


def sample_words(sample: dict[str, Any], side: str) -> list[str]:
    key = "left_phonetic" if side == "source" else "right_phonetic"
    return [
        word
        for word in str(sample.get(key, "")).split()
        if word and not word.startswith("∅")
    ]


@dataclass
class GroupCounts:
    word_counts: Counter[str] = field(default_factory=Counter)
    grammars: set[str] = field(default_factory=set)
    samples: int = 0
    base: dict[str, str] = field(default_factory=dict)

    def add_sample(self, grammar_name: str, sample: dict[str, Any], side: str) -> None:
        self.grammars.add(grammar_name)
        self.samples += 1
        self.word_counts.update(sample_words(sample, side))


def summarize_group(group: GroupCounts, encoder: tiktoken.Encoding) -> dict[str, Any]:
    words = list(group.word_counts)
    frequencies = [float(group.word_counts[word]) for word in words]
    char_lengths = [float(len(word.replace(" ", ""))) for word in words]
    token_lengths = [float(len(encoder.encode(word))) for word in words]
    top_words = [
        {
            "word": word,
            "frequency": count,
            "char_length": len(word.replace(" ", "")),
            "token_length": len(encoder.encode(word)),
        }
        for word, count in group.word_counts.most_common(20)
    ]
    return {
        **group.base,
        "n_grammars": len(group.grammars),
        "n_samples": group.samples,
        "n_types": len(group.word_counts),
        "n_tokens": sum(group.word_counts.values()),
        "spearman_frequency_char_length": spearman(frequencies, char_lengths),
        "spearman_frequency_token_length": spearman(frequencies, token_lengths),
        "top_words": top_words,
    }


def summarize(
    *,
    data_dir: Path,
    experiments: list[str] | None,
    model: str,
    max_samples_per_grammar: int | None,
    data_source: str = "hf",
    hf_repo_id: str | None = None,
) -> list[dict[str, Any]]:
    encoder = load_encoder(model)
    groups: dict[tuple[str, str, str, str, str], GroupCounts] = {}
    normalized_source = data_source.lower()
    if normalized_source == "local":
        records = iter_local_experiment_data(data_dir, experiments)
    elif normalized_source == "hf":
        records = iter_hf_experiment_data(hf_repo_id, experiments)
    else:
        raise ValueError("data_source must be one of: local, hf")

    for experiment, grammar_name, grammar, samples in records:
        side_bases = {
            side: {
                "experiment": experiment,
                "side": side,
                "grammar_n_words": str(grammar.get("n_words", "")),
                "grammar_n_rules": str(grammar.get("n_rules", "")),
                **side_metadata(grammar, side),
            }
            for side in ["source", "target"]
        }
        for side, base in side_bases.items():
            key = (
                base["experiment"],
                base["side"],
                base["orthography"],
                base["word_order"],
                base["lexical_frequency_profile"],
            )
            groups.setdefault(key, GroupCounts(base=base))

        for sample_index, sample in enumerate(samples):
            if (
                max_samples_per_grammar is not None
                and sample_index >= max_samples_per_grammar
            ):
                break
            for side, base in side_bases.items():
                key = (
                    base["experiment"],
                    base["side"],
                    base["orthography"],
                    base["word_order"],
                    base["lexical_frequency_profile"],
                )
                groups[key].add_sample(grammar_name, sample, side)

    return [
        summarize_group(group, encoder)
        for _, group in sorted(groups.items(), key=lambda item: item[0])
    ]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "experiment",
        "side",
        "orthography",
        "word_order",
        "lexical_frequency_profile",
        "n_grammars",
        "n_types",
        "n_tokens",
        "spearman_frequency_char_length",
        "spearman_frequency_token_length",
    ]
    widths = {
        column: max(len(column), *[len(fmt(row.get(column))) for row in rows])
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  ".join(fmt(row.get(column)).ljust(widths[column]) for column in columns)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize realized word frequency and word length correlations."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--data-source", choices=["hf", "local"], default="hf")
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help=(
            "Hugging Face dataset repo id. Defaults to LLM_SCFG_HF_REPO_ID "
            "or jowenpetty/scfg."
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "summary.json")
    parser.add_argument("--experiments", nargs="*")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--max-samples-per-grammar", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize(
        data_dir=args.data_dir,
        experiments=args.experiments,
        model=args.model,
        max_samples_per_grammar=args.max_samples_per_grammar,
        data_source=args.data_source,
        hf_repo_id=args.hf_repo_id,
    )
    print_table(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(
            {
                "metadata": {
                    "model": args.model,
                    "data_source": args.data_source,
                    "data_dir": str(args.data_dir),
                    "hf_repo_id": resolve_hf_dataset_repo(args.hf_repo_id),
                    "experiments": args.experiments,
                    "max_samples_per_grammar": args.max_samples_per_grammar,
                },
                "rows": rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
