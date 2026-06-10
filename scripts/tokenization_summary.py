from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pyrootutils
import tiktoken

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "cache" / "tokenization"


class TextTokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...


@dataclass
class TiktokenTokenizer:
    name: str
    encoding: tiktoken.Encoding

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)


@dataclass
class HFTokenizer:
    name: str
    tokenizer: Any

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))


@dataclass
class RunningStats:
    values: list[float] = field(default_factory=list)

    def add(self, value: int | float) -> None:
        self.values.append(float(value))

    @property
    def count(self) -> int:
        return len(self.values)

    def mean(self) -> float | None:
        if not self.values:
            return None
        return statistics.fmean(self.values)

    def median(self) -> float | None:
        if not self.values:
            return None
        return statistics.median(self.values)

    def percentile(self, q: float) -> float | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass
class SummaryAccumulator:
    n_grammars: set[str] = field(default_factory=set)
    n_samples: int = 0
    n_items: int = 0
    total_words: int = 0
    single_token_words: int = 0
    chars_per_word: RunningStats = field(default_factory=RunningStats)
    bytes_per_word: RunningStats = field(default_factory=RunningStats)
    tokens_per_word: RunningStats = field(default_factory=RunningStats)
    tokens_per_char: RunningStats = field(default_factory=RunningStats)
    words_per_item: RunningStats = field(default_factory=RunningStats)
    tokens_per_item: RunningStats = field(default_factory=RunningStats)
    tokens_per_item_word: RunningStats = field(default_factory=RunningStats)

    def add_text_item(
        self,
        text: str,
        tokenizer: TextTokenizer,
        *,
        grammar_name: str,
        sample_item: bool = False,
    ) -> None:
        if not text:
            return
        self.n_grammars.add(grammar_name)
        if sample_item:
            self.n_samples += 1
        self.n_items += 1

        words = text.split()
        item_tokens = len(tokenizer.encode(text))
        self.total_words += len(words)
        self.words_per_item.add(len(words))
        self.tokens_per_item.add(item_tokens)
        if words:
            self.tokens_per_item_word.add(item_tokens / len(words))

        for word in words:
            word_tokens = len(tokenizer.encode(word))
            self.chars_per_word.add(len(word))
            self.bytes_per_word.add(len(word.encode("utf-8")))
            self.tokens_per_word.add(word_tokens)
            if word:
                self.tokens_per_char.add(word_tokens / len(word))
            if word_tokens == 1:
                self.single_token_words += 1

    def to_row(self, base: dict[str, str]) -> dict[str, Any]:
        single_token_fraction = (
            self.single_token_words / self.total_words if self.total_words else None
        )
        return {
            **base,
            "n_grammars": len(self.n_grammars),
            "n_samples": self.n_samples,
            "n_items": self.n_items,
            "n_words": self.total_words,
            "mean_chars_per_word": self.chars_per_word.mean(),
            "mean_bytes_per_word": self.bytes_per_word.mean(),
            "mean_tokens_per_word": self.tokens_per_word.mean(),
            "median_tokens_per_word": self.tokens_per_word.median(),
            "p95_tokens_per_word": self.tokens_per_word.percentile(0.95),
            "single_token_word_fraction": single_token_fraction,
            "mean_words_per_item": self.words_per_item.mean(),
            "mean_tokens_per_item": self.tokens_per_item.mean(),
            "p95_tokens_per_item": self.tokens_per_item.percentile(0.95),
            "mean_tokens_per_item_word": self.tokens_per_item_word.mean(),
        }


def load_tokenizers(model_names: list[str]) -> list[TextTokenizer]:
    tokenizers: list[TextTokenizer] = []
    for model in model_names:
        normalized = model.lower()
        if normalized.startswith("hf:"):
            tokenizers.append(load_hf_tokenizer(model[3:]))
            continue
        if "gemma" in normalized or "/" in model:
            tokenizers.append(load_hf_tokenizer(model))
            continue
        try:
            encoding = tiktoken.encoding_for_model(model)
            tokenizer_name = model
        except KeyError:
            encoding = tiktoken.get_encoding(model)
            tokenizer_name = model
        tokenizers.append(TiktokenTokenizer(tokenizer_name, encoding))
    return tokenizers


def load_hf_tokenizer(model: str) -> HFTokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face tokenizers require transformers to be installed."
        ) from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    return HFTokenizer(model, AutoTokenizer.from_pretrained(model, token=token))


def side_vocab_tokens(side: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key in [
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
    ]:
        tokens.extend(str(item) for item in side.get(key, []))

    for key in [
        "verb_paradigms",
        "noun_paradigms",
        "propn_paradigms",
        "pronoun_paradigms",
    ]:
        for item in side.get(key, []):
            if not isinstance(item, dict):
                continue
            if "lemma" in item:
                tokens.append(str(item["lemma"]))
            if "form" in item:
                tokens.append(str(item["form"]))
            for form in (item.get("forms") or {}).values():
                tokens.append(str(form))

    split_tokens: list[str] = []
    for token in tokens:
        split_tokens.extend(token.split())
    return sorted(set(split_tokens))


def word_order_label(side: dict[str, Any]) -> str:
    head_initial = bool(side.get("head_initial", True))
    spec_initial = bool(side.get("spec_initial", True))
    if head_initial and spec_initial:
        return "SVO"
    if not head_initial and spec_initial:
        return "SOV"
    if not head_initial and not spec_initial:
        return "OVS"
    return "VOS"


def agreement_label(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_label = "Agr" if left.get("agreement_enabled") else "NoAgr"
    right_label = "Agr" if right.get("agreement_enabled") else "NoAgr"
    return f"{left_label} -> {right_label}"


def grammar_conditions(experiment: str, grammar: dict[str, Any]) -> dict[str, str]:
    left = grammar.get("a", {})
    right = grammar.get("b", {})
    return {
        "condition_key": "|".join(
            [
                f"n_words={grammar.get('n_words', '')}",
                f"word_order={word_order_label(left)}->{word_order_label(right)}",
                (
                    f"orthography={left.get('orthography', '')}"
                    f"->{right.get('orthography', '')}"
                ),
                f"agreement={agreement_label(left, right)}",
            ]
        ),
        "source_word_order": word_order_label(left),
        "target_word_order": word_order_label(right),
        "source_orthography": str(left.get("orthography", "")),
        "target_orthography": str(right.get("orthography", "")),
        "agreement_condition": agreement_label(left, right),
        "grammar_n_rules": str(grammar.get("n_rules", "")),
        "grammar_n_words": str(grammar.get("n_words", "")),
        "experiment": experiment,
    }


def discover_experiment_dirs(
    data_dir: Path, experiments: list[str] | None
) -> list[Path]:
    if experiments:
        dirs = []
        for exp in experiments:
            name = exp if exp.endswith("_exp") else f"{exp}_exp"
            path = data_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Experiment directory not found: {path}")
            dirs.append(path)
        return dirs
    return sorted(path for path in data_dir.glob("*_exp") if path.is_dir())


def iter_grammars(exp_dir: Path) -> list[Path]:
    index_path = exp_dir / f"{exp_dir.name.removesuffix('_exp')}_grammars.txt"
    if not index_path.exists():
        return sorted(exp_dir.glob("grammar_*.json"))
    paths: list[Path] = []
    for line in index_path.read_text().splitlines():
        grammar_name = line.strip()
        if grammar_name:
            paths.append(exp_dir / f"grammar_{grammar_name}.json")
    return [path for path in paths if path.exists()]


def add_to_accumulators(
    accumulators: dict[tuple[tuple[str, str], ...], SummaryAccumulator],
    *,
    base: dict[str, str],
    text: str,
    tokenizer: TextTokenizer,
    grammar_name: str,
    sample_item: bool = False,
) -> None:
    key = tuple(sorted({**base, "tokenizer": tokenizer.name}.items()))
    accumulators[key].add_text_item(
        text,
        tokenizer,
        grammar_name=grammar_name,
        sample_item=sample_item,
    )


def summarize(
    *,
    data_dir: Path,
    output_path: Path,
    experiments: list[str] | None,
    model_names: list[str],
    max_samples_per_grammar: int | None,
) -> dict[str, Any]:
    tokenizers = load_tokenizers(model_names)
    accumulators: dict[tuple[tuple[str, str], ...], SummaryAccumulator] = defaultdict(
        SummaryAccumulator
    )

    for exp_dir in discover_experiment_dirs(data_dir, experiments):
        experiment = exp_dir.name.removesuffix("_exp")
        for grammar_path in iter_grammars(exp_dir):
            with open(grammar_path) as handle:
                grammar = json.load(handle)
            grammar_name = str(grammar.get("name") or grammar_path.stem[8:])
            conditions = grammar_conditions(experiment, grammar)

            for tokenizer in tokenizers:
                add_to_accumulators(
                    accumulators,
                    base={
                        **conditions,
                        "corpus": "grammar_text",
                        "language_side": "both",
                        "language_role": "prompt",
                        "language_orthography": "mixed",
                    },
                    text=str(grammar.get("grammar_str", "")),
                    tokenizer=tokenizer,
                    grammar_name=grammar_name,
                )

                for side, role in [("a", "source"), ("b", "target")]:
                    side_data = grammar.get(side, {})
                    add_to_accumulators(
                        accumulators,
                        base={
                            **conditions,
                            "corpus": "vocabulary_words",
                            "language_side": side,
                            "language_role": role,
                            "language_orthography": str(
                                side_data.get("orthography", "")
                            ),
                        },
                        text=" ".join(side_vocab_tokens(side_data)),
                        tokenizer=tokenizer,
                        grammar_name=grammar_name,
                    )

            samples_path = exp_dir / f"samples_{grammar_name}.jsonl"
            if not samples_path.exists():
                continue
            with open(samples_path) as handle:
                for sample_idx, line in enumerate(handle):
                    if (
                        max_samples_per_grammar is not None
                        and sample_idx >= max_samples_per_grammar
                    ):
                        break
                    sample = json.loads(line)
                    for tokenizer in tokenizers:
                        for side, role, field_name in [
                            ("a", "source", "left_phonetic"),
                            ("b", "target", "right_phonetic"),
                        ]:
                            side_data = grammar.get(side, {})
                            add_to_accumulators(
                                accumulators,
                                base={
                                    **conditions,
                                    "corpus": "sample_sentences",
                                    "language_side": side,
                                    "language_role": role,
                                    "language_orthography": str(
                                        side_data.get("orthography", "")
                                    ),
                                },
                                text=str(sample.get(field_name) or ""),
                                tokenizer=tokenizer,
                                grammar_name=grammar_name,
                                sample_item=True,
                            )

    rows = [
        accumulator.to_row(dict(key))
        for key, accumulator in sorted(accumulators.items(), key=lambda item: item[0])
    ]
    payload = {
        "metadata": {
            "data_dir": str(data_dir),
            "models": model_names,
            "max_samples_per_grammar": max_samples_per_grammar,
            "n_rows": len(rows),
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_table(rows: list[dict[str, Any]], *, limit: int) -> None:
    columns = [
        "experiment",
        "corpus",
        "language_role",
        "language_orthography",
        "tokenizer",
        "grammar_n_words",
        "target_word_order",
        "agreement_condition",
        "n_items",
        "n_words",
        "mean_tokens_per_word",
        "p95_tokens_per_word",
        "single_token_word_fraction",
        "mean_tokens_per_item",
    ]
    display_rows = rows[:limit]
    widths = {
        column: max(
            len(column),
            *[len(format_number(row.get(column), digits=3)) for row in display_rows],
        )
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in display_rows:
        print(
            "  ".join(
                format_number(row.get(column), digits=3).ljust(widths[column])
                for column in columns
            )
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows written to JSON")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize tokenizer behavior for generated SCFG grammars, "
            "vocabularies, and sampled sentences."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing *_exp experiment folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "tokenization_summary.json",
        help="JSON file to write.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment names, e.g. size wordorder_large.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5"],
        help=(
            "Tokenizer names. Use tiktoken model/encoding names such as gpt-5 "
            "or cl100k_base, or Hugging Face model names such as "
            "google/gemma-3-12b-it."
        ),
    )
    parser.add_argument(
        "--max-samples-per-grammar",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    parser.add_argument(
        "--table-limit",
        type=int,
        default=80,
        help="Maximum number of rows to print to the terminal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(
        data_dir=args.data_dir,
        output_path=args.output,
        experiments=args.experiments,
        model_names=args.models,
        max_samples_per_grammar=args.max_samples_per_grammar,
    )
    print_table(payload["rows"], limit=args.table_limit)
    print(f"\nWrote JSON summary to {args.output}")


if __name__ == "__main__":
    main()
