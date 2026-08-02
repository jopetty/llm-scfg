"""Generate prompt-length histograms for the four main experiment groups.

Prompt text is retokenized with OpenAI's ``o200k_base`` encoding, so all rows
use one tokenizer regardless of the model that originally evaluated them.

Example:
    uv run python scripts/figures/generate_prompt_length_histograms.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pyrootutils
import tiktoken

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks import aesthetics as aes  # noqa: E402

BATCH_DIR = PROJECT_ROOT / "batches"
CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache" / "prompt-lengths"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
DEFAULT_OUTPUT = FIGURES_DIR / "prompt_length_histograms"
DEFAULT_SUMMARY_OUTPUT = CACHE_DIR / "prompt_lengths.csv"
OPENAI_ENCODING = "o200k_base"
INPUT_SENTENCE_PREFIX = "Here is the input sentence: `"
INPUT_SENTENCE_SUFFIX = "`.\n\n    Remember to end your response"
Source = Literal["old", "new"]

# Switch individual rows as their Hugging Face-backed reruns become available.
DATA_SOURCE_BY_EXPERIMENT: dict[str, Source] = {
    "size": "new",
    "wordorder": "new",
    "agreement": "new",
    "orthography": "new",
}

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "slug": "size",
        "label": "Size",
        "color": aes.sns.color_palette("Reds", n_colors=5)[3],
        "old_globs": ["size_exp/inputs_size_gpt-5_part*.jsonl"],
        "new_globs": ["size_exp/inputs_size_gpt-5.6-luna_part*.jsonl"],
    },
    {
        "slug": "wordorder",
        "label": "Word order",
        "color": aes.sns.color_palette("Reds", n_colors=5)[3],
        "old_globs": [
            "old/2026-06-10-drop-large-suffix/wordorder_large_exp/"
            "inputs_wordorder_large_gpt-5_part*.jsonl"
        ],
        "new_globs": ["wordorder_exp/inputs_wordorder_gpt-5.6-sol_part*.jsonl"],
    },
    {
        "slug": "agreement",
        "label": "Morphology",
        "color": aes.sns.color_palette("Reds", n_colors=5)[3],
        "old_globs": [
            "agreement_exp_compact/inputs_agreement_compact_gpt-5_part*.jsonl"
        ],
        "new_globs": ["agreement_exp/inputs_agreement_gpt-5.6-luna_part*.jsonl"],
    },
    {
        "slug": "orthography",
        "label": "Orthography",
        "color": aes.sns.color_palette("Reds", n_colors=5)[3],
        "old_globs": [
            "old/2026-06-10-drop-large-suffix/orthography_large_exp/"
            "inputs_orthography_large_gpt-5_part*.jsonl"
        ],
        "new_globs": ["orthography_exp/inputs_orthography_gpt-5.6-sol_part*.jsonl"],
    },
]


def input_paths(patterns: list[str]) -> list[Path]:
    """Resolve input files without accidentally treating outputs as inputs."""

    paths = {
        path
        for pattern in patterns
        for path in BATCH_DIR.glob(pattern)
        if not path.name.endswith("_output.jsonl")
    }
    return sorted(paths)


def prompt_text(record: dict[str, Any]) -> str:
    body = record.get("body")
    if not isinstance(body, dict):
        return ""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        message["content"]
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), str)
    )


def prompt_token_count(
    text: str,
    encoding,
    prefix_cache: dict[str, int],
    suffix_cache: dict[str, int],
) -> int:
    """Tokenize a repeated grammar prefix once and each sentence separately."""

    sentence_start = text.rfind(INPUT_SENTENCE_PREFIX)
    if sentence_start < 0:
        return len(encoding.encode(text))
    sentence_start += len(INPUT_SENTENCE_PREFIX)
    sentence_end = text.find(INPUT_SENTENCE_SUFFIX, sentence_start)
    if sentence_end < 0:
        return len(encoding.encode(text))

    prefix = text[:sentence_start]
    sentence = text[sentence_start:sentence_end]
    suffix = text[sentence_end:]
    prefix_tokens = prefix_cache.get(prefix)
    if prefix_tokens is None:
        prefix_tokens = len(encoding.encode(prefix))
        prefix_cache[prefix] = prefix_tokens
    suffix_tokens = suffix_cache.get(suffix)
    if suffix_tokens is None:
        suffix_tokens = len(encoding.encode(suffix))
        suffix_cache[suffix] = suffix_tokens
    return prefix_tokens + len(encoding.encode(sentence)) + suffix_tokens


def load_prompt_lengths(paths: list[Path], encoding) -> pd.DataFrame:
    """Tokenize every OpenAI batch-input prompt without repeated grammar work."""

    if not paths:
        raise FileNotFoundError("No matching batch input files were found.")

    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    prefix_cache: dict[str, int] = {}
    suffix_cache: dict[str, int] = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                custom_id = record.get("custom_id")
                if not isinstance(custom_id, str) or custom_id in seen_ids:
                    continue
                text = prompt_text(record)
                if not text:
                    continue
                seen_ids.add(custom_id)
                rows.append(
                    {
                        "input_file": path.name,
                        "custom_id": custom_id,
                        "prompt_tokens": prompt_token_count(
                            text, encoding, prefix_cache, suffix_cache
                        ),
                    }
                )
    if not rows:
        raise ValueError("No prompt text was found in the selected batch inputs.")
    return pd.DataFrame(rows)


def plot_histograms(rows: pd.DataFrame, configs: list[dict[str, Any]]):
    """Create a one-column prompt-length overview with shared log-scale bins."""

    minimum = int(rows["prompt_tokens"].min())
    maximum = int(rows["prompt_tokens"].max())
    bins = np.geomspace(minimum, maximum, num=41)

    figure = aes.plt.figure(
        figsize=(
            aes.ACL_COLUMN_WIDTH_IN,
            len(configs) * aes.FIG_HEIGHT_SINGLE_ROW_IN * 1.15,
        )
    )
    grid = figure.add_gridspec(len(configs), 1, hspace=0.2)
    axes = []
    for index, config in enumerate(configs):
        axis = figure.add_subplot(grid[index, 0], sharex=axes[0] if axes else None)
        axes.append(axis)
        experiment_rows = rows.loc[rows["experiment"].eq(config["slug"])]
        aes.sns.histplot(
            data=experiment_rows,
            x="prompt_tokens",
            stat="density",
            bins=bins,
            color=config["color"],
            edgecolor="white",
            linewidth=0.35,
            ax=axis,
        )

        axis.set_title(
            f"{config['label']}",
            loc="right",
            y=0.7,
        )
        axis.set_ylabel("Density")
        axis.grid(axis="y", linestyle="--", alpha=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if index < len(configs) - 1:
            axis.set_xlabel("")
            axis.tick_params(labelbottom=False)

    for axis in axes:
        axis.set_xscale("log")
    axes[-1].set_xlabel("Prompt length (OpenAI tokens)")
    axes[-1].xaxis.set_major_formatter(aes.NICE_FORMATTER)
    # figure.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.97)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prompt-length histograms for the main experiments."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    for slug in DATA_SOURCE_BY_EXPERIMENT:
        parser.add_argument(
            f"--{slug}-source",
            choices=("old", "new"),
            default=DATA_SOURCE_BY_EXPERIMENT[slug],
            help=f"Data source for the {slug} row (default: %(default)s).",
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    encoding = tiktoken.get_encoding(OPENAI_ENCODING)
    frames: list[pd.DataFrame] = []
    sources: dict[str, Source] = {}
    for config in EXPERIMENTS:
        slug = cast(str, config["slug"])
        source = cast(Source, getattr(args, f"{slug}_source"))
        paths = input_paths(cast(list[str], config[f"{source}_globs"]))
        frame = load_prompt_lengths(paths, encoding)
        frame["experiment"] = slug
        frame["source"] = source
        frames.append(frame)
        sources[slug] = source

    rows = pd.concat(frames, ignore_index=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.summary_output, index=False)
    figure = plot_histograms(rows, EXPERIMENTS)
    output_path, _ = aes.save_figure(args.output, fig=figure)
    aes.plt.close(figure)

    source_summary = ", ".join(f"{slug}={source}" for slug, source in sources.items())
    print(f"Saved {output_path}")
    print(f"Saved {args.summary_output}")
    print(f"Tokenizer: {OPENAI_ENCODING}; sources: {source_summary}")


if __name__ == "__main__":
    main()
