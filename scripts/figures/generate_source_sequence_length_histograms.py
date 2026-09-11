"""Plot source-language sequence-length distributions for main experiments.

Each source sentence is tokenized with OpenAI's ``o200k_base`` encoding. The
script reads the complete sample splits from the Hugging Face dataset rather
than relying on locally downloaded batch inputs.

Example:
    uv run python scripts/figures/generate_source_sequence_length_histograms.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyrootutils
import tiktoken

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks import aesthetics as aes  # noqa: E402
from scfg.hf_data import load_hf_split, resolve_hf_dataset_repo  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache" / "source-sequence-lengths"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
DEFAULT_OUTPUT = FIGURES_DIR / "source_sequence_length_histograms"
DEFAULT_SUMMARY_OUTPUT = CACHE_DIR / "source_sequence_lengths.csv"
OPENAI_ENCODING = "o200k_base"

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "slug": "size",
        "label": "Size",
        "color": aes.sns.color_palette("Reds", n_colors=5)[3],
    },
    {
        "slug": "wordorder",
        "label": "Word order",
        "color": aes.sns.color_palette("Reds", n_colors=5)[3],
    },
    {
        "slug": "agreement",
        "label": "Morphology",
        "color": aes.sns.color_palette("Reds", n_colors=5)[3],
    },
    {
        "slug": "orthography",
        "label": "Orthography",
        "color": aes.sns.color_palette("Reds", n_colors=5)[3],
    },
]


def source_sentence(sample: dict[str, Any]) -> str | None:
    """Return the source-language surface form stored in a sample record."""

    value = sample.get("left_phonetic") or sample.get("left")
    return value if isinstance(value, str) and value else None


def load_sequence_lengths(hf_repo_id: str | None) -> pd.DataFrame:
    """Load and tokenize every source sentence in the main experiments."""

    encoding = tiktoken.get_encoding(OPENAI_ENCODING)
    rows: list[dict[str, object]] = []
    for config in EXPERIMENTS:
        slug = str(config["slug"])
        for sample in load_hf_split(
            resolve_hf_dataset_repo(hf_repo_id), f"{slug}_exp", "samples"
        ):
            text = source_sentence(sample)
            if text is None:
                continue
            rows.append(
                {
                    "experiment": slug,
                    "grammar_name": sample.get("grammar_name"),
                    "sample_id": sample.get("row_index"),
                    "source_sentence": text,
                    "source_tokens": len(encoding.encode(text)),
                }
            )
    if not rows:
        raise ValueError("No source-language samples were found in the HF dataset.")
    return pd.DataFrame(rows)


def plot_histograms(rows: pd.DataFrame):
    """Create a one-column source-sequence-length distribution overview."""

    minimum = int(rows["source_tokens"].min())
    maximum = int(rows["source_tokens"].max())
    bins = np.arange(minimum - 0.5, maximum + 1.5, 1)

    figure = aes.plt.figure(
        figsize=(
            aes.ACL_COLUMN_WIDTH_IN,
            len(EXPERIMENTS) * aes.FIG_HEIGHT_SINGLE_ROW_IN * 1.15,
        )
    )
    grid = figure.add_gridspec(len(EXPERIMENTS), 1, hspace=0.2)
    axes = []
    for index, config in enumerate(EXPERIMENTS):
        axis = figure.add_subplot(grid[index, 0], sharex=axes[0] if axes else None)
        axes.append(axis)
        experiment_rows = rows.loc[rows["experiment"].eq(config["slug"])]
        aes.sns.histplot(
            data=experiment_rows,
            x="source_tokens",
            bins=bins,
            stat="density",
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
        if index < len(EXPERIMENTS) - 1:
            axis.set_xlabel("")
            axis.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Source sequence length\n(OpenAI tokens)")
    figure.tight_layout()
    # figure.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.97)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate source-language sequence-length histograms."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--hf-repo-id",
        help="Optional Hugging Face dataset repo; defaults to the configured repo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_sequence_lengths(args.hf_repo_id)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.summary_output, index=False)
    figure = plot_histograms(rows)
    output_path, _ = aes.save_figure(args.output, fig=figure)
    aes.plt.close(figure)

    counts = rows.groupby("experiment").size().to_dict()
    summary = ", ".join(f"{slug}={count:,}" for slug, count in counts.items())
    print(f"Saved {output_path}")
    print(f"Saved {args.summary_output}")
    print(f"Tokenizer: {OPENAI_ENCODING}; samples: {summary}")


if __name__ == "__main__":
    main()
