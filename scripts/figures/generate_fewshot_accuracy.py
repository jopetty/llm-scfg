"""Generate few-shot accuracy curves by grammar and sequence size.

The script discovers the multi-condition ``*_kshots_*`` OpenAI-style batch
inputs and their outputs in ``batches/fewshot_exp``.  It joins records by
``custom_id`` and plots the number of in-context examples as the series.

Example:
    uv run python scripts/figures/generate_fewshot_accuracy.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyrootutils
from matplotlib.ticker import SymmetricalLogLocator

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks import aesthetics as aes  # noqa: E402

BATCH_DIR = PROJECT_ROOT / "batches" / "fewshot_exp"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache" / "fewshot-accuracy"
DEFAULT_OUTPUT = FIGURES_DIR / "fewshot_accuracy"
DEFAULT_SUMMARY_OUTPUT = CACHE_DIR / "exact_match_rows.csv"
ANSWER_RE = re.compile(
    r"final\s*answer\s*(?::|-|—)?\s*(?:is\s*)?([^\n]+)",
    re.IGNORECASE | re.DOTALL,
)


def fuzzy_model(model: str | None) -> str:
    """Collapse dated provider model names to a stable model family."""

    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model or "")


def extract_answer(model_response: str | None) -> str | None:
    """Extract the final-answer string emitted by an evaluated model."""

    if not isinstance(model_response, str):
        return None
    matches = ANSWER_RE.findall(model_response)
    if not matches:
        return None
    answer = re.sub(r"[^\w\s]", "", matches[-1], flags=re.UNICODE).strip()
    return answer or None


def response_content(record: dict[str, Any]) -> str | None:
    body = (record.get("response") or {}).get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    content = ((choices[0] or {}).get("message") or {}).get("content")
    return content if isinstance(content, str) else None


def response_model(record: dict[str, Any]) -> str | None:
    model = ((record.get("response") or {}).get("body") or {}).get("model")
    return model if isinstance(model, str) else None


def load_inputs(batch_dir: Path) -> dict[str, dict[str, Any]]:
    """Load few-shot evaluation metadata keyed by provider custom ID."""

    inputs: dict[str, dict[str, Any]] = {}
    for path in sorted(batch_dir.glob("inputs_*_kshots_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                custom_id = record.get("custom_id")
                body = record.get("body") or {}
                metadata = body.get("metadata") or {}
                if not isinstance(custom_id, str) or not isinstance(metadata, dict):
                    continue
                if custom_id in inputs:
                    raise ValueError(f"Duplicate input custom ID: {custom_id}")
                inputs[custom_id] = {
                    "input_file": path.name,
                    "model": body.get("model"),
                    "input_sentence": metadata.get("input_sentence"),
                    "output_sentence": metadata.get("output_sentence"),
                    "n_rules": metadata.get("n_rules"),
                    "n_words": metadata.get("n_words"),
                    "k_shots": metadata.get("k_shots"),
                }
    if not inputs:
        raise FileNotFoundError(
            f"No multi-condition k-shot input JSONL files found under {batch_dir}"
        )
    return inputs


def load_rows(batch_dir: Path) -> pd.DataFrame:
    """Join available few-shot output records to their input metadata."""

    inputs = load_inputs(batch_dir)
    output_paths = sorted(batch_dir.glob("*_output.jsonl"))
    if not output_paths:
        raise FileNotFoundError(f"No output JSONL files found under {batch_dir}")

    rows: list[dict[str, Any]] = []
    ignored_output_ids = 0
    for path in output_paths:
        with path.open() as handle:
            for line in handle:
                output = json.loads(line)
                custom_id = output.get("custom_id")
                if not isinstance(custom_id, str):
                    continue
                input_metadata = inputs.get(custom_id)
                if input_metadata is None:
                    ignored_output_ids += 1
                    continue
                model = response_model(output) or input_metadata["model"]
                rows.append(
                    {
                        "batch_file": path.name,
                        "custom_id": custom_id,
                        "model": model,
                        "fuzzy_model": fuzzy_model(model),
                        "model_answer": extract_answer(response_content(output)),
                        **input_metadata,
                    }
                )

    if not rows:
        raise ValueError("No output records could be joined to k-shot few-shot inputs.")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["batch_file", "custom_id"])
    frame["grammar_size"] = pd.to_numeric(
        frame["n_rules"], errors="coerce"
    ) + pd.to_numeric(frame["n_words"], errors="coerce")
    frame["sequence_length"] = (
        frame["input_sentence"]
        .fillna("")
        .map(lambda sentence: len(str(sentence).split()))
    )
    frame["k_shots"] = pd.to_numeric(frame["k_shots"], errors="coerce")
    frame["exact_match"] = (
        frame["model_answer"].notna()
        & frame["output_sentence"].notna()
        & frame["model_answer"].eq(frame["output_sentence"])
    )
    required = ["fuzzy_model", "grammar_size", "sequence_length", "k_shots"]
    valid_rows = frame.dropna(subset=required).copy()
    if valid_rows.empty:
        raise ValueError("No joined output records include complete plotting metadata.")
    valid_rows["k_shots"] = valid_rows["k_shots"].astype(int)
    valid_rows.attrs["ignored_output_ids"] = ignored_output_ids
    return valid_rows


def shot_palette(k_values: list[int]) -> dict[int, Any]:
    """Give zero-shot a neutral color and order positive shots by magnitude."""

    palette: dict[int, Any] = {}
    if 0 in k_values:
        palette[0] = aes.sns.color_palette("Greys", n_colors=5)[3]
    positive = [k for k in k_values if k > 0]
    colors = aes.sns.color_palette("viridis", n_colors=len(positive) + 2)[1:-1]
    palette.update(dict(zip(positive, colors, strict=True)))
    return palette


def add_sentence_length_bins(rows: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Add the quintile-midpoint sentence-length axis used in related figures."""

    rows = rows.copy()
    intervals = pd.qcut(rows["sequence_length"], q=n_bins, duplicates="drop")
    rows["input_length_bin"] = intervals.map(
        lambda interval: (interval.left + interval.right) / 2
        if pd.notna(interval)
        else float("nan")
    )
    return rows


def plot_rows(rows: pd.DataFrame):
    """Plot k-shot accuracy by grammar size and sentence-length quintile."""

    models = rows["fuzzy_model"].dropna().unique()
    if len(models) != 1:
        raise ValueError(
            "Expected results for exactly one model in k-shot inputs; found "
            + ", ".join(sorted(models))
        )
    plot_data = add_sentence_length_bins(rows)
    k_values = sorted(rows["k_shots"].unique())
    palette = shot_palette(k_values)
    figure = aes.plt.figure(
        figsize=(aes.COLM_PAPER_WIDTH_IN, aes.FIG_HEIGHT_SINGLE_ROW_IN)
    )
    grid = figure.add_gridspec(1, 2, wspace=0.1)
    size_axis = figure.add_subplot(grid[0, 0])
    length_axis = figure.add_subplot(grid[0, 1], sharey=size_axis)

    for axis, x_column in (
        (size_axis, "grammar_size"),
        (length_axis, "input_length_bin"),
    ):
        aes.sns.lineplot(
            data=plot_data,
            x=x_column,
            y="exact_match",
            hue="k_shots",
            style="k_shots",
            hue_order=k_values,
            style_order=k_values,
            palette=palette,
            estimator="mean",
            errorbar=("ci", 95),
            n_boot=1_000,
            seed=42,
            markers=True,
            markersize=5,
            legend=True,
            ax=axis,
        )
        axis.set_ylim(-0.05, 1.05)
        axis.yaxis.set_major_formatter(aes.PCT_FORMATTER)
        axis.grid(axis="y", linestyle="--", alpha=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    size_axis.set_xlabel("Grammar size")
    size_axis.set_ylabel("Accuracy")
    size_axis.set_xlim(45, 10_500)
    size_axis.set_xscale("symlog")
    size_axis.xaxis.set_major_formatter(aes.KMB_FORMATTER)
    size_axis.xaxis.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=2, subs=tuple(range(2, 10)))
    )
    size_axis.tick_params(axis="x", which="minor", length=2.5)

    length_axis.set_xlabel("String length")
    length_axis.set_ylabel("")
    length_axis.set_xlim(5, 35)
    length_axis.set_xticks([10, 15, 20, 25, 30, 35])
    length_axis.tick_params(
        axis="both", which="both", labelbottom=True, labelleft=False
    )

    left_legend = size_axis.get_legend()
    if left_legend is not None:
        left_legend.remove()
    right_legend = length_axis.get_legend()
    if right_legend is not None:
        handles = right_legend.legend_handles
        right_legend.remove()
        figure.legend(
            handles=handles,
            labels=[f"{k} shot" if k == 1 else f"{k} shots" for k in k_values],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.9),
            ncols=3,
            frameon=False,
            columnspacing=1.6,
            handletextpad=0.6,
        )
    return figure, plot_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate few-shot accuracy curves by grammar and sequence size."
    )
    parser.add_argument("--batch-dir", type=Path, default=BATCH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.batch_dir)
    figure, plot_data = plot_rows(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(args.summary_output, index=False)
    output_path, _ = aes.save_figure(args.output, fig=figure)
    aes.plt.close(figure)

    models = ", ".join(aes.ordered_models(plot_data["fuzzy_model"].unique()))
    shots = ", ".join(str(k) for k in sorted(rows["k_shots"].unique()))
    print(f"Saved {output_path}")
    print(f"Saved {args.summary_output}")
    print(f"Rows: {len(rows):,}; models: {models}; shot counts: {shots}")
    ignored_output_ids = rows.attrs.get("ignored_output_ids", 0)
    if ignored_output_ids:
        print(f"Ignored {ignored_output_ids:,} outputs not from k-shot input files")


if __name__ == "__main__":
    main()
