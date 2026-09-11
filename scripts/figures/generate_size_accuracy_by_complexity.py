"""Generate the size-experiment exact-match accuracy figure.

The script discovers OpenAI-style batch inputs and outputs under
``batches/size_exp``. Outputs are joined to inputs by ``custom_id``, so new
model runs require no manual batch-ID-to-filename mapping.

Example:
    uv run python scripts/figures/generate_size_accuracy_by_complexity.py
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

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks import aesthetics as aes  # noqa: E402

BATCH_DIR = PROJECT_ROOT / "batches" / "size_exp"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache" / "size-accuracy"
DEFAULT_OUTPUT = FIGURES_DIR / "size_accuracy_by_complexity"
DEFAULT_SUMMARY_OUTPUT = CACHE_DIR / "exact_match_rows.csv"
ANSWER_RE = re.compile(
    r"final\s*answer\s*(?::|-|—)?\s*(?:is\s*)?([^\n]+)",
    re.IGNORECASE | re.DOTALL,
)
GPT_5_6_MODELS = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
MODEL_FILTER = GPT_5_6_MODELS


def fuzzy_model(model: str | None) -> str:
    """Collapse dated provider model names to their stable model family."""

    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model or "")


def extract_answer(model_response: str | None) -> str | None:
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
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else None


def response_model(record: dict[str, Any]) -> str | None:
    body = (record.get("response") or {}).get("body") or {}
    model = body.get("model")
    return model if isinstance(model, str) else None


def load_inputs(batch_dir: Path) -> dict[str, dict[str, Any]]:
    """Load evaluation metadata keyed by the custom ID sent to the provider."""

    inputs: dict[str, dict[str, Any]] = {}
    for path in sorted(batch_dir.glob("inputs_*.jsonl")):
        if path.name.endswith("_output.jsonl"):
            continue
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
                }
    if not inputs:
        raise FileNotFoundError(f"No input JSONL files found under {batch_dir}")
    return inputs


def load_rows(batch_dir: Path) -> pd.DataFrame:
    """Join all size-experiment output records to their input metadata."""

    inputs = load_inputs(batch_dir)
    rows: list[dict[str, Any]] = []
    unmatched_ids: set[str] = set()
    output_paths = sorted(batch_dir.glob("*_output.jsonl"))
    if not output_paths:
        raise FileNotFoundError(f"No output JSONL files found under {batch_dir}")

    for path in output_paths:
        with path.open() as handle:
            for line in handle:
                output = json.loads(line)
                custom_id = output.get("custom_id")
                if not isinstance(custom_id, str):
                    continue
                input_metadata = inputs.get(custom_id)
                if input_metadata is None:
                    unmatched_ids.add(custom_id)
                    continue
                model = response_model(output) or input_metadata["model"]
                rows.append(
                    {
                        "batch_file": path.name,
                        "custom_id": custom_id,
                        "model": model,
                        "fuzzy_model": fuzzy_model(str(model)),
                        "model_answer": extract_answer(response_content(output)),
                        **input_metadata,
                    }
                )

    if unmatched_ids:
        preview = ", ".join(sorted(unmatched_ids)[:3])
        raise ValueError(
            f"{len(unmatched_ids)} output IDs had no matching input records. "
            f"Examples: {preview}"
        )
    if not rows:
        raise ValueError("No output records could be joined to size-experiment inputs.")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["batch_file", "custom_id"])
    frame["size"] = pd.to_numeric(frame["n_rules"], errors="coerce") + pd.to_numeric(
        frame["n_words"], errors="coerce"
    )
    frame["input_words"] = (
        frame["input_sentence"].fillna("").map(lambda text: len(str(text).split()))
    )
    frame["exact_match"] = (
        frame["model_answer"].notna()
        & frame["output_sentence"].notna()
        & frame["model_answer"].eq(frame["output_sentence"])
    )
    required_columns = ["fuzzy_model", "size", "input_words"]
    missing_metadata = frame[required_columns].isna().any(axis=1)
    excluded_by_model = (
        frame.loc[missing_metadata]
        .groupby("fuzzy_model", dropna=False)
        .size()
        .to_dict()
    )
    valid_rows = frame.loc[~missing_metadata].copy()
    if valid_rows.empty:
        raise ValueError("No joined output records include complete plotting metadata.")
    valid_rows.attrs["excluded_by_model"] = excluded_by_model
    return valid_rows


def add_sentence_length_bins(rows: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Add globally consistent sentence-length quintile midpoints."""

    rows = rows.copy()
    intervals = pd.qcut(rows["input_words"], q=n_bins, duplicates="drop")
    rows["input_length_bin"] = intervals.map(
        lambda interval: (interval.left + interval.right) / 2
        if pd.notna(interval)
        else float("nan")
    )
    return rows


def model_palette(models: list[str]) -> dict[str, Any]:
    """Use shared colors for established models and stable fallbacks for new ones."""

    palette = {
        model: aes.PALETTE_MODELS[model]
        for model in models
        if model in aes.PALETTE_MODELS
    }
    gpt_5_6_palette = aes.set_model_palette(
        "GPT-5.6", "Reds", list(reversed(GPT_5_6_MODELS))
    )
    palette.update(
        {model: gpt_5_6_palette[model] for model in models if model in gpt_5_6_palette}
    )
    fallback_colors = aes.sns.color_palette("colorblind", n_colors=len(models) + 2)
    fallback_index = 0
    for model in models:
        if model not in palette:
            palette[model] = fallback_colors[fallback_index]
            fallback_index += 1
    return palette


def plot_rows(rows: pd.DataFrame):
    """Create the grammar-size and sentence-length exact-match panels."""

    plot_data = add_sentence_length_bins(rows)
    models = aes.ordered_models(plot_data["fuzzy_model"].dropna().unique())
    palette = model_palette(models)

    figure = aes.plt.figure(
        figsize=(aes.COLM_PAPER_WIDTH_IN, aes.FIG_HEIGHT_SINGLE_ROW_IN * 1.45)
    )
    grid = figure.add_gridspec(1, 2, wspace=0.18)
    size_axis = figure.add_subplot(grid[0, 0])
    length_axis = figure.add_subplot(grid[0, 1])

    for axis, x_column in [
        (size_axis, "size"),
        (length_axis, "input_length_bin"),
    ]:
        aes.sns.lineplot(
            data=plot_data,
            x=x_column,
            y="exact_match",
            hue="fuzzy_model",
            style="fuzzy_model",
            hue_order=models,
            style_order=models,
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
    size_axis.set_ylabel("Exact match")
    size_axis.xaxis.set_major_formatter(aes.NICE_FORMATTER)
    size_axis.set_xticks([0, 4_000, 8_000])

    length_axis.set_xlabel("Sentence length (quintile midpoint)")
    length_axis.set_ylabel("")
    length_axis.yaxis.set_ticklabels([])

    left_legend = size_axis.get_legend()
    if left_legend is not None:
        left_legend.remove()
    right_legend = length_axis.get_legend()
    if right_legend is not None:
        right_legend.set_title("")
        aes.sns.move_legend(
            length_axis,
            "upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            labels=[aes.MODEL_DISPLAY_NAMES.get(model, model) for model in models],
        )

    figure.subplots_adjust(left=0.11, bottom=0.3, right=0.79, top=0.96)
    return figure, plot_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the size-experiment accuracy-by-complexity figure."
    )
    parser.add_argument("--batch-dir", type=Path, default=BATCH_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.batch_dir)
    if MODEL_FILTER is not None:
        rows = rows.loc[rows["fuzzy_model"].isin(MODEL_FILTER)].copy()
        if rows.empty:
            raise ValueError(f"No rows found for MODEL_FILTER={MODEL_FILTER}")
    figure, plot_data = plot_rows(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(args.summary_output, index=False)
    output_path, _ = aes.save_figure(args.output, fig=figure)
    aes.plt.close(figure)

    models = ", ".join(aes.ordered_models(plot_data["fuzzy_model"].unique()))
    print(f"Saved {output_path}")
    print(f"Saved {args.summary_output}")
    print(f"Rows: {len(plot_data):,}; models: {models}")
    excluded_by_model = rows.attrs.get("excluded_by_model", {})
    if excluded_by_model:
        excluded = ", ".join(
            f"{model}: {count:,}" for model, count in sorted(excluded_by_model.items())
        )
        print(f"Excluded rows with incomplete input metadata: {excluded}")


if __name__ == "__main__":
    main()
