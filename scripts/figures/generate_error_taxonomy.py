"""Generate the GPT-5 error-taxonomy figure used in the paper.

Run ``uv run python scripts/figures/generate_error_taxonomy.py`` to plot the
checked-in summary data. Pass ``--recompute`` after generating
``notebooks/cache/error-analysis/rows.csv`` with ``notebooks/error_analysis.py``
to refresh that summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyrootutils
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks import aesthetics as aes  # noqa: E402

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "paper" / "figures" / "gpt-5_error_taxonomy_by_experiment"
)
DEFAULT_ROWS_PATH = PROJECT_ROOT / "notebooks" / "cache" / "error-analysis" / "rows.csv"
DEFAULT_PLOT_DATA_PATH = Path(__file__).with_name("error_taxonomy_plot_data.csv")
DEFAULT_MODEL = "gpt-5"
FONT_SCALE = 0.8
EXPERIMENT_ORDER = ["size", "wordorder", "agreement", "orthography"]
WORD_ORDER_DATASET = "wordorder_exp"
ORTHOGRAPHY_DATASET = "orthography_exp"

FIGURE_WIDTH_IN = aes.ACL_PAPER_WIDTH_IN
# Reserve fixed room for titles and x-axis labels, then add space for each row.
FIGURE_HEIGHT_FIXED_IN = 0.65
FIGURE_HEIGHT_PER_CATEGORY_IN = 0.1

FAMILY_ORDER = [
    "word_order_error",
    "omission",
    "extra_words",
    "recall_error",
    "source_vocab_error",
    "hallucinated_vocab",
    "orthography_error",
]
FAMILY_PALETTE = {
    "word_order_error": "#5B8FF9",
    "omission": "#F4A261",
    "extra_words": "#C77DFF",
    "recall_error": "#2A9D8F",
    "source_vocab_error": "#52B788",
    "english_vocab": "#E9C46A",
    "hallucinated_vocab": "#E76F51",
    "orthography_error": "#7AA6E8",
    "mixed_other": "#ADB5BD",
}
FAMILY_LABELS = {
    "word_order_error": "word ordering",
    "omission": "omission",
    "extra_words": "extra words",
    "recall_error": "recall",
    "source_vocab_error": "source vocab",
    "english_vocab": "english vocab",
    "hallucinated_vocab": "hallucination",
    "orthography_error": "orthography",
    "mixed_other": "other",
}
EXPERIMENT_LABELS = {
    "wordorder": "Word Order",
    "size": "Size",
    "agreement": "Agreement",
    "orthography": "Orthography",
}


def parse_error_tags(value: str | float | None) -> list[str]:
    """Parse the pipe-delimited tags produced by the error-analysis script."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [
        tag.strip()
        for tag in value.split("|")
        if tag.strip() and tag.strip() != "exact_match"
    ]


def filter_requested_datasets(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep the canonical Word Order and Orthography datasets."""
    if "dataset" not in rows.columns:
        rows = rows.copy()
        rows["dataset"] = (
            rows["exp"]
            .map(
                {
                    "wordorder": WORD_ORDER_DATASET,
                    "size": "size_exp",
                    "agreement": "agreement_exp",
                    "orthography": ORTHOGRAPHY_DATASET,
                }
            )
            .fillna("unknown")
        )
    return rows.loc[
        (
            rows["exp"].astype(str).ne("wordorder")
            | rows["dataset"].eq(WORD_ORDER_DATASET)
        )
        & (
            rows["exp"].astype(str).ne("orthography")
            | rows["dataset"].eq(ORTHOGRAPHY_DATASET)
        )
    ].copy()


def compute_plot_data(
    rows_path: Path, model: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load a model's wrong-answer error-family shares by experiment."""
    if not rows_path.exists():
        raise FileNotFoundError(
            f"Missing error-analysis rows at {rows_path}. Generate them with "
            "`uv run python notebooks/error_analysis.py` first."
        )

    rows = pd.read_csv(rows_path, low_memory=False)
    required_columns = {"exp", "fuzzy_model", "exact_match", "failure_tags_str"}
    missing_columns = required_columns - set(rows.columns)
    if missing_columns:
        raise ValueError(
            "The error-analysis cache is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    rows = filter_requested_datasets(rows)
    wrong_tags = rows.loc[
        rows["fuzzy_model"].astype(str).eq(model) & ~rows["exact_match"]
    ].copy()
    wrong_tags["error_family"] = wrong_tags["failure_tags_str"].map(parse_error_tags)
    wrong_tags = wrong_tags.explode("error_family")
    wrong_tags = wrong_tags.loc[
        wrong_tags["error_family"].notna() & wrong_tags["error_family"].ne("")
    ].copy()
    if wrong_tags.empty:
        raise ValueError(f"No non-exact error tags found for model {model!r}.")

    counts = wrong_tags.groupby(["exp", "error_family"], as_index=False).size()
    counts = counts.rename(columns={"size": "count"})
    totals = wrong_tags.groupby("exp", as_index=False).size()
    totals = totals.rename(columns={"size": "total"})
    plot_data = counts.merge(totals, on="exp", how="left")
    plot_data["share"] = plot_data["count"] / plot_data["total"]

    present_families = [
        family for family in FAMILY_ORDER if family in set(plot_data["error_family"])
    ]
    plot_data = plot_data.loc[plot_data["error_family"].isin(present_families)].copy()
    plot_data["error_family"] = pd.Categorical(
        plot_data["error_family"], categories=present_families, ordered=True
    )
    experiment_panels = [
        experiment
        for experiment in EXPERIMENT_ORDER
        if (plot_data["exp"].astype(str) == experiment).any()
    ]
    return plot_data, present_families, experiment_panels


def load_cached_plot_data(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the compact, version-controlled dataframe used for plotting."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached plot data at {path}. Run this script with "
            "`--recompute` to create it."
        )
    plot_data = pd.read_csv(path)
    required_columns = {"exp", "error_family", "count", "total", "share"}
    missing_columns = required_columns - set(plot_data.columns)
    if missing_columns:
        raise ValueError(
            f"Cached plot data is missing columns: {', '.join(sorted(missing_columns))}"
        )
    present_families = [
        family for family in FAMILY_ORDER if family in set(plot_data["error_family"])
    ]
    experiment_panels = [
        experiment
        for experiment in EXPERIMENT_ORDER
        if (plot_data["exp"].astype(str) == experiment).any()
    ]
    return plot_data, present_families, experiment_panels


def make_figure(
    plot_data: pd.DataFrame,
    present_families: list[str],
    experiment_panels: list[str],
):
    """Create the compact, four-panel error-taxonomy chart."""
    figure_height_in = (
        FIGURE_HEIGHT_FIXED_IN + len(present_families) * FIGURE_HEIGHT_PER_CATEGORY_IN
    )
    figure = aes.plt.figure(
        figsize=(FIGURE_WIDTH_IN, figure_height_in), layout="constrained"
    )
    grid = GridSpec(1, 4, figure=figure, wspace=0.12)
    axes = [figure.add_subplot(grid[0, 0])]
    axes.extend(
        figure.add_subplot(grid[0, index], sharey=axes[0]) for index in range(1, 4)
    )

    y_positions = np.arange(len(present_families))
    for panel_index, experiment in enumerate(experiment_panels):
        axis = axes[panel_index]
        experiment_data = plot_data.loc[
            plot_data["exp"].astype(str).eq(experiment)
        ].copy()
        experiment_data = (
            experiment_data.set_index("error_family")["share"]
            .reindex(present_families, fill_value=0.0)
            .rename("share")
            .reset_index()
        )

        axis.barh(
            y_positions,
            experiment_data["share"],
            height=0.72,
            color=[
                FAMILY_PALETTE[family] for family in experiment_data["error_family"]
            ],
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_title(EXPERIMENT_LABELS[experiment], fontweight="normal")
        axis.set_xlim(0, 0.5)
        axis.set_xticks([0, 0.5])
        axis.xaxis.set_major_formatter(aes.PCT_FORMATTER)
        axis.get_xticklabels()[0].set_horizontalalignment("left")
        axis.get_xticklabels()[-1].set_horizontalalignment("right")
        axis.set_ylim(-0.5, len(present_families) - 0.5)
        axis.set_yticks(y_positions)
        axis.invert_yaxis()
        axis.set_ylabel("")
        axis.tick_params(axis="y", pad=4)
        if panel_index == 0:
            axis.set_yticklabels(
                [FAMILY_LABELS[family] for family in experiment_data["error_family"]]
            )
            axis.set_xlabel("Share of wrong answers")
        else:
            axis.tick_params(axis="y", labelleft=False)
            axis.set_xlabel("")
        aes.sns.despine(ax=axis, left=False, bottom=False)

    for panel_index in range(len(experiment_panels), len(axes)):
        axes[panel_index].set_visible(False)

    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the error-taxonomy figure.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-path", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument(
        "--plot-data-path",
        type=Path,
        default=DEFAULT_PLOT_DATA_PATH,
        help="Cached dataframe used to render the figure (default: %(default)s).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Rebuild the cached plotting dataframe from --rows-path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recompute:
        plot_data, present_families, experiment_panels = compute_plot_data(
            args.rows_path, args.model
        )
        args.plot_data_path.parent.mkdir(parents=True, exist_ok=True)
        plot_data.to_csv(args.plot_data_path, index=False)
        print(f"Saved {args.plot_data_path}")
    else:
        plot_data, present_families, experiment_panels = load_cached_plot_data(
            args.plot_data_path
        )
    with aes.sns.plotting_context("paper", font_scale=FONT_SCALE):
        figure = make_figure(plot_data, present_families, experiment_panels)
        output_path, _ = aes.save_figure(args.output, fig=figure, tight=False)
    aes.plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
