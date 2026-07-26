"""Generate the GPT-5 error-taxonomy figure used in the paper.

Run ``uv run python scripts/figures/generate_gpt5_error_taxonomy_by_experiment.py``
after generating ``notebooks/cache/error-analysis/rows.csv`` with
``notebooks/error_analysis.py``.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrootutils
import seaborn as sns
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import aesthetics as aes  # noqa: E402

# Figure configuration. Change these values to write a variant without editing
# the plotting or data-preparation code below.
OUTPUT_DIR = PROJECT_ROOT / "paper" / "figures"
OUTPUT_FILENAME = "gpt-5_error_taxonomy_by_experiment"
ROWS_PATH = PROJECT_ROOT / "notebooks" / "cache" / "error-analysis" / "rows.csv"
MODEL = "gpt-5"
EXPERIMENT_ORDER = ["size", "wordorder", "agreement", "orthography"]
WORD_ORDER_DATASET = "wordorder_exp"
ORTHOGRAPHY_DATASET = "orthography_exp"

FAMILY_ORDER = [
    "word_order_error",
    "omission",
    "extra_words",
    "recall_error",
    "source_vocab_error",
    "english_vocab",
    "hallucinated_vocab",
    "orthography_error",
    "mixed_other",
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


def family_share(df: pd.DataFrame) -> pd.DataFrame:
    """Return each error family's share among wrong answers per experiment."""
    counts = (
        df.groupby(["exp", "error_family"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        df.groupby("exp", observed=False).size().rename("total").reset_index()
    )
    shares = counts.merge(totals, on="exp", how="left")
    shares["share"] = shares["count"] / shares["total"]
    return shares


def parse_error_tags(value: str | float | None) -> list[str]:
    """Parse the pipe-delimited tags produced by the error-analysis script."""
    if not isinstance(value, str) or not value.strip():
        return []
    return [
        tag.strip()
        for tag in value.split("|")
        if tag.strip() and tag.strip() != "exact_match"
    ]


def load_plot_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load GPT-5 wrong-answer tags and calculate per-experiment shares."""
    if not ROWS_PATH.exists():
        raise FileNotFoundError(
            f"Missing error-analysis rows at {ROWS_PATH}. Generate them with "
            "`uv run python notebooks/error_analysis.py` first."
        )

    rows = pd.read_csv(ROWS_PATH)
    required_columns = {"exp", "fuzzy_model", "exact_match", "failure_tags_str"}
    missing_columns = required_columns - set(rows.columns)
    if missing_columns:
        raise ValueError(
            "The error-analysis cache is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    if "dataset" not in rows.columns:
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
    rows = rows.loc[
        (
            rows["exp"].astype(str).ne("wordorder")
            | rows["dataset"].eq(WORD_ORDER_DATASET)
        )
        & (
            rows["exp"].astype(str).ne("orthography")
            | rows["dataset"].eq(ORTHOGRAPHY_DATASET)
        )
    ].copy()
    rows["exp"] = pd.Categorical(
        rows["exp"], categories=EXPERIMENT_ORDER, ordered=True
    )
    wrong_tags = rows.loc[
        rows["fuzzy_model"].astype(str).eq(MODEL) & ~rows["exact_match"]
    ].copy()
    wrong_tags["error_family"] = wrong_tags["failure_tags_str"].map(parse_error_tags)
    wrong_tags = wrong_tags.explode("error_family")
    wrong_tags = wrong_tags.loc[
        wrong_tags["error_family"].notna() & wrong_tags["error_family"].ne("")
    ].copy()

    plot_data = family_share(wrong_tags)
    present_families = [
        family for family in FAMILY_ORDER if family in set(plot_data["error_family"])
    ]
    if not present_families:
        raise ValueError(f"No non-exact error tags found for model {MODEL!r}.")

    plot_data["error_family"] = pd.Categorical(
        plot_data["error_family"], categories=present_families, ordered=True
    )
    experiment_panels = [
        experiment
        for experiment in EXPERIMENT_ORDER
        if (plot_data["exp"].astype(str) == experiment).any()
    ]
    return plot_data, present_families, experiment_panels


def make_figure() -> plt.Figure:
    """Create the error-taxonomy panel figure."""
    plot_data, present_families, experiment_panels = load_plot_data()
    figure = plt.figure(
        figsize=(aes.COLM_PAPER_WIDTH_IN, aes.FIG_HEIGHT_SINGLE_ROW_IN)
    )
    grid = GridSpec(1, 4, figure=figure, wspace=0.12)
    axes = [figure.add_subplot(grid[0, 0])]
    for panel_idx in range(1, 4):
        axes.append(figure.add_subplot(grid[0, panel_idx], sharey=axes[0]))

    y_positions = np.arange(len(present_families))
    for panel_idx, experiment in enumerate(experiment_panels):
        axis = axes[panel_idx]
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
        axis.set_title(EXPERIMENT_LABELS[experiment], loc="left", fontweight="normal")
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
        if panel_idx == 0:
            axis.set_yticklabels(
                [FAMILY_LABELS[family] for family in experiment_data["error_family"]]
            )
            axis.set_xlabel("Share of wrong answers")
        else:
            axis.tick_params(axis="y", labelleft=False)
            axis.set_xlabel("")
        sns.despine(ax=axis, left=False, bottom=False)

    for panel_idx in range(len(experiment_panels), len(axes)):
        axes[panel_idx].set_visible(False)

    figure.subplots_adjust(left=0, bottom=0, right=1, top=1)
    return figure


def main() -> None:
    """Generate and save the configured figure."""
    figure = make_figure()
    output_path, _ = aes.save_figure(OUTPUT_DIR / OUTPUT_FILENAME, fig=figure)
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
