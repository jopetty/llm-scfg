"""Generate the combined experiment-overview figure.

The plotting dataframe is checked in beside this script. Pass ``--recompute``
to rebuild it from the current Hugging Face-backed batch results; otherwise the
script only loads that compact dataframe and renders the figure.

Example:
    uv run python scripts/figures/generate_combined_experiment_overview.py
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
from scfg.hf_data import load_hf_split, resolve_hf_dataset_repo  # noqa: E402
from scripts.figures.generate_fewshot_accuracy import (  # noqa: E402
    load_rows as load_fewshot_rows,
)
from scripts.figures.generate_fewshot_accuracy import (  # noqa: E402
    shot_palette,
)

FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache"
COMBINED_CACHE_DIR = CACHE_DIR / "combined-exp"
REVISED_CACHE_DIR = COMBINED_CACHE_DIR / "revised"
SIZE_CACHE_PATH = CACHE_DIR / "size-accuracy" / "exact_match_rows.csv"
BATCH_DIR = PROJECT_ROOT / "batches"
DEFAULT_OUTPUT = FIGURES_DIR / "combined_experiment_overview"
DEFAULT_PLOT_DATA_PATH = Path(__file__).with_name(
    "combined_experiment_overview_plot_data.csv"
)
FONT_SCALE = 0.8

GPT_5_6_MODELS = ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
ANSWER_RE = re.compile(
    r"final\s*answer\s*(?::|-|—)?\s*(?:is\s*)?([^\n]+)",
    re.IGNORECASE | re.DOTALL,
)

AGREEMENT_CONDITION_ORDER = [
    "NoAgr → NoAgr",
    "Agr → NoAgr",
    "Agr → Agr",
    "NoAgr → Agr",
]
ORTHOGRAPHY_ORDER = [
    "Latin → Latin",
    "Latin → Latin (diacritics)",
    "Latin → Cyrillic",
    "Latin → Hebrew",
    "Latin → Hebrew (pointed)",
]
WORD_ORDER_PLOT_ORDER = ["SVO", "SOV", "OVS"]
WORD_ORDER_PLOT_LABELS = {
    "SVO": "SVO → SVO",
    "SOV": "SVO → SOV",
    "OVS": "SVO → OVS",
}

PALETTE_AGREEMENT = aes.darken(
    {
        "NoAgr → NoAgr": aes.sns.color_palette("Dark2", n_colors=4)[0],
        "Agr → NoAgr": aes.sns.color_palette("Dark2", n_colors=4)[1],
        "Agr → Agr": aes.sns.color_palette("Dark2", n_colors=4)[2],
        "NoAgr → Agr": aes.sns.color_palette("Dark2", n_colors=4)[3],
    },
    by=0.25,
)
PALETTE_ORTHOGRAPHY = aes.darken(
    {
        "Latin → Latin": aes.sns.color_palette("Reds", n_colors=7)[1],
        "Latin → Latin (diacritics)": aes.sns.color_palette("Reds", n_colors=7)[3],
        "Latin → Cyrillic": aes.sns.color_palette("Greens", n_colors=7)[2],
        "Latin → Hebrew (pointed)": aes.sns.color_palette("Blues", n_colors=7)[3],
        "Latin → Hebrew": aes.sns.color_palette("Blues", n_colors=7)[5],
    },
    by=0.15,
)

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "slug": "size",
        "display_name": "Size",
        "size_x": "grammar_size_bin",
        "size_value": "size",
        "length_x": "input_length_bin",
        "length_value": "input_words",
        "hue": "model_display_name",
        "style": "model_display_name",
        "order": None,
        "palette": None,
        "legend_labels": None,
        "legend_cols": 2,
    },
    {
        "slug": "wordorder",
        "display_name": "Word order (gpt-5)",
        "size_x": "grammar_size_bin",
        "size_value": "size",
        "length_x": "input_length_bin",
        "length_value": "input_words",
        "hue": "target_word_order",
        "style": "target_word_order",
        "order": WORD_ORDER_PLOT_ORDER,
        "palette": aes.PALETTE_WORDORDER,
        "legend_labels": WORD_ORDER_PLOT_LABELS,
        "legend_cols": 2,
    },
    {
        "slug": "agreement",
        "display_name": "Morphology (gpt-5.6-luna)",
        "size_x": "grammar_size_bin",
        "size_value": "grammar_size",
        "length_x": "input_length_bin",
        "length_value": "input_length",
        "hue": "agreement_condition",
        "style": "agreement_condition",
        "order": AGREEMENT_CONDITION_ORDER,
        "palette": PALETTE_AGREEMENT,
        "legend_labels": None,
        "legend_cols": 2,
    },
    {
        "slug": "orthography",
        "display_name": "Orthography (gpt-5.6-luna)",
        "size_x": "grammar_size_bin",
        "size_value": "grammar_size",
        "length_x": "input_length_bin",
        "length_value": "input_length",
        "hue": "target_orthography",
        "style": "target_orthography",
        "order": ORTHOGRAPHY_ORDER,
        "palette": PALETTE_ORTHOGRAPHY,
        "legend_labels": None,
        "legend_cols": 1,
    },
    {
        "slug": "fewshot",
        "display_name": "Few-shot",
        "size_x": "grammar_size_bin",
        "size_value": "grammar_size",
        "length_x": "input_length_bin",
        "length_value": "sequence_length",
        "hue": "shot_label",
        "style": "shot_label",
        "order": ["0 shots", "1 shot", "2 shots", "4 shots", "8 shots", "16 shots"],
        "palette": None,
        "legend_labels": None,
        "legend_cols": 2,
    },
]

NEW_PANEL_PATHS = {
    slug: (
        REVISED_CACHE_DIR / f"{slug}_grammar_size.csv",
        REVISED_CACHE_DIR / f"{slug}_string_length.csv",
    )
    for slug in ("wordorder", "agreement", "orthography")
}


def ordered_model_names(models: pd.Series) -> list[str]:
    """Return a stable order that keeps the GPT-5.6 variants together."""

    seen = {str(model) for model in models.dropna()}
    gpt_5_6 = [model for model in GPT_5_6_MODELS if model in seen]
    remaining = aes.ordered_models(seen - set(GPT_5_6_MODELS))
    return remaining + gpt_5_6


def size_model_palette(models: list[str]) -> dict[str, Any]:
    """Use established model colors plus an ordered red family for GPT-5.6."""

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
    for index, model in enumerate(model for model in models if model not in palette):
        palette[model] = fallback_colors[index]
    return palette


def load_revised_size_panels() -> dict[str, pd.DataFrame]:
    """Adapt the current size-figure cache to the combined-figure schema."""

    if not SIZE_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Missing revised size cache: {SIZE_CACHE_PATH}. Run "
            "generate_size_accuracy_by_complexity.py first."
        )

    rows = pd.read_csv(SIZE_CACHE_PATH)
    required = {"fuzzy_model", "size", "input_length_bin", "exact_match"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Revised size cache is missing columns: {', '.join(missing)}")

    rows["model_display_name"] = rows["fuzzy_model"].map(
        lambda model: aes.MODEL_DISPLAY_NAMES.get(model, model)
    )
    rows["match_value"] = rows["exact_match"].astype(float)
    return {"size": rows.copy(), "length": rows.copy()}


def load_revised_fewshot_panels() -> dict[str, pd.DataFrame]:
    """Adapt the dedicated few-shot batch analysis to the overview schema."""

    rows = load_fewshot_rows(BATCH_DIR / "fewshot_exp")
    rows["match_value"] = rows["exact_match"].astype(float)
    rows["shot_label"] = rows["k_shots"].map(
        lambda k: f"{k} shot" if k == 1 else f"{k} shots"
    )
    return {"size": rows.copy(), "length": rows.copy()}


def extract_answer(model_response: str | None) -> str | None:
    """Extract and normalize a model's requested final answer."""

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


def word_order_label(grammar: dict[str, Any]) -> str:
    """Return the target-order label used by the paper-facing plot."""

    share_head = grammar["a"]["head_initial"] == grammar["b"]["head_initial"]
    share_spec = grammar["a"]["spec_initial"] == grammar["b"]["spec_initial"]
    if share_head and share_spec:
        return "SVO"
    if not share_head and share_spec:
        return "SOV"
    if not share_head and not share_spec:
        return "OVS"
    return "VOS"


def condition_label(slug: str, grammar: dict[str, Any]) -> str:
    """Derive the paper-facing condition label from HF grammar metadata."""

    if slug == "wordorder":
        return word_order_label(grammar)
    if slug == "agreement":
        metadata = grammar["agreement_metadata"]
        source = "Agr" if metadata["a"]["config"]["enabled"] else "NoAgr"
        target = "Agr" if metadata["b"]["config"]["enabled"] else "NoAgr"
        return f"{source} → {target}"
    if slug == "orthography":
        orthography = grammar["b"]["orthography"]
        labels = {
            "latin": "Latin → Latin",
            "latin_diacritic": "Latin → Latin (diacritics)",
            "cyrillic": "Latin → Cyrillic",
            "hebrew": "Latin → Hebrew (pointed)",
            "yiddish": "Latin → Hebrew (pointed)",
            "hebrew_unpointed": "Latin → Hebrew",
        }
        return labels[orthography]
    raise ValueError(f"No condition-label mapping for experiment: {slug}")


def condition_column(slug: str) -> str:
    return {
        "wordorder": "target_word_order",
        "agreement": "agreement_condition",
        "orthography": "target_orthography",
    }[slug]


def load_revised_experiment_panels(slug: str) -> dict[str, pd.DataFrame]:
    """Build revised condition panels directly from batches and HF grammars."""

    batch_dir = BATCH_DIR / f"{slug}_exp"
    input_metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(batch_dir.glob("inputs_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                custom_id = record.get("custom_id")
                body = record.get("body") or {}
                metadata = body.get("metadata") or {}
                if not isinstance(custom_id, str) or not isinstance(metadata, dict):
                    continue
                if custom_id in input_metadata:
                    raise ValueError(f"Duplicate {slug} input custom ID: {custom_id}")
                input_metadata[custom_id] = {
                    "grammar_name": metadata.get("grammar_name"),
                    "input_sentence": metadata.get("input_sentence"),
                    "output_sentence": metadata.get("output_sentence"),
                    "n_rules": metadata.get("n_rules"),
                    "n_words": metadata.get("n_words"),
                }
    if not input_metadata:
        raise FileNotFoundError(f"No {slug} input JSONL files found under {batch_dir}")

    grammars = {
        str(grammar["name"]): grammar
        for grammar in load_hf_split(
            resolve_hf_dataset_repo(), f"{slug}_exp", "grammars"
        )
    }
    rows: list[dict[str, Any]] = []
    unmatched_ids: set[str] = set()
    for path in sorted(batch_dir.glob("*_output.jsonl")):
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                custom_id = record.get("custom_id")
                if not isinstance(custom_id, str):
                    continue
                metadata = input_metadata.get(custom_id)
                if metadata is None:
                    unmatched_ids.add(custom_id)
                    continue
                body = (record.get("response") or {}).get("body") or {}
                rows.append(
                    {
                        "custom_id": custom_id,
                        "fuzzy_model": re.sub(
                            r"-\d{4}-\d{2}-\d{2}$", "", str(body.get("model") or "")
                        ),
                        "model_answer": extract_answer(response_content(record)),
                        **metadata,
                    }
                )
    if unmatched_ids:
        preview = ", ".join(sorted(unmatched_ids)[:3])
        raise ValueError(
            f"{len(unmatched_ids)} {slug} output IDs had no matching inputs. "
            f"Examples: {preview}"
        )
    if not rows:
        raise ValueError(f"No {slug} output records could be joined to inputs.")

    panel = pd.DataFrame(rows).drop_duplicates(subset=["custom_id"])
    panel["input_words"] = (
        panel["input_sentence"].fillna("").map(lambda text: len(str(text).split()))
    )
    labels = {
        name: condition_label(slug, grammar) for name, grammar in grammars.items()
    }
    panel[condition_column(slug)] = panel["grammar_name"].map(labels)
    if slug == "agreement":
        panel["grammar_size"] = panel["grammar_name"].map(
            lambda name: 5 * len(grammars[str(name)]["a"]["verbs"])
            if name in grammars
            else float("nan")
        )
    if slug in {"agreement", "orthography"}:
        panel["input_length"] = panel["input_words"]
    if slug != "agreement":
        panel["size"] = pd.to_numeric(
            panel["n_rules"], errors="coerce"
        ) + pd.to_numeric(panel["n_words"], errors="coerce")
    if slug == "orthography":
        panel["grammar_size"] = panel["size"]
    panel["match_value"] = (
        panel["model_answer"].notna()
        & panel["output_sentence"].notna()
        & panel["model_answer"].eq(panel["output_sentence"])
    ).astype(float)
    size_column = "grammar_size" if slug == "agreement" else "size"
    required = [size_column, "input_words", condition_column(slug)]
    panel = panel.dropna(subset=required)
    orders = {
        "wordorder": WORD_ORDER_PLOT_ORDER,
        "agreement": AGREEMENT_CONDITION_ORDER,
        "orthography": ORTHOGRAPHY_ORDER,
    }
    panel = panel.loc[panel[condition_column(slug)].isin(orders[slug])].copy()
    if panel.empty:
        raise ValueError(
            f"No complete {slug} records remain for the plotted conditions."
        )

    REVISED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path in NEW_PANEL_PATHS[slug]:
        panel.to_csv(path, index=False)
    return {"size": panel.copy(), "length": panel.copy()}


def load_experiment_panels(slug: str) -> dict[str, pd.DataFrame]:
    """Load the revised source data for one overview row."""

    if slug == "size":
        return load_revised_size_panels()
    if slug == "fewshot":
        return load_revised_fewshot_panels()
    return load_revised_experiment_panels(slug)


def add_length_quintile_bins(panel: pd.DataFrame, length_column: str) -> pd.DataFrame:
    """Assign sentences to global length quintiles using their midpoints."""

    if length_column not in panel:
        raise ValueError(f"Panel is missing its input-length column: {length_column}")
    panel = panel.copy()
    values = pd.to_numeric(panel[length_column], errors="coerce")
    intervals = pd.qcut(values, q=5, duplicates="drop")
    panel["input_length_bin"] = intervals.map(
        lambda interval: (interval.left + interval.right) / 2
        if pd.notna(interval)
        else float("nan")
    )
    return panel


def add_grammar_size_quintile_bins(
    panel: pd.DataFrame, size_column: str
) -> pd.DataFrame:
    """Assign grammar sizes to quintile bins represented by numeric midpoints."""

    if size_column not in panel:
        raise ValueError(f"Panel is missing its grammar-size column: {size_column}")
    panel = panel.copy()
    values = pd.to_numeric(panel[size_column], errors="coerce")
    intervals = pd.qcut(values, q=5, duplicates="drop")
    panel["grammar_size_bin"] = intervals.map(
        lambda interval: (interval.left + interval.right) / 2
        if pd.notna(interval)
        else float("nan")
    )
    return panel


def configure_size_plot(
    config: dict[str, Any], panels: dict[str, pd.DataFrame]
) -> None:
    models = ordered_model_names(panels["size"]["fuzzy_model"])
    config["order"] = [aes.MODEL_DISPLAY_NAMES.get(model, model) for model in models]
    config["palette"] = {
        aes.MODEL_DISPLAY_NAMES.get(model, model): color
        for model, color in size_model_palette(models).items()
    }


def configure_wordorder_plot(
    config: dict[str, Any], panels: dict[str, pd.DataFrame]
) -> None:
    """Identify the model when the revised row contains a single run."""

    models = sorted(panels["size"]["fuzzy_model"].dropna().unique())
    if len(models) == 1:
        config["display_name"] = f"Word order ({models[0]})"


def configure_fewshot_plot(
    config: dict[str, Any], panels: dict[str, pd.DataFrame]
) -> None:
    """Set a shot-count palette and identify the single few-shot model."""

    models = sorted(panels["size"]["fuzzy_model"].dropna().unique())
    if len(models) != 1:
        raise ValueError(
            "Expected exactly one model in few-shot results; found " + ", ".join(models)
        )
    config["display_name"] = f"Few-shot ({models[0]})"
    config["palette"] = {
        f"{k} shot" if k == 1 else f"{k} shots": color
        for k, color in shot_palette(sorted(panels["size"]["k_shots"].unique())).items()
    }


def format_row_legend(axis, config: dict[str, Any]) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    if config["order"] is not None:
        label_to_handle = dict(zip(labels, handles, strict=True))
        labels = [label for label in config["order"] if label in label_to_handle]
        handles = [label_to_handle[label] for label in labels]
    if config["legend_labels"] is not None:
        labels = [config["legend_labels"].get(label, label) for label in labels]
    if config["slug"] == "orthography":
        format_orthography_legend(axis, config, handles, labels)
        return
    legend = axis.legend(
        handles,
        labels,
        title=config["display_name"],
        bbox_to_anchor=(1.04, 1.1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        alignment="left",
        columnspacing=0.8,
        markerscale=0.75,
        handlelength=1.5,
        # fontsize=8,
        # title_fontproperties={"size": 9},
        ncol=config["legend_cols"],
    )
    legend.get_title().set_fontweight("bold")


def format_orthography_legend(axis, config: dict[str, Any], handles, labels) -> None:
    """Place concise orthography conditions in columns and long ones on rows."""

    label_to_handle = dict(zip(labels, handles, strict=True))
    short_labels = ["Latin → Latin", "Latin → Hebrew"]
    long_labels = [
        "Latin → Cyrillic",
        "Latin → Latin (diacritics)",
        "Latin → Hebrew (pointed)",
    ]

    short_legend = axis.legend(
        [label_to_handle[label] for label in short_labels],
        short_labels,
        title=config["display_name"],
        bbox_to_anchor=(1.04, 1.1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        alignment="left",
        columnspacing=0.8,
        markerscale=0.75,
        handlelength=1.5,
        # fontsize=8,
        # title_fontproperties={"size": 9},
        ncol=2,
    )
    short_legend.get_title().set_fontweight("bold")
    axis.add_artist(short_legend)
    axis.legend(
        [label_to_handle[label] for label in long_labels],
        long_labels,
        bbox_to_anchor=(1.04, 0.64),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        alignment="left",
        markerscale=0.75,
        handlelength=1.5,
        # fontsize=8,
        # title_fontproperties={"size": 9},
        ncol=1,
    )


def plot_experiments(
    configs: list[dict[str, Any]],
    panels_by_experiment: dict[str, dict[str, pd.DataFrame]],
):
    figure = aes.plt.figure(
        figsize=(
            aes.ACL_PAPER_WIDTH_IN,
            len(configs) * aes.FIG_HEIGHT_SINGLE_ROW_IN * 0.9,
        ),
        layout="constrained",
    )
    grid = figure.add_gridspec(len(configs), 2, hspace=0, wspace=0)

    axes = []
    for row_index, config in enumerate(configs):
        shared_left_axis = axes[0][0] if axes else None
        shared_right_axis = axes[0][1] if axes else None
        left_axis = figure.add_subplot(grid[row_index, 0], sharex=shared_left_axis)
        right_axis = figure.add_subplot(
            grid[row_index, 1], sharex=shared_right_axis, sharey=left_axis
        )
        axes.append((left_axis, right_axis))
        panels = panels_by_experiment[config["slug"]]

        for axis, panel, x_column in (
            (left_axis, panels["size"], config["size_x"]),
            (right_axis, panels["length"], config["length_x"]),
        ):
            aes.sns.lineplot(
                data=panel,
                x=x_column,
                y="match_value",
                hue=config["hue"],
                style=config["style"],
                hue_order=config["order"],
                style_order=config["order"],
                palette=config["palette"],
                markers=True,
                markersize=7,
                linewidth=2,
                errorbar=None,
                ax=axis,
            )
            axis.set_ylim(-0.05, 1.05)
            axis.yaxis.set_major_formatter(aes.PCT_FORMATTER)
            axis.yaxis.grid(True, linestyle="--", alpha=0.5)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        left_axis.set_ylabel("Accuracy")
        right_axis.set_ylabel("")
        right_axis.tick_params(axis="y", which="both", labelleft=False)
        left_legend = left_axis.get_legend()
        if left_legend is not None:
            left_legend.remove()
        format_row_legend(right_axis, config)

        if row_index < len(configs) - 1:
            left_axis.set_xlabel("")
            right_axis.set_xlabel("")
            left_axis.tick_params(labelbottom=False)
            right_axis.tick_params(labelbottom=False)
        else:
            left_axis.set_xlabel("Grammar size")
            right_axis.set_xlabel("String length")
        left_axis.xaxis.set_major_formatter(aes.KMB_FORMATTER)
        right_axis.xaxis.set_major_formatter(aes.NICE_FORMATTER)

    axes[0][0].set_xlim(45, 10_500)
    axes[0][0].set_xscale("symlog")
    for left_axis, _ in axes:
        left_axis.xaxis.set_major_formatter(aes.KMB_FORMATTER)
        left_axis.xaxis.set_minor_locator(
            SymmetricalLogLocator(base=10, linthresh=2, subs=tuple(range(2, 10)))
        )
        left_axis.tick_params(axis="x", which="minor", length=2.5)
    axes[0][1].set_xlim(5, 35)
    axes[0][1].set_xticks([10, 15, 20, 25, 30, 35])
    return figure


def summarize_panel(
    panel: pd.DataFrame, config: dict[str, Any], panel_name: str
) -> pd.DataFrame:
    """Reduce individual results to the accuracy values plotted as curves."""

    x_column = str(config[f"{panel_name}_x"])
    hue_column = str(config["hue"])
    group_columns = [x_column, hue_column]
    if "k_shots" in panel.columns:
        group_columns.append("k_shots")
    summary = (
        panel.groupby(group_columns, dropna=False, observed=True)["match_value"]
        .mean()
        .reset_index()
    )
    # These rows identify the model in titles/palettes without retaining examples.
    if "fuzzy_model" in panel.columns:
        model_names = sorted(panel["fuzzy_model"].dropna().astype(str).unique())
        summary["fuzzy_model"] = ", ".join(model_names)
    return summary.assign(experiment=config["slug"], panel=panel_name)


def compute_plot_data() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Load revised result files and prepare the dataframe used by the plot."""

    configs = [dict(config) for config in EXPERIMENTS]
    plot_rows: list[pd.DataFrame] = []
    for config in configs:
        slug = str(config["slug"])
        panels = load_experiment_panels(slug)
        panels["length"] = add_length_quintile_bins(
            panels["length"], str(config["length_value"])
        )
        panels["size"] = add_grammar_size_quintile_bins(
            panels["size"], str(config["size_value"])
        )
        if slug == "size":
            configure_size_plot(config, panels)
        if slug == "wordorder":
            configure_wordorder_plot(config, panels)
        if slug == "fewshot":
            configure_fewshot_plot(config, panels)
        for panel_name, panel in panels.items():
            plot_rows.append(summarize_panel(panel, config, panel_name))
    return configs, pd.concat(plot_rows, ignore_index=True, sort=False)


def load_cached_plot_data(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, pd.DataFrame]]]:
    """Load the compact, version-controlled dataframe used for plotting."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached plot data at {path}. Run this script with "
            "`--recompute` to create it."
        )
    plot_data = pd.read_csv(path)
    required = {"experiment", "panel", "match_value"}
    missing = required - set(plot_data.columns)
    if missing:
        raise ValueError(
            f"Cached plot data is missing columns: {', '.join(sorted(missing))}"
        )
    configs = [dict(config) for config in EXPERIMENTS]
    panels_by_experiment: dict[str, dict[str, pd.DataFrame]] = {}
    for config in configs:
        slug = str(config["slug"])
        panels = {
            panel_name: plot_data.loc[
                plot_data["experiment"].eq(slug) & plot_data["panel"].eq(panel_name)
            ].copy()
            for panel_name in ("size", "length")
        }
        if any(panel.empty for panel in panels.values()):
            raise ValueError(f"Cached plot data has no complete panels for {slug!r}.")
        if slug == "size":
            configure_size_plot(config, panels)
        if slug == "wordorder":
            configure_wordorder_plot(config, panels)
        if slug == "fewshot":
            configure_fewshot_plot(config, panels)
        panels_by_experiment[slug] = panels
    return configs, panels_by_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the combined experiment-overview figure."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plot-data-path",
        type=Path,
        default=DEFAULT_PLOT_DATA_PATH,
        help="Cached dataframe used to render the figure (default: %(default)s).",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Rebuild the cached plotting dataframe from revised result files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recompute:
        configs, plot_data = compute_plot_data()
        args.plot_data_path.parent.mkdir(parents=True, exist_ok=True)
        plot_data.to_csv(args.plot_data_path, index=False)
        print(f"Saved {args.plot_data_path}")
        _, panels_by_experiment = load_cached_plot_data(args.plot_data_path)
    else:
        configs, panels_by_experiment = load_cached_plot_data(args.plot_data_path)

    with aes.sns.plotting_context("paper", font_scale=FONT_SCALE):
        figure = plot_experiments(configs, panels_by_experiment)
        output_path, _ = aes.save_figure(
            args.output,
            fig=figure,
            tight=False,
        )
    aes.plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
