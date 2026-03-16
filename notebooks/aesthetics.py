"""Shared plotting defaults for notebooks and project code.

This repo keeps a local copy of the shared aesthetics module and layers in
project-specific palettes for the LLM-SCFG experiments.
"""

from __future__ import annotations

import colorsys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

Color = tuple[float, float, float]
Palette = dict[str, Color]

# Widths are text block widths in inches, taken from the official conference
# LaTeX styles/templates current as of March 10, 2026.
_CM_TO_IN: float = 1.0 / 2.54

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

PAPER_WIDTH_IN: float = 5.5
FIG_HEIGHT_SINGLE_ROW_IN: float = 1.3
FIG_HEIGHT_DOUBLE_ROW_IN: float = 2.8

# Backwards-compatible aliases used throughout existing notebooks in this repo.
FIG_HEIGHT_SINGLEROW_IN: float = FIG_HEIGHT_SINGLE_ROW_IN
FIG_HEIGHT_DOUBLEROW_DIFFAXES_IN: float = FIG_HEIGHT_DOUBLE_ROW_IN

# ACL-style venues use A4 paper with 7.7 cm columns and a 0.6 cm gutter.
ACL_COLUMN_WIDTH_IN: float = 7.7 * _CM_TO_IN
ACL_PAPER_WIDTH_IN: float = (7.7 * 2.0 + 0.6) * _CM_TO_IN

# COLM and ICLR use the OpenReview single-column template with 5.5 in text width.
COLM_COLUMN_WIDTH_IN: float = 5.5
COLM_PAPER_WIDTH_IN: float = COLM_COLUMN_WIDTH_IN
ICLR_COLUMN_WIDTH_IN: float = 5.5
ICLR_PAPER_WIDTH_IN: float = ICLR_COLUMN_WIDTH_IN

# ICML uses a wider single-column layout.
ICML_COLUMN_WIDTH_IN: float = 6.75
ICML_PAPER_WIDTH_IN: float = ICML_COLUMN_WIDTH_IN

# NeurIPS uses a single-column 5.5 in text width.
NEURIPS_COLUMN_WIDTH_IN: float = 5.5
NEURIPS_PAPER_WIDTH_IN: float = NEURIPS_COLUMN_WIDTH_IN

NICE_FORMATTER = mtick.EngFormatter(places=0, sep="")
PCT_FORMATTER = mtick.PercentFormatter(1.0)


def format_compact_number(value: float, _: float | None = None) -> str:
    """Format values with compact decimal suffixes like 1K, 1M, 1B, and 1T."""

    abs_value = abs(value)
    if abs_value < 1_000:
        return f"{value:g}"

    suffixes = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ]
    for scale, suffix in suffixes:
        if abs_value >= scale:
            scaled = value / scale
            if float(scaled).is_integer():
                return f"{int(scaled)}{suffix}"
            return f"{scaled:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:g}"


def format_kmb(value: float, pos: float | None = None) -> str:
    """Backwards-compatible alias for the previous compact-number formatter."""

    return format_compact_number(value, pos)


COMPACT_NUMBER_FORMATTER = mtick.FuncFormatter(format_compact_number)
KMB_FORMATTER = COMPACT_NUMBER_FORMATTER

DEFAULT_RCS: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.labelsize": "medium",
    "axes.titlelocation": "left",
    "axes.titlesize": "medium",
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xaxis.labellocation": "left",
    "xtick.alignment": "center",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "figure.dpi": 120,
}

# Alias used by notebooks/scripts that want an explicit rc payload.
rcs = DEFAULT_RCS

sns.set_theme(style="ticks", context="paper", rc=DEFAULT_RCS)


def _to_rgb_tuple(color: str | Color) -> Color:
    if isinstance(color, str):
        rgb = sns.color_palette([color])[0]
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    c = tuple(float(x) for x in color)
    if len(c) != 3:
        raise ValueError(f"Expected RGB tuple with 3 values, got {color!r}")
    if any(x > 1.0 for x in c):
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
    return (c[0], c[1], c[2])


def _darken_color(color: str | Color, by: float) -> Color:
    by = min(max(float(by), 0.0), 1.0)
    rgb = _to_rgb_tuple(color)
    h, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    darker = colorsys.hls_to_rgb(h, lightness * (1.0 - by), saturation)
    return (float(darker[0]), float(darker[1]), float(darker[2]))


def _family_palette_key(family: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in family).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    if not slug:
        raise ValueError("family must contain at least one alphanumeric character")
    return f"models_{slug}"


def _model_display_name(family: str, model: str) -> str:
    family_clean = family.strip()
    model_clean = model.strip()
    if model_clean.lower().startswith(family_clean.lower()):
        return model_clean
    return f"{family_clean} {model_clean}"


def _relative_luminance(color: Color) -> float:
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def text_color_for_background(
    color: str | Color,
    *,
    dark: str = "white",
    light: str = "black",
) -> str:
    return dark if _relative_luminance(_to_rgb_tuple(color)) < 0.6 else light


def darken(
    color: str | Color | Mapping[str, str | Color] | Iterable[str | Color],
    by: float = 0.2,
):
    if isinstance(color, Mapping):
        return {k: _darken_color(cast(str | Color, v), by) for k, v in color.items()}
    if isinstance(color, str):
        return _darken_color(color, by)
    if (
        isinstance(color, tuple)
        and len(color) == 3
        and all(isinstance(x, (int, float)) for x in color)
    ):
        return _darken_color(cast(Color, color), by)
    return [_darken_color(cast(str | Color, c), by) for c in color]


PALETTES: dict[str, Any] = {"models": {}}


def set_model_palette(family: str, color: str, models: list[str]) -> Palette:
    if not models:
        raise ValueError("models must be a non-empty list of model names")

    # Assign darker shades to earlier models, independent of seaborn's
    # palette ordering for a particular color ramp.
    palette_colors = sorted(
        (
            _to_rgb_tuple(rgb)
            for rgb in sns.color_palette(color, n_colors=len(models) + 2)[1:-1]
        ),
        key=_relative_luminance,
    )
    family_palette = {
        model: rgb for model, rgb in zip(models, palette_colors, strict=True)
    }

    family_key = _family_palette_key(family)
    existing_family_palette = PALETTES.get(family_key, {})
    if isinstance(existing_family_palette, Mapping):
        for model in existing_family_palette:
            PALETTES["models"].pop(_model_display_name(family, model), None)

    PALETTES[family_key] = family_palette
    PALETTES["models"].update(
        {
            _model_display_name(family, model): rgb
            for model, rgb in family_palette.items()
        }
    )
    return family_palette


NONSEMANTIC_COLOR: Color = darken("#ffcc66", by=0.3)

# Project-specific palettes for the llm-scfg experiments.
PALETTE_METRICS_BASE = "Set2"
PALETTE_METRICS: Palette = darken(
    {
        "Exact Match": sns.color_palette(PALETTE_METRICS_BASE, n_colors=5)[0],
        "Bag of Words": sns.color_palette(PALETTE_METRICS_BASE, n_colors=5)[1],
        "BLEU Score": sns.color_palette(PALETTE_METRICS_BASE, n_colors=5)[3],
        "chrF++": sns.color_palette(PALETTE_METRICS_BASE, n_colors=5)[4],
        "Edit Similarity": "#ffcc66",
    },
    by=0.4,
)

PALETTE_MODELS: Palette = darken(
    {
        "gpt-5-nano": sns.color_palette("Reds", n_colors=4)[0],
        "gpt-5-mini": sns.color_palette("Reds", n_colors=4)[1],
        "gpt-5": sns.color_palette("Reds", n_colors=4)[2],
        "gemini-2.5-flash": sns.color_palette("Blues", n_colors=2)[0],
        "gemini-2.5-pro": sns.color_palette("Blues", n_colors=2)[1],
    }
)

PALETTE_WORDORDER_BASE = sns.color_palette(palette="Dark2", n_colors=4)
PALETTE_WORDORDER: Palette = darken(
    {
        "all-same": PALETTE_WORDORDER_BASE[0],
        "SVO": PALETTE_WORDORDER_BASE[0],
        "head-diff": PALETTE_WORDORDER_BASE[1],
        "SOV": PALETTE_WORDORDER_BASE[1],
        "all-diff": PALETTE_WORDORDER_BASE[2],
        "OVS": PALETTE_WORDORDER_BASE[2],
    }
)

PALETTE_ORTHOGRAPHY_BASE = sns.color_palette(palette="rainbow", n_colors=4)
PALETTE_ORTHOGRAPHY: Palette = darken(
    {
        "Latin → Latin": PALETTE_ORTHOGRAPHY_BASE[0],
        "Latin → Cyrillic": PALETTE_ORTHOGRAPHY_BASE[1],
        "Latin → Hebrew (nikkud)": PALETTE_ORTHOGRAPHY_BASE[2],
        "Latin → Hebrew (no nikkud)": PALETTE_ORTHOGRAPHY_BASE[3],
    },
    by=0.4,
)

PALETTES["metrics"] = PALETTE_METRICS
PALETTES["wordorder"] = PALETTE_WORDORDER
PALETTES["orthography"] = PALETTE_ORTHOGRAPHY
PALETTES["models"].update(PALETTE_MODELS)


def set_figure_title(
    fig,
    title: str,
    subtitle: str | None = None,
    *,
    x: float = 0.0,
    y: float = 0.995,
    title_kwargs: Mapping[str, Any] | None = None,
    subtitle_kwargs: Mapping[str, Any] | None = None,
):
    """Add a left-aligned bold figure title and optional subtitle."""

    title_y = y if subtitle is not None else y - 0.07
    title_text = fig.suptitle(
        title,
        x=x,
        y=title_y,
        ha="left",
        fontweight="bold",
        **dict(title_kwargs or {}),
    )
    subtitle_text = None
    if subtitle is not None:
        subtitle_defaults = {
            "fontsize": plt.rcParams["font.size"],
            "fontweight": "normal",
            "ha": "left",
            "va": "top",
        }
        subtitle_defaults.update(dict(subtitle_kwargs or {}))
        subtitle_text = fig.text(x, y - 0.07, subtitle, **subtitle_defaults)
    return title_text, subtitle_text


def save_figure(
    path: str | Path,
    *,
    fig=None,
    tight: bool = True,
    transparent: bool = True,
    png_dpi: int = 300,
    **savefig_kwargs: Any,
) -> tuple[Path, Path]:
    """Save the current figure as both PDF and PNG."""

    figure = fig if fig is not None else plt.gcf()
    output_base = Path(path)
    if output_base.suffix.lower() in {".pdf", ".png"}:
        output_base = output_base.with_suffix("")
    output_base = FIGURES_DIR / output_base
    output_base.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = dict(savefig_kwargs)
    save_kwargs.setdefault("transparent", transparent)
    if tight:
        save_kwargs.setdefault("bbox_inches", "tight")

    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")

    figure.savefig(pdf_path, **save_kwargs)
    figure.savefig(png_path, dpi=png_dpi, **save_kwargs)
    return pdf_path, png_path


__all__ = [
    "Color",
    "Palette",
    "PROJECT_ROOT",
    "FIGURES_DIR",
    "PAPER_WIDTH_IN",
    "FIG_HEIGHT_SINGLE_ROW_IN",
    "FIG_HEIGHT_DOUBLE_ROW_IN",
    "FIG_HEIGHT_SINGLEROW_IN",
    "FIG_HEIGHT_DOUBLEROW_DIFFAXES_IN",
    "ACL_COLUMN_WIDTH_IN",
    "ACL_PAPER_WIDTH_IN",
    "COLM_COLUMN_WIDTH_IN",
    "COLM_PAPER_WIDTH_IN",
    "ICLR_COLUMN_WIDTH_IN",
    "ICLR_PAPER_WIDTH_IN",
    "ICML_COLUMN_WIDTH_IN",
    "ICML_PAPER_WIDTH_IN",
    "NEURIPS_COLUMN_WIDTH_IN",
    "NEURIPS_PAPER_WIDTH_IN",
    "NICE_FORMATTER",
    "PCT_FORMATTER",
    "format_compact_number",
    "format_kmb",
    "COMPACT_NUMBER_FORMATTER",
    "KMB_FORMATTER",
    "DEFAULT_RCS",
    "rcs",
    "darken",
    "PALETTES",
    "set_model_palette",
    "set_figure_title",
    "save_figure",
    "text_color_for_background",
    "NONSEMANTIC_COLOR",
    "PALETTE_METRICS_BASE",
    "PALETTE_METRICS",
    "PALETTE_MODELS",
    "PALETTE_WORDORDER_BASE",
    "PALETTE_WORDORDER",
    "PALETTE_ORTHOGRAPHY_BASE",
    "PALETTE_ORTHOGRAPHY",
]
