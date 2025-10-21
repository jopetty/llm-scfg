import colorsys
import os
from collections import defaultdict
from typing import Any

import matplotlib as mpl  # noqa: F401
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt  # noqa: F401
import matplotlib.ticker as mtick
import pyrootutils
import seaborn as sns
from IPython.display import display
from matplotlib.axes._axes import Axes
from seaborn.palettes import _ColorPalette as ColorPalette

PROJECT_ROOT = pyrootutils.find_root(
    search_from=os.path.abspath(""), indicator=".project-root"
)

# Define types
Color = tuple[float, float, float]
Palette = dict[str, Color]
BasePalette = str
SNSPalette = sns.palettes._ColorPalette


FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
PAPER_WIDTH_IN: float = 5.5
FIG_HEIGHT_SINGLEROW_IN: float = 1.5

NICE_FORMATTER = mtick.EngFormatter(places=0, sep="")
PCT_FORMATTER = mtick.PercentFormatter(1.0)

rcs = {
    "font.size": 10.0,
    "axes.labelsize": "medium",
    "axes.titlesize": "medium",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
}

def darken(
    color: str
    | Color
    | dict[str, Any]
    | SNSPalette,
    by: float = 0.2,
):
    """
    Darken a color by provided amount.
    """

    def _darken_color(c: str | Color, by: float):
        by = min(max(0, by), 1)
        pct_darken = 1 - by

        if isinstance(c, str):
            c = sns.color_palette([c])[0]

        for c_i in c:
            if c_i > 1:
                c_i /= 255
        c_hls = colorsys.rgb_to_hls(c[0], c[1], c[2])
        # Darken the color by reducing the lightness

        c_hls = (
            c_hls[0],  # hue
            c_hls[1] * pct_darken,  # lightness
            c_hls[2],  # saturation
        )
        # Convert back to RGB
        c_rgb = colorsys.hls_to_rgb(c_hls[0], c_hls[1], c_hls[2])
        return c_rgb

    if isinstance(color, dict):
        # If color is a dictionary, assume it's a palette
        # and darken each color in the palette
        return {k: _darken_color(v, by) for k, v in color.items()}
    elif isinstance(color, SNSPalette):
        colors = [_darken_color(c, by) for c in color]
        return SNSPalette(colors)
    else:
        return _darken_color(color, by)


# Palettes
PALETTE_METRICS_BASE: BasePalette = "Set2"
PALETTE_METRICS: Palette = darken({
    "Exact Match": sns.color_palette(PALETTE_METRICS_BASE, n_colors=5)[0],
    "Bag of Words": sns.color_palette(PALETTE_METRICS_BASE, n_colors=5)[1],
    "BLEU Score": sns.color_palette(PALETTE_METRICS_BASE, n_colors=5)[3],
    "chrF++": sns.color_palette(PALETTE_METRICS_BASE, n_colors=5)[4],
    # "Edit Similarity": sns.color_palette(PALETTE_METRICS_BASE, n_colors=5)[2],
    "Edit Similarity": "#ffcc66",
}, by=0.4)

PALETTE_MODELS: Palette = darken({
    "gpt-5-nano": sns.color_palette("Reds", n_colors=4)[0],
    "gpt-5-mini": sns.color_palette("Reds", n_colors=4)[1],
    "gpt-5": sns.color_palette("Reds", n_colors=4)[2],
    "gemini-2.5-flash": sns.color_palette("Blues", n_colors=2)[0],
    "gemini-2.5-pro": sns.color_palette("Blues", n_colors=2)[1],
})

PALETTE_WORDORDER_BASE = sns.color_palette(
    palette="Dark2", n_colors=4
)
PALETTE_WORDORDER: Palette = darken({
    "all-same": PALETTE_WORDORDER_BASE[0],
    "head-diff": PALETTE_WORDORDER_BASE[1],
    "all-diff": PALETTE_WORDORDER_BASE[2],
})

PALETTE_ORTHOGRAPHY_BASE = sns.color_palette(
    palette="rainbow", n_colors=3
)
PALETTE_ORTHOGRAPHY: Palette = darken({
    "latin": PALETTE_ORTHOGRAPHY_BASE[0],
    "cyrillic": PALETTE_ORTHOGRAPHY_BASE[1],
    "yiddish": PALETTE_ORTHOGRAPHY_BASE[2],
}, by=0.4)
