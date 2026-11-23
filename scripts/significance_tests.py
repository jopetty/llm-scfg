"""Cluster-respecting significance tests for the condition experiments.

The word-order result is a *null*: we claim that changing the target
word order does not change translation accuracy. A null claim needs two
things a bare p-value does not give.

1. A test whose unit of analysis matches the design. Condition is a
   property of a *grammar*, not of an item: each grammar realizes one
   target word order and contributes 120 sentences. Treating the 5,040
   sentences as independent understates the variance and so overstates
   significance, which for a null result means manufacturing a
   difference that is not there.
2. A bound on the effect size. "p > 0.05" is equally consistent with a
   small effect and a badly powered test, so we report confidence
   intervals on every pairwise difference and read the null off the
   interval, not off the p-value.

Design. All three condition experiments share one structure: grammars
are generated in matched sets, two per (grammar-size block, condition).
Grammar size dominates accuracy -- from near-ceiling at 46 rules to
under 20% at 8026 -- so size is a blocking factor and the condition
contrast is tested *within* blocks.

Tests.

- Omnibus: an exact randomization test. Condition labels are permuted
  among the grammars inside each size block, which is precisely the
  randomization the design implies; the statistic is the
  between-condition sum of squares of block-centered grammar accuracy.
  No distributional assumption, and nothing has to converge -- which
  matters because several cells sit at or near 100% accuracy, where a
  mixed-effects logistic regression would be fighting separation.
- Pairwise: the same randomization test for each pair, plus a
  confidence interval from the randomized-block ANOVA on grammar-level
  accuracy (block + condition + interaction). The error term is chosen
  by testing the interaction: when block x condition is not
  significant the interaction and within-cell sums of squares are
  pooled, which uses the two grammars per cell instead of discarding
  them; otherwise the interaction mean square is used, which is the
  conservative choice. The test and the chosen term are both reported
  so the decision is visible rather than assumed.

  Two earlier versions were wrong and are worth recording. Bootstrapping
  grammars inside each cell underestimates the variance so badly (two
  grammars per cell) that it produced intervals excluding zero for
  differences the randomization test put at p = 0.35. Using only the
  per-block differences of cell means is valid but discards the
  replication, widening the word-order bound from +/-6.8 to +/-12.7
  percentage points for no good reason once the interaction is known to
  be negligible.
- Equivalence: the TOST bound, i.e. the smallest margin against which
  equivalence is established at the stated level. This is what actually
  supports the null.
- Naive comparison: the omnibus statistic recomputed over items as if
  they were independent. Reported only to show how much the clustering
  matters.

Usage:
    uv run python scripts/significance_tests.py
    uv run python scripts/significance_tests.py --experiment wordorder
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyrootutils
from scipy import stats

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PANEL_DIR = PROJECT_ROOT / "notebooks" / "cache" / "combined-exp" / "revised"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "cache" / "significance"

CONDITION_COLUMN = {
    "wordorder": "target_word_order",
    "orthography": "target_orthography",
    "agreement": "agreement_condition",
}
SIZE_COLUMN = {
    "wordorder": "size",
    "orthography": "grammar_size",
    "agreement": "grammar_size",
}
EXPERIMENT_LABEL = {
    "wordorder": "Word order",
    "orthography": "Orthography",
    "agreement": "Agreement",
}
DEFAULT_SEED = 20261123
DEFAULT_PERMUTATIONS = 100_000
NAIVE_PERMUTATIONS = 20_000
CI_LEVEL = 0.95
TOST_ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_panel(slug: str) -> pd.DataFrame:
    """Load the item-level panel written by the overview figure script."""

    path = PANEL_DIR / f"{slug}_grammar_size.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run "
            "`uv run python scripts/figures/"
            "generate_combined_experiment_overview.py` first."
        )
    panel = pd.read_csv(path)
    needed = [CONDITION_COLUMN[slug], SIZE_COLUMN[slug], "grammar_name", "match_value"]
    missing = [column for column in needed if column not in panel.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return panel


def grammar_table(panel: pd.DataFrame, slug: str) -> pd.DataFrame:
    """Collapse items to one row per grammar: the unit of randomization."""

    table = (
        panel.groupby(
            ["grammar_name", CONDITION_COLUMN[slug], SIZE_COLUMN[slug]],
            as_index=False,
        )
        .agg(n_items=("match_value", "size"), accuracy=("match_value", "mean"))
        .rename(
            columns={
                CONDITION_COLUMN[slug]: "condition",
                SIZE_COLUMN[slug]: "block",
            }
        )
    )
    return table.sort_values(["block", "condition", "grammar_name"]).reset_index(
        drop=True
    )


# --------------------------------------------------------------------------- #
# Randomization test
# --------------------------------------------------------------------------- #


def block_center(values: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Subtract each block's mean, leaving only within-block contrast."""

    centered = values.astype(float).copy()
    for block in np.unique(blocks):
        mask = blocks == block
        centered[mask] -= centered[mask].mean()
    return centered


def omnibus_statistic(
    centered: np.ndarray, labels: np.ndarray, conditions: list[str]
) -> float:
    """Between-condition sum of squares of block-centered accuracies."""

    total = 0.0
    for name in conditions:
        mask = labels == name
        if mask.any():
            total += mask.sum() * centered[mask].mean() ** 2
    return float(total)


def pairwise_statistic(
    centered: np.ndarray, labels: np.ndarray, left: str, right: str
) -> float:
    """Block-adjusted difference in accuracy between two conditions."""

    return float(centered[labels == left].mean() - centered[labels == right].mean())


def permute_within_blocks(
    labels: np.ndarray, blocks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Reassign condition labels among the grammars inside each block."""

    permuted = labels.copy()
    for block in np.unique(blocks):
        mask = blocks == block
        permuted[mask] = rng.permutation(labels[mask])
    return permuted


def permutation_tests(
    table: pd.DataFrame, n_permutations: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Omnibus and pairwise randomization tests, blocked by grammar size."""

    values = table["accuracy"].to_numpy(dtype=float)
    labels = table["condition"].to_numpy()
    blocks = table["block"].to_numpy()
    centered = block_center(values, blocks)
    conditions = sorted(set(labels))
    pairs = list(itertools.combinations(conditions, 2))

    observed_omnibus = omnibus_statistic(centered, labels, conditions)
    observed_pairs = {
        pair: pairwise_statistic(centered, labels, *pair) for pair in pairs
    }

    omnibus_hits = 0
    pair_hits = dict.fromkeys(pairs, 0)
    for _ in range(n_permutations):
        shuffled = permute_within_blocks(labels, blocks, rng)
        if omnibus_statistic(centered, shuffled, conditions) >= observed_omnibus:
            omnibus_hits += 1
        for pair in pairs:
            if abs(pairwise_statistic(centered, shuffled, *pair)) >= abs(
                observed_pairs[pair]
            ):
                pair_hits[pair] += 1

    # Add-one correction: a randomization p-value is never exactly zero.
    return {
        "omnibus_p": (omnibus_hits + 1) / (n_permutations + 1),
        "pairs": {
            pair: {"p": (pair_hits[pair] + 1) / (n_permutations + 1)} for pair in pairs
        },
    }


# --------------------------------------------------------------------------- #
# Randomized-block intervals
# --------------------------------------------------------------------------- #


def cell_means(table: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy per (block, condition) cell."""

    return table.pivot_table(
        index="block", columns="condition", values="accuracy", aggfunc="mean"
    )


def is_balanced(table: pd.DataFrame) -> bool:
    """True when every (block, condition) cell holds the same count."""

    counts = table.groupby(["block", "condition"]).size()
    expected = table["block"].nunique() * table["condition"].nunique()
    return len(counts) == expected and counts.nunique() == 1


def randomized_block_anova(table: pd.DataFrame) -> dict[str, Any] | None:
    """Two-way ANOVA on grammar accuracy; None if the design is unbalanced.

    Returns the interaction test and the error mean square to use for
    condition contrasts. Grammar size is the blocking factor and
    dominates accuracy, so it is always removed first.
    """

    if not is_balanced(table):
        return None

    values = table["accuracy"].to_numpy(dtype=float)
    grand = float(values.mean())
    n_blocks = table["block"].nunique()
    n_conditions = table["condition"].nunique()
    per_cell = len(table) // (n_blocks * n_conditions)

    def between(keys: list[str]) -> float:
        return float(
            sum(
                len(group) * (group["accuracy"].mean() - grand) ** 2
                for _, group in table.groupby(keys)
            )
        )

    ss_total = float(((values - grand) ** 2).sum())
    ss_block = between(["block"])
    ss_condition = between(["condition"])
    ss_cells = between(["block", "condition"])
    ss_interaction = ss_cells - ss_block - ss_condition
    ss_error = ss_total - ss_cells

    df_condition = n_conditions - 1
    df_interaction = (n_blocks - 1) * (n_conditions - 1)
    df_error = n_blocks * n_conditions * (per_cell - 1)

    ms_interaction = ss_interaction / df_interaction
    f_interaction = float("nan")
    p_interaction = float("nan")
    if df_error > 0 and ss_error > 0:
        ms_error = ss_error / df_error
        f_interaction = ms_interaction / ms_error
        p_interaction = float(stats.f.sf(f_interaction, df_interaction, df_error))

    # Pool the interaction into the error unless it is significant.
    pooled = not (p_interaction == p_interaction and p_interaction < 0.05)
    if pooled and df_error > 0:
        error_ms = (ss_interaction + ss_error) / (df_interaction + df_error)
        error_df = df_interaction + df_error
        error_term = "interaction + within-cell (pooled)"
    else:
        error_ms = ms_interaction
        error_df = df_interaction
        error_term = "interaction"

    ms_condition = ss_condition / df_condition
    f_condition = ms_condition / error_ms if error_ms > 0 else float("nan")
    return {
        "f_interaction": f_interaction,
        "p_interaction": p_interaction,
        "f_condition": float(f_condition),
        "p_condition": float(stats.f.sf(f_condition, df_condition, error_df)),
        "error_ms": float(error_ms),
        "error_df": int(error_df),
        "error_term": error_term,
        "grammars_per_condition": int(len(table) // n_conditions),
    }


def anova_interval(
    anova: dict[str, Any], difference: float, level: float = CI_LEVEL
) -> dict[str, float]:
    """Interval on a pairwise difference using the ANOVA error term."""

    se = float(np.sqrt(2 * anova["error_ms"] / anova["grammars_per_condition"]))
    df = anova["error_df"]
    half = float(stats.t.ppf(0.5 + level / 2, df=df)) * se
    tost_half = float(stats.t.ppf(1 - TOST_ALPHA, df=df)) * se
    return {
        "difference_pp": 100 * difference,
        "ci_low_pp": 100 * (difference - half),
        "ci_high_pp": 100 * (difference + half),
        "tost_bound_pp": 100 * (abs(difference) + tost_half),
        "se_pp": 100 * se,
        "df": df,
    }


def paired_interval(
    cells: pd.DataFrame, left: str, right: str, level: float = CI_LEVEL
) -> dict[str, float]:
    """Fallback interval for unbalanced designs: blocks as the unit.

    Valid but conservative, since it throws away within-cell
    replication. Used only when the ANOVA above cannot be formed.
    """

    paired = cells[[left, right]].dropna()
    diffs = (paired[left] - paired[right]).to_numpy(dtype=float)
    n = len(diffs)
    mean = float(diffs.mean()) if n else float("nan")
    if n < 2:
        return {
            "difference_pp": 100 * mean,
            "ci_low_pp": float("nan"),
            "ci_high_pp": float("nan"),
            "tost_bound_pp": float("nan"),
            "se_pp": float("nan"),
            "df": max(n - 1, 0),
        }

    se = float(diffs.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.5 + level / 2, df=n - 1)) * se
    tost_half = float(stats.t.ppf(1 - TOST_ALPHA, df=n - 1)) * se
    return {
        "difference_pp": 100 * mean,
        "ci_low_pp": 100 * (mean - half),
        "ci_high_pp": 100 * (mean + half),
        "tost_bound_pp": 100 * (abs(mean) + tost_half),
        "se_pp": 100 * se,
        "df": n - 1,
    }


def naive_omnibus_p(
    panel: pd.DataFrame, slug: str, n_permutations: int, rng: np.random.Generator
) -> float:
    """The omnibus test rerun as if items were independent.

    Reported only for contrast: this is the p-value one gets by ignoring
    that condition is assigned to grammars rather than to sentences.
    """

    outcomes = panel["match_value"].to_numpy(dtype=float)
    labels = panel[CONDITION_COLUMN[slug]].to_numpy()
    conditions = sorted(set(labels))
    sizes = [int((labels == name).sum()) for name in conditions]
    grand = float(outcomes.mean())
    bounds = np.cumsum([0, *sizes])

    def statistic(vec: np.ndarray) -> float:
        total = 0.0
        for index, size in enumerate(sizes):
            segment = vec[bounds[index] : bounds[index + 1]]
            total += size * (segment.mean() - grand) ** 2
        return total

    ordered = np.concatenate([outcomes[labels == name] for name in conditions])
    observed = statistic(ordered)
    hits = sum(
        statistic(rng.permutation(outcomes)) >= observed for _ in range(n_permutations)
    )
    return (hits + 1) / (n_permutations + 1)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def analyze(
    slug: str,
    n_permutations: int,
    seed: int,
    run_naive: bool = True,
) -> dict[str, Any]:
    panel = load_panel(slug)
    table = grammar_table(panel, slug)
    cells = cell_means(table)
    rng = np.random.default_rng(seed)

    perm = permutation_tests(table, n_permutations, rng)
    anova = randomized_block_anova(table)
    naive_p = (
        naive_omnibus_p(panel, slug, NAIVE_PERMUTATIONS, rng) if run_naive else None
    )

    centered = block_center(
        table["accuracy"].to_numpy(dtype=float), table["block"].to_numpy()
    )
    labels = table["condition"].to_numpy()

    rows: list[dict[str, Any]] = []
    for pair, result in perm["pairs"].items():
        difference = pairwise_statistic(centered, labels, *pair)
        interval = (
            anova_interval(anova, difference)
            if anova is not None
            else paired_interval(cells, *pair)
        )
        rows.append(
            {
                "experiment": slug,
                "left": pair[0],
                "right": pair[1],
                "p_permutation": result["p"],
                **interval,
            }
        )

    condition_means = {
        str(name): 100 * float(group["accuracy"].mean())
        for name, group in table.groupby("condition")
    }
    finite = [row for row in rows if np.isfinite(row["ci_low_pp"])]
    return {
        "experiment": slug,
        "label": EXPERIMENT_LABEL[slug],
        "n_items": int(len(panel)),
        "n_grammars": int(len(table)),
        "n_blocks": int(table["block"].nunique()),
        "model": sorted({str(m) for m in panel["fuzzy_model"].dropna()}),
        "condition_accuracy_pct": condition_means,
        "omnibus_p": perm["omnibus_p"],
        "anova": anova,
        "naive_item_level_omnibus_p": naive_p,
        "max_abs_difference_pp": max(abs(row["difference_pp"]) for row in rows),
        "max_tost_bound_pp": max(row["tost_bound_pp"] for row in finite)
        if finite
        else float("nan"),
        "pairs": rows,
        "n_permutations": n_permutations,
        "seed": seed,
    }


def print_report(result: dict[str, Any]) -> None:
    print(f"=== {result['label']} ({result['experiment']})")
    print(
        f"  {result['n_items']:,} items, {result['n_grammars']} grammars, "
        f"{result['n_blocks']} size blocks, model(s): {', '.join(result['model'])}"
    )
    print("  condition accuracy (%):")
    for name, value in sorted(
        result["condition_accuracy_pct"].items(), key=lambda kv: -kv[1]
    ):
        print(f"    {name:<28} {value:6.2f}")
    print(f"  omnibus randomization p    = {result['omnibus_p']:.4f}")
    anova = result.get("anova")
    if anova is not None:
        print(
            f"  block x condition          : F = {anova['f_interaction']:.2f}, "
            f"p = {anova['p_interaction']:.4f}  ->  error term: "
            f"{anova['error_term']} ({anova['error_df']} df)"
        )
        print(
            f"  omnibus ANOVA F            = {anova['f_condition']:.3f}, "
            f"p = {anova['p_condition']:.4f}"
        )
    else:
        print("  unbalanced design; intervals use blocks as the unit")
    if result["naive_item_level_omnibus_p"] is not None:
        print(
            "  naive item-level p         = "
            f"{result['naive_item_level_omnibus_p']:.4f}"
            "  (ignores grammar clustering)"
        )
    print("  pairwise (block-adjusted, 95% CI):")
    for row in sorted(result["pairs"], key=lambda r: -abs(r["difference_pp"])):
        print(
            f"    {row['left']:<28} - {row['right']:<28} "
            f"{row['difference_pp']:+7.2f} pp  "
            f"[{row['ci_low_pp']:+7.2f}, {row['ci_high_pp']:+7.2f}]  "
            f"p = {row['p_permutation']:.4f}"
        )
    print(
        f"  largest |difference| = {result['max_abs_difference_pp']:.2f} pp;"
        f" equivalence established at +/-{result['max_tost_bound_pp']:.2f} pp"
    )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster-respecting significance tests for condition experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(CONDITION_COLUMN),
        action="append",
        help="Experiment to test; repeatable. Default: all three.",
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--skip-naive",
        action="store_true",
        help="Skip the item-level comparison test, which is the slow part.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slugs = args.experiment or ["wordorder", "orthography", "agreement"]

    results = []
    for slug in slugs:
        result = analyze(
            slug,
            n_permutations=args.permutations,
            seed=args.seed,
            run_naive=not args.skip_naive,
        )
        print_report(result)
        results.append(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "significance_tests.json"
    with json_path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {json_path}")

    flat = pd.DataFrame([row for result in results for row in result["pairs"]])
    csv_path = OUTPUT_DIR / "significance_pairs.csv"
    flat.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
