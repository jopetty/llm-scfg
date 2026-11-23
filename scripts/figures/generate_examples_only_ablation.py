"""Generate the examples-only (grammar ablation) comparison.

The ablation asks how much of the model's accuracy comes from interpreting
the in-context grammar rather than from generalizing over the parallel
exemplars. It contrasts three conditions on matched items:

* **grammar only** -- the ``k=0`` cell of the existing few-shot run;
* **grammar + examples** -- the ``k>0`` cells of that run;
* **examples only** -- the new ablation, same exemplars, no grammar.

The examples-only condition carries a hard analytic ceiling. A sentence can
only be exact-matched if every one of its source words appears somewhere in
the k exemplars, so the ceiling is the fraction of items that are fully
covered. This script reports that ceiling alongside the measured accuracy and
also reports accuracy conditional on full coverage, which is the comparison
that is not confounded by vocabulary coverage.

The two runs do not cover the same grammars -- the original run used all 66
generated grammars, the ablation the 42 named in ``fewshot_grammars.txt`` --
so the grammar conditions are restricted to the ablation's grammars before
anything is compared.

Missing ``k`` cells are skipped rather than fatal, so this can be run against
a partial sweep and re-run as more batches land.

Example:
    uv run python scripts/figures/generate_examples_only_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyrootutils

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from generate_fewshot_accuracy import (  # noqa: E402
    extract_answer,
    load_rows,
    response_content,
)

from notebooks import aesthetics as aes  # noqa: E402

BATCH_DIR = PROJECT_ROOT / "batches" / "fewshot_exp"
DATA_DIR = PROJECT_ROOT / "data" / "fewshot_exp"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache" / "examples-only"
DEFAULT_OUTPUT = FIGURES_DIR / "examples_only_ablation"

ABLATION_INPUT_GLOB = "inputs_*_examples_only_k*.jsonl"

CONDITION_GRAMMAR_ONLY = "Grammar only"
CONDITION_GRAMMAR_EXAMPLES = "Grammar + examples"
CONDITION_EXAMPLES_ONLY = "Examples only"


def _read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def grammar_name_from_custom_id(custom_id: str) -> str:
    """Custom IDs are ``<grammar>-<filehash>-k<k>-sample-<i>``."""

    return custom_id.split("-", 1)[0]


def load_ablation_inputs(batch_dir: Path) -> dict[str, dict[str, Any]]:
    """Index examples-only request metadata by custom ID.

    Metadata is not echoed in OpenAI batch output records, so the gold
    translation and the shot count have to come from the request files.
    """

    inputs: dict[str, dict[str, Any]] = {}
    for path in sorted(batch_dir.glob(ABLATION_INPUT_GLOB)):
        for record in _read_jsonl(path):
            custom_id = record.get("custom_id")
            metadata = (record.get("body") or {}).get("metadata") or {}
            if not isinstance(custom_id, str) or not metadata:
                continue
            inputs[custom_id] = {
                "grammar": grammar_name_from_custom_id(custom_id),
                "k_shots": int(metadata["k_shots"]),
                "input_sentence": metadata.get("input_sentence"),
                "output_sentence": metadata.get("output_sentence"),
                "model": (record.get("body") or {}).get("model"),
            }
    if not inputs:
        raise FileNotFoundError(
            f"No examples-only inputs matching {ABLATION_INPUT_GLOB} under {batch_dir}"
        )
    return inputs


def exemplar_source_vocab(data_dir: Path, k_values: set[int]) -> dict[tuple, set]:
    """Source-side vocabulary of the first k held-out exemplars per grammar.

    Mirrors ``main.shot_examples_from_samples``, which takes the first ``k``
    rows of the dedicated shot pool, so the reconstruction is exact rather
    than approximate.
    """

    vocab: dict[tuple, set] = {}
    for path in sorted(data_dir.glob("shots_*.jsonl")):
        grammar = path.stem.removeprefix("shots_")
        shots = list(_read_jsonl(path))
        for k in k_values:
            words: set = set()
            for shot in shots[:k]:
                sentence = shot.get("left_phonetic") or shot.get("left") or ""
                words |= set(str(sentence).split())
            vocab[(grammar, k)] = words
    return vocab


def load_ablation_rows(batch_dir: Path, data_dir: Path) -> pd.DataFrame:
    """Score every examples-only output that joins to a known request."""

    inputs = load_ablation_inputs(batch_dir)
    vocab = exemplar_source_vocab(
        data_dir, {meta["k_shots"] for meta in inputs.values()}
    )

    rows: list[dict[str, Any]] = []
    for path in sorted(batch_dir.glob("*_output.jsonl")):
        for record in _read_jsonl(path):
            custom_id = record.get("custom_id")
            meta = inputs.get(custom_id) if isinstance(custom_id, str) else None
            if meta is None:
                continue  # belongs to the grammar run, not the ablation
            answer = extract_answer(response_content(record))
            source_words = set(str(meta["input_sentence"] or "").split())
            covered = vocab.get((meta["grammar"], meta["k_shots"]), set())
            rows.append(
                {
                    "batch_file": path.name,
                    "custom_id": custom_id,
                    "grammar": meta["grammar"],
                    "k_shots": meta["k_shots"],
                    "model_answer": answer,
                    "output_sentence": meta["output_sentence"],
                    "exact_match": bool(
                        answer is not None and answer == meta["output_sentence"]
                    ),
                    "fully_covered": bool(source_words and source_words <= covered),
                }
            )

    if not rows:
        raise ValueError(
            "No examples-only outputs found. Download the batch results into "
            f"{batch_dir} as batch_<id>_output.jsonl first."
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["custom_id"])


def coverage_ceiling(
    data_dir: Path, grammars: set[str], k_values: list[int]
) -> pd.DataFrame:
    """Fraction of items whose source words are fully covered by k exemplars.

    This is an exact upper bound on examples-only exact-match accuracy and
    needs no model outputs, so it is defined for every k, including cells that
    have not been run.
    """

    vocab = exemplar_source_vocab(data_dir, set(k_values))
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("samples_*.jsonl")):
        grammar = path.stem.removeprefix("samples_")
        if grammar not in grammars:
            continue
        samples = list(_read_jsonl(path))
        for k in k_values:
            covered = vocab.get((grammar, k), set())
            for sample in samples:
                sentence = sample.get("left_phonetic") or sample.get("left") or ""
                words = set(str(sentence).split())
                if not words:
                    continue
                rows.append({"k_shots": k, "covered": bool(words <= covered)})
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("k_shots")["covered"]
        .mean()
        .mul(100)
        .rename("ceiling_pct")
        .reset_index()
    )


def build_summary(batch_dir: Path, data_dir: Path) -> pd.DataFrame:
    """Assemble the three-condition comparison on matched grammars."""

    ablation = load_ablation_rows(batch_dir, data_dir)
    grammars = set(ablation["grammar"].unique())

    grammar_rows = load_rows(batch_dir)
    grammar_rows = grammar_rows.assign(
        grammar=grammar_rows["custom_id"].map(grammar_name_from_custom_id)
    )
    matched = grammar_rows[grammar_rows["grammar"].isin(grammars)]
    if matched.empty:
        raise ValueError("No grammar-condition rows share grammars with the ablation.")

    records: list[dict[str, Any]] = []
    for raw_k, group in matched.groupby("k_shots"):
        k = int(cast(int, raw_k))
        condition = CONDITION_GRAMMAR_ONLY if k == 0 else CONDITION_GRAMMAR_EXAMPLES
        records.append(
            {
                "condition": condition,
                "k_shots": k,
                "accuracy_pct": group["exact_match"].mean() * 100,
                "n": len(group),
                "covered_n": pd.NA,
                "accuracy_given_covered_pct": pd.NA,
            }
        )

    for raw_k, group in ablation.groupby("k_shots"):
        covered = group[group["fully_covered"]]
        records.append(
            {
                "condition": CONDITION_EXAMPLES_ONLY,
                "k_shots": int(cast(int, raw_k)),
                "accuracy_pct": group["exact_match"].mean() * 100,
                "n": len(group),
                "covered_n": len(covered),
                "accuracy_given_covered_pct": (
                    covered["exact_match"].mean() * 100 if len(covered) else pd.NA
                ),
            }
        )

    summary = pd.DataFrame(records)
    ceiling = coverage_ceiling(data_dir, grammars, sorted(summary["k_shots"].unique()))
    summary = summary.merge(ceiling, on="k_shots", how="left")
    summary.loc[summary["condition"] != CONDITION_EXAMPLES_ONLY, "ceiling_pct"] = pd.NA
    return summary.sort_values(["condition", "k_shots"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame):
    """Plot accuracy against shot count, with the ablation's ceiling shaded."""

    plt = aes.plt
    fig, ax = plt.subplots(
        figsize=(aes.COLM_PAPER_WIDTH_IN, aes.FIG_HEIGHT_DOUBLE_ROW_IN)
    )

    grammar_only = summary[summary["condition"] == CONDITION_GRAMMAR_ONLY]
    with_examples = summary[summary["condition"] == CONDITION_GRAMMAR_EXAMPLES]
    grammar_curve = pd.concat([grammar_only, with_examples]).sort_values("k_shots")
    ablation = summary[summary["condition"] == CONDITION_EXAMPLES_ONLY].sort_values(
        "k_shots"
    )

    palette = aes.sns.color_palette("Dark2", n_colors=3)

    if not grammar_curve.empty:
        ax.plot(
            grammar_curve["k_shots"],
            grammar_curve["accuracy_pct"] / 100.0,
            marker="o",
            color=palette[0],
            label="Grammar + examples",
        )
    if not grammar_only.empty:
        ax.axhline(
            float(grammar_only["accuracy_pct"].iloc[0]) / 100.0,
            color=palette[0],
            linestyle=":",
            linewidth=1.0,
            label="Grammar only ($k{=}0$)",
        )
    if not ablation.empty:
        ax.plot(
            ablation["k_shots"],
            ablation["accuracy_pct"] / 100.0,
            marker="s",
            color=palette[1],
            label="Examples only",
        )
        ax.plot(
            ablation["k_shots"],
            ablation["ceiling_pct"] / 100.0,
            linestyle="--",
            linewidth=1.0,
            color=palette[1],
            alpha=0.7,
            label="Examples-only ceiling",
        )
        ax.fill_between(
            ablation["k_shots"],
            ablation["accuracy_pct"] / 100.0,
            ablation["ceiling_pct"] / 100.0,
            color=palette[1],
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xlabel("In-context examples ($k$)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(aes.PCT_FORMATTER)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(frameon=False, fontsize="x-small", loc="upper left", ncol=2)
    aes.sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=BATCH_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    summary = build_summary(args.batch_dir, args.data_dir)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = CACHE_DIR / "examples_only_summary.csv"
    summary.to_csv(summary_path, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    print(f"\nWrote {summary_path}")

    if not args.no_figure:
        fig = plot_summary(summary)
        pdf_path, _ = aes.save_figure(args.output, fig=fig)
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
