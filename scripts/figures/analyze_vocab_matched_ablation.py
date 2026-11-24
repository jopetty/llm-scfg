"""Score the vocabulary-conditioned examples-only condition.

The paper's examples-only ablation shows every item the *first* k sentences
of its grammar's held-out shot pool, so which words an item sees is an
accident of pool order. The vocabulary-conditioned condition
(``scripts/generate_vocab_matched_batchfiles.py``) holds the evaluation
targets fixed and instead retrieves, per item, the k pool sentences covering
the most of that item's source vocabulary.

Comparing the two isolates coverage from k. Vocabulary-conditioned k=16
covers 28.21% of items against random k=16's 25.95% -- nearly matched -- so
a difference between those cells is about *which* words the exemplars carry,
not how many items are translatable in principle. Vocabulary-conditioned
k=8 reaches that same 28.21% with half the exemplars.

Coverage comes from the request metadata (``source_vocab_covered``) rather
than being reconstructed, because the retrieved exemplar set varies per item
and cannot be recovered from pool order.

Example:
    uv run python scripts/figures/analyze_vocab_matched_ablation.py
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

BATCH_DIR = PROJECT_ROOT / "batches" / "fewshot_exp"
CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache" / "vocab-matched"

# Matches both pool variants: "..._vocabmatched_k8_..." (the shipped
# 18-sentence pool) and "..._vocabmatched_highcov_k8_..." (large pools).
# The random condition's "..._examples_only_k8_..." does not match.
VOCAB_MATCHED_GLOB = "inputs_*_examples_only_vocabmatched*_k*.jsonl"
RANDOM_GLOB = "inputs_*_examples_only_k*.jsonl"

# The paper's table columns. VME is the high-coverage condition: exemplars
# retrieved from the large held-out pools, which lift coverage to ~100%.
# The earlier 18-sentence-pool retrieval run is kept in the detailed
# summary under its own label, since the k=8 and k=16 cells there cover an
# identical item set and so support the coverage-pinned comparison, but it
# is superseded for reporting.
SELECTION_RANDOM = "random"
SELECTION_VME = "vocab_matched_highcov"
SELECTION_SUPERSEDED = "vocab_matched"


def _read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def grammar_name_from_custom_id(custom_id: str) -> str:
    return custom_id.split("-", 1)[0]


def load_requests(batch_dir: Path, glob: str, selection: str) -> dict[str, dict]:
    """Index examples-only request metadata by custom ID.

    OpenAI batch outputs do not echo request metadata, so the gold
    translation, the shot count and the coverage flag all come from the
    request files.
    """

    requests: dict[str, dict[str, Any]] = {}
    for path in sorted(batch_dir.glob(glob)):
        for record in _read_jsonl(path):
            custom_id = record.get("custom_id")
            metadata = (record.get("body") or {}).get("metadata") or {}
            if not isinstance(custom_id, str) or not metadata:
                continue
            covered = metadata.get("source_vocab_covered")
            requests[custom_id] = {
                "grammar": grammar_name_from_custom_id(custom_id),
                "k_shots": int(metadata["k_shots"]),
                "sample_id": metadata.get("sample_id"),
                "input_sentence": metadata.get("input_sentence"),
                "output_sentence": metadata.get("output_sentence"),
                "selection": metadata.get("exemplar_selection", selection),
                # Only the vocabulary-conditioned requests carry the flag;
                # for the random ones coverage is recomputed below.
                "covered": None if covered is None else covered == "True",
            }
    return requests


def exemplar_source_vocab(data_dir: Path, k_values: set[int]) -> dict[tuple, set]:
    """Source vocabulary of the first k pool sentences, per grammar.

    Mirrors ``main.shot_examples_from_samples`` for the random condition,
    whose exemplars are the same for every item of a grammar.
    """

    vocab: dict[tuple, set] = {}
    for path in sorted(data_dir.glob("shots_*.jsonl")):
        grammar = path.stem.removeprefix("shots_")
        shots = list(_read_jsonl(path))
        for k in k_values:
            words: set[str] = set()
            for shot in shots[:k]:
                sentence = shot.get("left_phonetic") or shot.get("left") or ""
                words |= set(str(sentence).split())
            vocab[(grammar, k)] = words
    return vocab


def score(batch_dir: Path, data_dir: Path) -> pd.DataFrame:
    """Score every examples-only output that joins to a known request."""

    requests = load_requests(batch_dir, VOCAB_MATCHED_GLOB, "vocab_matched")
    requests.update(load_requests(batch_dir, RANDOM_GLOB, "random"))
    if not requests:
        raise FileNotFoundError(f"No examples-only requests under {batch_dir}")

    random_ks = {
        meta["k_shots"] for meta in requests.values() if meta["covered"] is None
    }
    vocab = exemplar_source_vocab(data_dir, random_ks) if random_ks else {}

    rows: list[dict[str, Any]] = []
    for path in sorted(batch_dir.glob("*_output.jsonl")):
        for record in _read_jsonl(path):
            custom_id = record.get("custom_id")
            meta = requests.get(custom_id) if isinstance(custom_id, str) else None
            if meta is None:
                continue  # belongs to the grammar run, not an ablation
            answer = extract_answer(response_content(record))
            covered = meta["covered"]
            if covered is None:
                source_words = set(str(meta["input_sentence"] or "").split())
                pool = vocab.get((meta["grammar"], meta["k_shots"]), set())
                covered = bool(source_words and source_words <= pool)
            rows.append(
                {
                    "custom_id": custom_id,
                    "grammar": meta["grammar"],
                    "sample_id": meta["sample_id"],
                    "selection": meta["selection"],
                    "k_shots": meta["k_shots"],
                    "exact_match": bool(
                        answer is not None and answer == meta["output_sentence"]
                    ),
                    "covered": bool(covered),
                    "answered": answer is not None,
                }
            )

    if not rows:
        raise ValueError(
            "No examples-only outputs found. Download batch results into "
            f"{batch_dir} as batch_<id>_output.jsonl first."
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["custom_id"])


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """Accuracy overall, among covered items, and among uncovered items."""

    records: list[dict[str, Any]] = []
    for raw_key, group in scored.groupby(["selection", "k_shots"]):
        selection, k = cast(tuple[str, int], raw_key)
        covered = group[group["covered"]]
        uncovered = group[~group["covered"]]
        records.append(
            {
                "selection": selection,
                "k_shots": int(k),
                "n": len(group),
                "accuracy_pct": group["exact_match"].mean() * 100,
                "coverage_pct": group["covered"].mean() * 100,
                "covered_n": len(covered),
                "accuracy_given_covered_pct": (
                    covered["exact_match"].mean() * 100
                    if len(covered)
                    else float("nan")
                ),
                "uncovered_n": len(uncovered),
                "correct_uncovered": int(uncovered["exact_match"].sum()),
                "unparsed_n": int((~group["answered"]).sum()),
            }
        )
    return pd.DataFrame(records).sort_values(["selection", "k_shots"])


def grammar_conditions(batch_dir: Path, grammars: set[str]) -> pd.DataFrame:
    """Accuracy per k for the grammar run, restricted to ablation grammars.

    The grammar always supplies the vocabulary, so these cells carry no
    coverage constraint; the paper reports their coverage as 100.
    """

    rows = load_rows(batch_dir)
    rows = rows.assign(grammar=rows["custom_id"].map(grammar_name_from_custom_id))
    rows = rows[rows["grammar"].isin(grammars)]
    if rows.empty:
        raise ValueError("No grammar-condition rows share grammars with the ablation.")
    return (
        rows.groupby("k_shots")["exact_match"]
        .mean()
        .mul(100)
        .rename("accuracy_pct")
        .reset_index()
    )


def requested_coverage(batch_dir: Path, data_dir: Path) -> dict[tuple[str, int], float]:
    """Coverage per (selection, k) computed from the requests, not the outputs.

    Coverage is analytic: it depends only on which exemplars a prompt shows,
    so it is defined for cells whose batches have not come back yet. Those
    cells then print their coverage beside a --- accuracy.
    """

    requests = load_requests(batch_dir, VOCAB_MATCHED_GLOB, SELECTION_VME)
    requests.update(load_requests(batch_dir, RANDOM_GLOB, SELECTION_RANDOM))
    random_ks = {m["k_shots"] for m in requests.values() if m["covered"] is None}
    vocab = exemplar_source_vocab(data_dir, random_ks) if random_ks else {}

    totals: dict[tuple[str, int], list[int]] = {}
    for meta in requests.values():
        covered = meta["covered"]
        if covered is None:
            words = set(str(meta["input_sentence"] or "").split())
            pool = vocab.get((meta["grammar"], meta["k_shots"]), set())
            covered = bool(words and words <= pool)
        key = (meta["selection"], meta["k_shots"])
        slot = totals.setdefault(key, [0, 0])
        slot[0] += bool(covered)
        slot[1] += 1
    return {k: 100.0 * c / n for k, (c, n) in totals.items() if n}


def paper_table(
    scored: pd.DataFrame,
    grammar: pd.DataFrame,
    coverage: dict[tuple[str, int], float] | None = None,
) -> pd.DataFrame:
    """Build the table as the paper prints it: G+RE / RE / VME, each with coverage.

    Cells with no outputs yet are left as NaN accuracy. Coverage is still
    reported for them when the requests exist, since coverage is analytic
    and does not need model outputs.
    """

    coverage = coverage or {}
    summary = summarize(scored).set_index(["selection", "k_shots"])
    g = grammar.set_index("k_shots")["accuracy_pct"]

    ks = sorted(
        {int(k) for k in scored["k_shots"].unique()}
        | set(g.index)
        | {k for _, k in coverage}
    )
    records: list[dict[str, Any]] = []
    for k in ks:
        row: dict[str, Any] = {"k_shots": k}
        row["g_re_pct"] = float(g[k]) if k in g.index else float("nan")
        row["g_re_cov"] = 100.0 if k in g.index else float("nan")
        for label, selection in (("re", SELECTION_RANDOM), ("vme", SELECTION_VME)):
            if (selection, k) in summary.index:
                cell = summary.loc[(selection, k)]
                row[f"{label}_pct"] = float(cell["accuracy_pct"])
                row[f"{label}_cov"] = float(cell["coverage_pct"])
            else:
                row[f"{label}_pct"] = float("nan")
                row[f"{label}_cov"] = coverage.get((selection, k), float("nan"))
        records.append(row)
    return pd.DataFrame(records)


def format_paper_table(table: pd.DataFrame) -> str:
    """Render the paper table, with pending cells as --- beside their coverage."""

    def cell(acc: float, cov: float) -> str:
        if cov != cov:
            return "---"
        shown = "---" if acc != acc else f"{acc:.2f}"
        return f"{shown} ({cov:.2f})"

    lines = [f"{'k':>3} {'G+RE (cov)':>16} {'RE (cov)':>16} {'VME (cov)':>16}"]
    for _, r in table.iterrows():
        lines.append(
            f"{int(r['k_shots']):>3} "
            f"{cell(r['g_re_pct'], r['g_re_cov']):>16} "
            f"{cell(r['re_pct'], r['re_cov']):>16} "
            f"{cell(r['vme_pct'], r['vme_cov']):>16}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=BATCH_DIR)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "fewshot_exp"
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args()

    scored = score(args.batch_dir, args.data_dir)
    summary = summarize(scored)
    grammar = grammar_conditions(args.batch_dir, set(scored["grammar"].unique()))
    coverage = requested_coverage(args.batch_dir, args.data_dir)
    table = paper_table(scored, grammar, coverage)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.cache_dir / "vocab_matched_items.csv", index=False)
    summary.to_csv(args.cache_dir / "vocab_matched_summary.csv", index=False)
    table.to_csv(args.cache_dir / "paper_table.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("Paper table (accuracy %, coverage % in parentheses)")
    print(f"  VME = {SELECTION_VME}\n")
    print(format_paper_table(table))
    print("\n\nAll conditions scored")
    print(f"  ({SELECTION_SUPERSEDED} is the superseded 18-sentence-pool run)\n")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nWrote {args.cache_dir / 'paper_table.csv'}")


if __name__ == "__main__":
    main()
