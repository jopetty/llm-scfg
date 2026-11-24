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

from generate_fewshot_accuracy import extract_answer, response_content  # noqa: E402

BATCH_DIR = PROJECT_ROOT / "batches" / "fewshot_exp"
CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache" / "vocab-matched"

# Matches both pool variants: "..._vocabmatched_k8_..." (the shipped
# 18-sentence pool) and "..._vocabmatched_highcov_k8_..." (large pools).
# The random condition's "..._examples_only_k8_..." does not match.
VOCAB_MATCHED_GLOB = "inputs_*_examples_only_vocabmatched*_k*.jsonl"
RANDOM_GLOB = "inputs_*_examples_only_k*.jsonl"


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

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.cache_dir / "vocab_matched_items.csv", index=False)
    summary.to_csv(args.cache_dir / "vocab_matched_summary.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nWrote {args.cache_dir / 'vocab_matched_summary.csv'}")


if __name__ == "__main__":
    main()
