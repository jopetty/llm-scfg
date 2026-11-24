"""Build examples-only batch files with vocabulary-conditioned exemplars.

The examples-only ablation in the paper shows the model the *first* ``k``
sentences of each grammar's held-out shot pool, independently of which
sentence it is asked to translate. That makes the vocabulary-coverage
ceiling almost perfectly binding: across the 23,398 items whose source
vocabulary is not fully covered by the exemplars, the model is never once
correct. Accuracy in that condition therefore rises with ``k`` mostly
because more of the source lexicon happens to appear, not because the
model translates better.

This script holds the evaluation targets fixed and changes only *which*
exemplars each prompt shows: for every item it greedily picks the ``k``
shot-pool sentences that cover as much of that item's source vocabulary as
possible. The retriever reads only the source side of the test item and the
source sides of the candidate exemplars, both of which are available at
inference time, so this is a realistic retrieval policy and not an oracle
that peeks at the gold translation.

Two things are worth knowing before spending inference budget on the
outputs.

First, the shot pool holds only 18 sentences per grammar, so coverage
saturates: greedy selection reaches 28.21% of items at ``k=8`` and cannot
exceed that at ``k=16``, because 28.21% is what *all 18* exemplars cover.
Testing coverage beyond that point would need a larger held-out pool, which
means regenerating data rather than reselecting from it.

Second, and the reason the capped version is still worth running: it
decouples coverage from ``k``. Vocabulary-conditioned ``k=4`` covers 27.16%
of items, almost exactly what random ``k=16`` covers (25.95%) with four
times the exemplars. Comparing those two cells asks whether coverage alone
accounts for examples-only accuracy, or whether seeing more pairs helps
independently of which words they contain.

Coverage by condition, computed over all 5,040 items:

      k   vocab-matched   first-k (paper)
      1           2.32%             0.00%
      2          14.72%             0.02%
      4          27.16%             0.97%
      8          28.21%             8.81%
     16          28.21%            25.95%

Usage:

    uv run python scripts/generate_vocab_matched_batchfiles.py \
        --model gpt-5.6-luna
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (  # noqa: E402
    BATCH_DIR,
    DEFAULT_BATCH_FILE_SIZE_MIB,
    _load_experiment_data,
    add_prompt_token_estimates,
    drop_rows_over_model_context,
    partition_batch_rows_by_size,
    prompt_agreement_metadata,
    sample_metadata,
    warn_for_large_prompts,
)
from scfg.prompt import ChatCompletionResponse, examples_only_prompt  # noqa: E402

log = logging.getLogger(__name__)

EXPERIMENT = "fewshot"
SELECTION_LABEL = "vocab_matched"
DEFAULT_K_VALUES = [1, 2, 4, 8, 16]


def source_tokens(sample: dict[str, Any]) -> set[str]:
    """Return the source-side vocabulary of one sample.

    Mirrors the tokenization used by the coverage analysis in
    ``scripts/figures/generate_examples_only_ablation.py``: whitespace
    splitting of the phonetic source string.
    """

    text = str(sample.get("left_phonetic") or sample.get("left") or "")
    return set(text.split())


def select_exemplars(
    target_vocab: set[str],
    shot_vocabs: list[set[str]],
    k: int,
) -> tuple[list[int], bool]:
    """Greedily choose ``k`` shot indices covering ``target_vocab``.

    Returns the chosen indices in ascending pool order together with whether
    the selection covers the target's source vocabulary completely. Indices
    are sorted rather than left in selection order so that the exemplars
    appear in the prompt in the same pool order the random condition uses;
    otherwise the most informative exemplar would always come first and
    position would be confounded with relevance.

    Maximum-coverage selection is NP-hard, so this is the standard greedy
    approximation. The selection is padded out to exactly ``k`` exemplars
    even once coverage is complete, which keeps the prompt budget matched to
    the corresponding random-selection cell.
    """

    if k <= 0:
        raise ValueError("k must be positive for the examples-only condition")

    remaining = set(target_vocab)
    available = list(range(len(shot_vocabs)))
    chosen: list[int] = []

    while len(chosen) < k and available:
        # Most newly covered target words wins; ties go to the earlier
        # pool index so the selection is deterministic.
        best = max(available, key=lambda i: (len(remaining & shot_vocabs[i]), -i))
        chosen.append(best)
        remaining -= shot_vocabs[best]
        available.remove(best)

    if len(chosen) < k:
        raise ValueError(
            f"Requested {k} exemplars but the shot pool holds only {len(shot_vocabs)}"
        )

    return sorted(chosen), not remaining


def build_requests(
    *,
    model: str,
    k_values: list[int],
    max_completion_tokens: int | None,
    n: int,
) -> dict[int, pd.DataFrame]:
    """Build one DataFrame of batch requests per ``k``."""

    experiment_data = _load_experiment_data(
        exp=EXPERIMENT, data_source="local", hf_repo_id=None
    )

    per_k: dict[int, list[pd.DataFrame]] = {k: [] for k in k_values}

    for grammar_name, data in experiment_data.items():
        samples = cast(list[dict[str, Any]], data["samples"])
        shots = cast(list[dict[str, Any]] | None, data.get("shots"))
        if not shots:
            raise ValueError(
                f"Grammar {grammar_name} has no dedicated shot pool; the "
                "vocabulary-conditioned condition must draw exemplars from a "
                "pool held out from the evaluated samples."
            )
        shot_pool: list[dict[str, Any]] = shots
        grammar = cast(dict[str, object], data["grammar"])
        agreement_metadata = prompt_agreement_metadata(grammar)
        shot_vocabs = [source_tokens(shot) for shot in shot_pool]

        log.info(
            "Selecting exemplars for %s: %s samples, %s candidate shots",
            grammar_name,
            len(samples),
            len(shots),
        )

        for k in k_values:
            df = pd.DataFrame(samples)

            selections = [
                select_exemplars(source_tokens(sample), shot_vocabs, k)
                for sample in samples
            ]
            df["exemplar_indices"] = [indices for indices, _ in selections]
            df["source_vocab_covered"] = [covered for _, covered in selections]

            df["prompt"] = df.apply(
                lambda row: examples_only_prompt(
                    grammar_str="",
                    sample=row["left_phonetic"],
                    agreement_metadata=agreement_metadata,
                    few_shot_examples=[
                        {
                            "input": str(
                                shot_pool[i].get("left_phonetic")
                                or shot_pool[i].get("left")
                                or ""
                            ),
                            "output": str(
                                shot_pool[i].get("right_phonetic")
                                or shot_pool[i].get("right")
                                or ""
                            ),
                        }
                        for i in row["exemplar_indices"]
                    ],
                ),
                axis=1,
            )

            df = add_prompt_token_estimates(df, model)
            df = drop_rows_over_model_context(
                df, model=model, grammar_name=grammar_name
            )
            warn_for_large_prompts(df, model=model, grammar_name=grammar_name)

            df["json"] = df.apply(
                lambda row: ChatCompletionResponse(
                    user_prompt=row["prompt"],
                    max_completion_tokens=max_completion_tokens,
                    n=n,
                    metadata={
                        **sample_metadata(
                            grammar_name=grammar_name,
                            sample_id=str(row.name),
                            row=row,
                            grammar=grammar,
                            k_shots=k,
                        ),
                        "exemplar_selection": SELECTION_LABEL,
                        "exemplar_indices": ",".join(
                            str(i) for i in row["exemplar_indices"]
                        ),
                        "source_vocab_covered": str(row["source_vocab_covered"]),
                    },
                ).to_openai_batched_json(
                    model=model,
                    custom_id=(f"{grammar_name}-vm-k{k}-sample-{row.name}"),
                ),
                axis=1,
            )

            per_k[k].append(df)

    out: dict[int, pd.DataFrame] = {}
    for k, frames in per_k.items():
        if not frames:
            raise ValueError(f"No samples found for experiment {EXPERIMENT}")
        out[k] = pd.concat(frames, ignore_index=True)
    return out


def write_batch_files(
    *,
    k: int,
    all_df: pd.DataFrame,
    model: str,
    max_filesize_mb: float,
) -> list[Path]:
    """Partition and write the batch files for one ``k``."""

    exp_batch_dir = BATCH_DIR / f"{EXPERIMENT}_exp"
    exp_batch_dir.mkdir(parents=True, exist_ok=True)

    max_file_size_bytes = int(max_filesize_mb * 1024 * 1024)
    partitioned = partition_batch_rows_by_size(
        all_df, max_file_size_bytes=max_file_size_bytes
    )
    num_files = len(partitioned)
    model_pathsafe_name = model.replace("/", "_")

    written: list[Path] = []
    for i, part_df in enumerate(partitioned):
        fname_hash = secrets.token_hex(3)

        # Stamp the file hash into each custom_id, as the main batch builder
        # does, so an output file can be traced back to its input file
        # without relying on metadata.
        part_df = part_df.copy()
        part_df["json"] = part_df["json"].apply(json.loads)
        part_df["json"] = part_df["json"].apply(
            lambda x: {
                **x,
                "custom_id": x["custom_id"].replace("-", f"-{fname_hash}-", 1),
            }
        )
        part_df["json"] = part_df["json"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

        fname = (
            f"inputs_{EXPERIMENT}_examples_only_vocabmatched_k{k}"
            f"_{model_pathsafe_name}_part{i + 1}_of_{num_files}_{fname_hash}.jsonl"
        )
        fpath = exp_batch_dir / fname
        log.info("Writing batch job to %s", fpath)
        with open(fpath, "w") as handle:
            for j in part_df["json"]:
                handle.write(f"{j}\n")

        written_size = fpath.stat().st_size
        if written_size > max_file_size_bytes:
            raise RuntimeError(
                f"Batch file {fpath} exceeds the {max_file_size_bytes:,}-byte "
                f"limit: {written_size:,} bytes"
            )
        written.append(fpath)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument(
        "--max-filesize-mb", type=float, default=DEFAULT_BATCH_FILE_SIZE_MIB
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    k_values = list(dict.fromkeys(args.k))
    if any(k <= 0 for k in k_values):
        raise ValueError(
            "The examples-only condition needs k > 0; with no grammar and no "
            "examples the prompt says nothing about the target language."
        )

    frames = build_requests(
        model=args.model,
        k_values=k_values,
        max_completion_tokens=args.max_completion_tokens,
        n=args.n,
    )

    print()
    print(f"{'k':>3} {'requests':>9} {'covered':>8} {'coverage':>9}  files")
    for k in k_values:
        df = frames[k]
        paths = write_batch_files(
            k=k,
            all_df=df,
            model=args.model,
            max_filesize_mb=args.max_filesize_mb,
        )
        covered = int(df["source_vocab_covered"].sum())
        pct = 100.0 * covered / len(df)
        names = ", ".join(p.name for p in paths)
        print(f"{k:>3} {len(df):>9,} {covered:>8,} {pct:>8.2f}%  {names}")


if __name__ == "__main__":
    main()
