"""Sample large held-out exemplar pools that make full coverage reachable.

The few-shot experiment ships an 18-sentence shot pool per grammar. That
caps vocabulary coverage of the examples-only condition at 28.21% even when
exemplars are chosen to cover the test item's source vocabulary as well as
possible: 28.21% is simply what all 18 pool sentences cover between them.

Probing showed the 16-exemplar budget is never what binds. Across grammars
spanning 30 to 8,010 lexical items, the fraction of items whose source
vocabulary greedy selection packs into 16 exemplars is *exactly* the
fraction whose every source word appears anywhere in the pool. Whenever the
words are present at all, 16 sentences suffice to carry them. Coverage is
therefore a pool-size question, and pool size scales with the lexicon:

    lexicon    pool 1,200   pool 3,000   pool 8,000
         30        100.0%            -            -
        810         93.3%       100.0%            -
      8,010         34.2%        56.7%        87.5%

So this script grows each grammar's pool until coverage stops being the
constraint, rather than using one size everywhere. Pools are written as
``bigshots_<grammar>.jsonl`` alongside the original ``shots_*.jsonl``.

Two safeguards matter more as the pool grows.

Leakage: a large pool eventually contains a sentence whose source side is
identical to an evaluation item's. A vocabulary-targeted retriever would
preferentially return exactly that sentence, since it covers the target
perfectly, and its exemplar line would hand over the gold translation. Any
pool sentence whose source string matches *any* evaluation source is
dropped, so no item can ever be shown its own answer.

Duplication: random sampling repeats itself, especially at low depth in
small grammars, and duplicate exemplars waste the prompt budget without
adding vocabulary. Pools are deduplicated by source string.

Sampling, not selection, is what costs: drawing 2,000 sentences from the
8,010-item grammar takes ~98s while scoring coverage over them takes ~0.2s.
Grammars are therefore processed in parallel across cores, each with its own
deterministic seed so results do not depend on scheduling order.

Example:
    uv run python scripts/generate_highcoverage_pools.py --target-coverage 0.99
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import DATA_DIR, deterministic_seed  # noqa: E402
from scfg.scfg import SCFG, SCFGParams  # noqa: E402

log = logging.getLogger(__name__)

EXP_DIR = DATA_DIR / "fewshot_exp"
POOL_PREFIX = "bigshots"
SEED_NAMESPACE = "fewshot_bigshots"
MAX_DEPTH = 5
K_REFERENCE = 16

# Candidate pruning. Selection only ever needs *some* pool sentence carrying
# a given word, so considering every sentence that contains a common word is
# wasted work: the posting list for a determiner can run to half the pool.
# Capping how many candidates each word contributes keeps selection linear in
# the target's vocabulary instead of in the pool size. Coverage is always
# recomputed exactly from the chosen exemplars, so a cap can only ever cost a
# little coverage, never overstate it.
CANDIDATES_PER_WORD = 300


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_of(sample: dict[str, Any]) -> str:
    return str(sample.get("left_phonetic") or sample.get("left") or "")


def build_postings(sources: list[str]) -> dict[str, list[int]]:
    postings: dict[str, list[int]] = defaultdict(list)
    for index, sentence in enumerate(sources):
        for word in set(sentence.split()):
            postings[word].append(index)
    return postings


def select_exemplars(
    target_vocab: set[str],
    postings: dict[str, list[int]],
    k: int,
    candidates_per_word: int = CANDIDATES_PER_WORD,
) -> tuple[list[int], bool]:
    """Greedily choose k pool indices covering ``target_vocab``.

    Returns the chosen indices in ascending pool order and whether they
    cover the target's source vocabulary completely.
    """

    candidate_words: dict[int, set[str]] = defaultdict(set)
    for word in target_vocab:
        for index in postings.get(word, ())[:candidates_per_word]:
            candidate_words[index].add(word)

    remaining = set(target_vocab)
    chosen: list[int] = []
    while len(chosen) < k and candidate_words:
        best = max(
            candidate_words,
            key=lambda i: (len(candidate_words[i] & remaining), -i),
        )
        gain = candidate_words[best] & remaining
        chosen.append(best)
        del candidate_words[best]
        if not gain and not remaining:
            # Coverage is already complete; keep padding to exactly k so the
            # prompt budget stays matched to the corresponding random cell.
            continue
        remaining -= gain
    return sorted(chosen), not remaining


def coverage_at_k(
    targets: list[set[str]], postings: dict[str, list[int]], k: int
) -> float:
    if not targets:
        return 0.0
    covered = sum(select_exemplars(t, postings, k)[1] for t in targets)
    return covered / len(targets)


def grow_pool(
    *,
    grammar_name: str,
    target_coverage: float,
    block_size: int,
    max_pool: int,
) -> tuple[list[dict[str, Any]], float]:
    """Sample a deduplicated, leakage-free pool until coverage plateaus."""

    grammar = json.loads((EXP_DIR / f"grammar_{grammar_name}.json").read_text())
    scfg = SCFG(SCFGParams.from_dict(grammar))
    rng = random.Random(deterministic_seed(SEED_NAMESPACE, grammar_name))

    samples = read_jsonl(EXP_DIR / f"samples_{grammar_name}.jsonl")
    eval_sources = {source_of(s) for s in samples}
    targets = [set(source_of(s).split()) for s in samples]

    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    coverage = 0.0
    stalls = 0

    while len(pool) < max_pool:
        before = len(pool)
        for _ in range(block_size // (MAX_DEPTH + 1) + 1):
            for depth in range(MAX_DEPTH + 1):
                sample = scfg.sample(rng=rng, min_depth=depth, max_depth=depth)
                source = source_of(sample)
                # Never let an item be shown its own source sentence, whose
                # exemplar line would carry the gold translation.
                if source in eval_sources or source in seen:
                    continue
                seen.add(source)
                sample["grammar_name"] = grammar_name
                pool.append(sample)

        gained = len(pool) - before
        sources = [source_of(s) for s in pool]
        coverage = coverage_at_k(targets, build_postings(sources), K_REFERENCE)
        log.debug(
            "  %s: pool=%d coverage@k=%d %.1f%%",
            grammar_name,
            len(pool),
            K_REFERENCE,
            100 * coverage,
        )
        if coverage >= target_coverage:
            break
        if gained == 0:
            # The grammar's sentence space is exhausted; more sampling cannot
            # add vocabulary, so stop rather than spin.
            stalls += 1
            if stalls >= 2:
                log.warning(
                    "  %s: sentence space exhausted at pool=%d, coverage %.1f%%",
                    grammar_name,
                    len(pool),
                    100 * coverage,
                )
                break
        else:
            stalls = 0

    return pool, coverage


def process_grammar(
    name: str,
    target_coverage: float,
    block_size: int,
    max_pool: int,
) -> tuple[str, int, int, float]:
    """Grow and write one grammar's pool. Runs in a worker process."""

    grammar = json.loads((EXP_DIR / f"grammar_{name}.json").read_text())
    n_words = int(grammar.get("n_words") or 0)
    pool, coverage = grow_pool(
        grammar_name=name,
        target_coverage=target_coverage,
        block_size=max(block_size, 2 * n_words),
        max_pool=max(max_pool, 5 * n_words),
    )
    out = EXP_DIR / f"{POOL_PREFIX}_{name}.jsonl"
    with out.open("w") as handle:
        for sample in pool:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return name, n_words, len(pool), coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-coverage", type=float, default=0.99)
    parser.add_argument("--block-size", type=int, default=2000)
    parser.add_argument("--max-pool", type=int, default=40000)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Grammars to sample in parallel; sampling is the bottleneck.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    names = [
        n.strip()
        for n in (EXP_DIR / "fewshot_grammars.txt").read_text().split()
        if n.strip()
    ]
    if args.only:
        names = [n for n in names if n in set(args.only)]

    log.info("Sampling pools for %d grammars on %d workers", len(names), args.workers)
    report: list[tuple[str, int, int, float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
        futures = {
            pool_exec.submit(
                process_grammar,
                name,
                args.target_coverage,
                args.block_size,
                args.max_pool,
            ): name
            for name in names
        }
        for done, future in enumerate(as_completed(futures), start=1):
            name, n_words, size, coverage = future.result()
            report.append((name, n_words, size, coverage))
            log.info(
                "[%d/%d] %s (%d items): pool=%d coverage@16 %.1f%%",
                done,
                len(names),
                name,
                n_words,
                size,
                100 * coverage,
            )
    report.sort(key=lambda r: (r[1], r[0]))

    print()
    print(f"{'grammar':>18} {'lexicon':>8} {'pool':>7} {'coverage@16':>12}")
    for name, n_words, size, coverage in report:
        print(f"{name:>18} {n_words:>8,} {size:>7,} {100 * coverage:>11.1f}%")
    worst = min(report, key=lambda r: r[3])
    print(f"\nlowest coverage: {worst[0]} at {100 * worst[3]:.1f}% (pool {worst[2]:,})")


if __name__ == "__main__":
    main()
