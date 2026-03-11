from __future__ import annotations

import json
import random
from pathlib import Path
import sys

import fire
import pyrootutils

PROJECT_ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scfg.scfg import CFGParams, SCFG, SCFGParams


def _load_params(
    grammar: str | None,
    latent_gender: bool,
    realize_gender_a: bool,
    realize_gender_b: bool,
) -> SCFGParams:
    if grammar is None:
        return SCFGParams(
            a=CFGParams(
                agreement_enabled=True,
                avg_syllables_per_word=1.2,
                syllable_max=2,
                latent_gender=latent_gender,
                realize_gender=realize_gender_a,
                verbs=4,
                nouns=4,
                propns=3,
                prons=2,
                adjs=2,
                det_def=1,
                det_indef=1,
                comps=1,
                orthography="latin",
                rng_seed=41,
            ),
            b=CFGParams(
                agreement_enabled=True,
                avg_syllables_per_word=1.2,
                syllable_max=2,
                latent_gender=latent_gender,
                realize_gender=realize_gender_b,
                verbs=4,
                nouns=4,
                propns=3,
                prons=2,
                adjs=2,
                det_def=1,
                det_indef=1,
                comps=1,
                orthography="latin",
                rng_seed=42,
            ),
        )

    grammar_path = Path(grammar)
    if not grammar_path.exists():
        candidate = Path("data") / f"grammar_{grammar}.json"
        if candidate.exists():
            grammar_path = candidate
        else:
            raise FileNotFoundError(f"Could not find grammar at {grammar} or {candidate}")

    with open(grammar_path) as handle:
        data = json.load(handle)
    return SCFGParams.from_dict(data)


def main(
    grammar: str | None = None,
    n_samples: int = 5,
    min_depth: int = 0,
    max_depth: int = 2,
    seed: int = 42,
    latent_gender: bool = False,
    realize_gender_a: bool = False,
    realize_gender_b: bool = False,
):
    params = _load_params(
        grammar,
        latent_gender=latent_gender,
        realize_gender_a=realize_gender_a,
        realize_gender_b=realize_gender_b,
    )
    scfg = SCFG(params)
    rng = random.Random(seed)

    print("=== Grammar ===")
    print(params.to_dict()["grammar_str"])
    print()

    agreement = params.to_dict().get("agreement_metadata", {})
    if agreement.get("enabled"):
        print("=== Agreement Metadata ===")
        print(json.dumps(agreement, indent=2, ensure_ascii=False))
        print()

    print("=== Samples ===")
    for index in range(n_samples):
        sample = scfg.sample(min_depth=min_depth, max_depth=max_depth, rng=rng)
        print(f"[{index + 1}] left:  {sample['left_phonetic']}")
        print(f"    right: {sample['right_phonetic']}")
        if "subject_features" in sample:
            print(
                "    agreement:",
                json.dumps(
                    {
                        "subject_features": sample["subject_features"],
                        "verb_features": sample["verb_features"],
                        "agreement_ok": sample["agreement_ok"],
                    },
                    ensure_ascii=False,
                ),
            )
        print()


if __name__ == "__main__":
    fire.Fire(main)
