# Generating samples from scfgs

import json
import logging
import random
from itertools import product
from typing import Dict, Union

import fire
import pyrootutils

from scfg.scfg import SCFG, CFGParams, SCFGParams
from scfg.utils import get_logger, set_all_seeds

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)
DATA_DIR = PROJECT_ROOT / "data"


def create_grammars_and_samples(
    sizes: list[int] = [5, 10, 50, 100],
    grammars_per_size: int = 2,
    rng_seed: int = 80,
    n_samples: int = 200,
):
    rng = random.Random(rng_seed)
    for s in sizes:
        for _ in range(grammars_per_size):
            # enumerate choices for head_initial, spec_initial, pro_drop
            for ha, hb, sa, sb, pa, pb in product([True, False], repeat=6):
                g_seed = rng.randint(0, 10000)
                grammar_name = create_grammar(
                    rng_seed=g_seed,
                    head_initial_a=ha,
                    head_initial_b=hb,
                    spec_initial_a=sa,
                    spec_initial_b=sb,
                    pro_drop_a=pa,
                    pro_drop_b=pb,
                    n_verbs=s,
                    n_nouns=s,
                    n_adjectives=s,
                    n_propns=max(2, s // 5),
                    n_det_def=max(2, s // 5),
                    n_det_indef=max(2, s // 5),
                    n_prons=max(2, s // 5),
                    n_comps=max(2, s // 5),
                )
                generate_samples(
                    filepath=DATA_DIR / f"grammar_{grammar_name}.json",
                    rng_seed=g_seed,
                    n_samples=n_samples,
                    min_depth=0,
                    max_depth=10,
                )


def create_grammar(
    rng_seed: int = 80,
    syllable_structure_a: str | None = None,
    syllable_structure_b: str | None = None,
    head_initial_a: bool = True,
    head_initial_b: bool = True,
    spec_initial_a: bool = True,
    spec_initial_b: bool = True,
    pro_drop_a: bool = False,
    pro_drop_b: bool = False,
    n_verbs: int = 10,
    n_nouns: int = 10,
    n_adjectives: int = 10,
    n_propns: int = 5,
    n_det_def: int = 2,
    n_det_indef: int = 2,
    n_prons: int = 2,
    n_comps: int = 2,
) -> str:
    set_all_seeds(rng_seed)

    a_params = CFGParams(
        rng_seed=rng_seed,
        syllable_structure=syllable_structure_a,
        head_initial=head_initial_a,
        spec_initial=spec_initial_a,
        pro_drop=pro_drop_a,
        verbs=n_verbs,
        nouns=n_nouns,
        adjs=n_adjectives,
        propns=n_propns,
        det_def=n_det_def,
        det_indef=n_det_indef,
        prons=n_prons,
        comps=n_comps,
    )
    b_params = CFGParams(
        rng_seed=rng_seed + 1,
        syllable_structure=syllable_structure_b,
        head_initial=head_initial_b,
        spec_initial=spec_initial_b,
        pro_drop=pro_drop_b,
        verbs=n_verbs,
        nouns=n_nouns,
        adjs=n_adjectives,
        propns=n_propns,
        det_def=n_det_def,
        det_indef=n_det_indef,
        prons=n_prons,
        comps=n_comps,
    )
    params = SCFGParams(a=a_params, b=b_params)

    with open(DATA_DIR / f"grammar_{params.name}.json", "w") as f:
        json.dump(params.to_dict(), f, indent=2)

    log.info(f"Grammar saved to {DATA_DIR / f'grammar_{params.name}.json'}")

    return params.name


def generate_samples(
    filepath: str = "grammar.json",
    rng_seed: int = 42,
    n_samples: int = 2,
    min_depth: int = 0,
    max_depth: int = 10,
):
    with open(DATA_DIR / filepath, "r") as f:
        data = json.load(f)

    log.info(f"Loaded grammar from {DATA_DIR / filepath}")

    params = SCFGParams.from_dict(data)
    scfg = SCFG(params)

    rng = random.Random(rng_seed)

    samples: list[Dict[str, Union[str, int]]] = []

    for _ in range(n_samples):
        sample = scfg.sample(rng=rng, min_depth=min_depth, max_depth=max_depth)
        samples.append(sample)

    for s in samples:
        s["grammar_name"] = params.name
        s["min_depth"] = min_depth
        s["max_depth"] = max_depth
        s["rng_seed"] = rng_seed

    # Save samples to a file
    samples_filepath = DATA_DIR / f"samples_{params.name}.jsonl"
    with open(samples_filepath, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def load_grammar(filepath: str = "grammar.json", rng_seed: int = 42):
    with open(DATA_DIR / filepath, "r") as f:
        data = json.load(f)

    log.info(f"Loaded grammar from {DATA_DIR / filepath}")

    params = SCFGParams.from_dict(data)
    scfg = SCFG(params)

    rng = random.Random(rng_seed)
    for _ in range(2):
        sample = scfg.sample(rng=rng)
        print(sample["left_phonetic"])
        print(sample["right_phonetic"], "\n")


if __name__ == "__main__":
    fire.Fire(
        {
            "create_grammar": create_grammar,
            "load_grammar": load_grammar,
            "generate_samples": generate_samples,
            "cgs": create_grammars_and_samples,
        }
    )
