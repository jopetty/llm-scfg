# Generating samples from scfgs

import json
import logging
import random
from itertools import product
from typing import Dict, Union

import fire
import pandas as pd
import pyrootutils

from scfg.prompt import ChatCompletionResponse, basic_prompt
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
BATCH_DIR = PROJECT_ROOT / "batches"


def create_wordorder_data(
    max_depth: int = 5,
    n_grammars_per_size: int = 1,
    n_sentences_per_depth: int = 20,
):
    """
    Generates grammars which systematically vary word order parameters.
    """

    syllable_structure: str = "C*VC"
    g_sizes: list[int] = [7500, 10000]
    source_head_spec_params: tuple[bool, bool] = (True, True)
    target_head_spec_params: list[tuple[bool, bool]] = [
        (True, True),
        (False, True),
        (False, False)
    ]

    grammar_names: list[str] = []

    for (hi_b, si_b) in target_head_spec_params:
        for g_size in g_sizes:
            for _ in range(n_grammars_per_size):
                g_seed = 42 + hash((hi_b, si_b, g_size, _)) % 10000
                grammar_name = create_grammar(
                    rng_seed=g_seed,
                    syllable_structure_a=syllable_structure,
                    syllable_structure_b=syllable_structure,
                    head_initial_a=source_head_spec_params[0],
                    head_initial_b=hi_b,
                    spec_initial_a=source_head_spec_params[1],
                    spec_initial_b=si_b,
                    pro_drop_a=False,
                    pro_drop_b=False,
                    n_verbs=g_size // 5,
                    n_nouns=g_size // 5,
                    n_adjectives=g_size // 5,
                    n_propns=max(2, g_size // 5),
                    n_det_def=2,
                    n_det_indef=2,
                    n_prons=2,
                    n_comps=2,
                )
                log.info(
                    f"Created grammar {grammar_name} with hi_b={hi_b}, si_b={si_b}, seed={g_seed}"
                )
                generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=g_seed,
                    min_depth=0,
                    max_depth=max_depth,
                    n_samples_per_depth=n_sentences_per_depth,
                )
                grammar_names.append(grammar_name)

    with open(DATA_DIR / "wordorder_grammars.txt", "w") as f:
        for name in grammar_names:
            f.write(f"{name}\n")



def create_complexity_data(
    grammar_sizes: list[int] = [25, 50, 100, 500, 1000],
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
):
    """
    Generates grammars to test how grammatical and sentential complexity
    affect transduction performance.

    Vary:
        - grammar size (number of words): [25, 50, 100, 500, 1000]
        - clause depth: [0, 1, 2, 3, 4, 5]
    
    Fixed:
        - all grammar hyperparameters (head-initial, spec-initial)
        - all lexical parameters (syllable structure)
    """

    syllable_structure = "CVC"
    head_initial = True
    spec_initial = True
    pro_drop = False

    grammar_names: list[str] = []

    for g_size in grammar_sizes:
        for _ in range(n_grammars_per_size):
            g_seed = 42 + hash((g_size, _)) % 10000
            grammar_name = create_grammar(
                rng_seed=g_seed,
                syllable_structure_a=syllable_structure,
                syllable_structure_b=syllable_structure,
                head_initial_a=head_initial,
                head_initial_b=head_initial,
                spec_initial_a=spec_initial,
                spec_initial_b=spec_initial,
                pro_drop_a=pro_drop,
                pro_drop_b=pro_drop,
                n_verbs=g_size // 5,
                n_nouns=g_size // 5,
                n_adjectives=g_size // 5,
                n_propns=max(2, g_size // 5),
                n_det_def=2,
                n_det_indef=2,
                n_prons=2,
                n_comps=2,
            )
            log.info(
                f"Created grammar {grammar_name} with g_size={g_size}, seed={g_seed}"
            )
            generate_samples(
                grammar_name=grammar_name,
                rng_seed=g_seed,
                min_depth=0,
                max_depth=max_depth,
                n_samples_per_depth=n_sentences_per_depth,
            )
            grammar_names.append(grammar_name)

    with open(DATA_DIR / "complexity_grammars.txt", "w") as f:
        for name in grammar_names:
            f.write(f"{name}\n")


def create_large_complexity_data(
    grammar_sizes: list[int] = [1500, 2500, 10000],
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
):
    """
    Generates grammars to test how grammatical and sentential complexity
    affect transduction performance.

    Vary:
        - grammar size (number of words): [1500, 2500, 10000]
        - clause depth: [0, 1, 2, 3, 4, 5]
    
    Fixed:
        - all grammar hyperparameters (head-initial, spec-initial)
        - all lexical parameters (syllable structure)
    """

    syllable_structure = "CVC"
    head_initial = True
    spec_initial = True
    pro_drop = False

    grammar_names: list[str] = []

    for g_size in grammar_sizes:
        for _ in range(n_grammars_per_size):
            g_seed = 42 + hash((g_size, _)) % 10000
            grammar_name = create_grammar(
                rng_seed=g_seed,
                syllable_structure_a=syllable_structure,
                syllable_structure_b=syllable_structure,
                head_initial_a=head_initial,
                head_initial_b=head_initial,
                spec_initial_a=spec_initial,
                spec_initial_b=spec_initial,
                pro_drop_a=pro_drop,
                pro_drop_b=pro_drop,
                n_verbs=g_size // 5,
                n_nouns=g_size // 5,
                n_adjectives=g_size // 5,
                n_propns=max(2, g_size // 5),
                n_det_def=2,
                n_det_indef=2,
                n_prons=2,
                n_comps=2,
            )
            log.info(
                f"Created grammar {grammar_name} with g_size={g_size}, seed={g_seed}"
            )
            generate_samples(
                grammar_name=grammar_name,
                rng_seed=g_seed,
                min_depth=0,
                max_depth=max_depth,
                n_samples_per_depth=n_sentences_per_depth,
            )
            grammar_names.append(grammar_name)

    with open(DATA_DIR / "complexity_grammars_large.txt", "w") as f:
        for name in grammar_names:
            f.write(f"{name}\n")

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
    grammar_name: str,
    rng_seed: int = 42,
    min_depth: int = 0,
    max_depth: int = 10,
    n_samples_per_depth: int = 2,
):
    filepath = f"grammar_{grammar_name}.json"
    with open(DATA_DIR / filepath, "r") as f:
        data = json.load(f)

    log.info(f"Loaded grammar from {DATA_DIR / filepath}")

    params = SCFGParams.from_dict(data)
    scfg = SCFG(params)

    rng = random.Random(rng_seed)

    samples: list[Dict[str, Union[str, int]]] = []

    depths = list(range(min_depth, max_depth + 1))
    for d in depths:
        for _ in range(n_samples_per_depth):
            sample = scfg.sample(rng=rng, min_depth=d, max_depth=d)
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


def generate_batchfile(
    grammar_name: str,
    prompt_type: str = "basic",
    model: str = "o4-mini",
    max_completion_tokens: int | None = None,
    n: int = 1,
):
    samples = []
    with open(DATA_DIR / f"samples_{grammar_name}.jsonl", "r") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)

    grammar_path = DATA_DIR / f"grammar_{grammar_name}.json"
    with open(grammar_path, "r") as f:
        grammar = json.load(f)
    grammar_str = grammar["grammar_str"]
    n_words = grammar["n_words"]
    n_rules = grammar["n_rules"]

    df = pd.DataFrame(samples)

    if prompt_type == "basic":
        prompt_func = basic_prompt
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    df["prompt"] = df.apply(
        lambda row: prompt_func(grammar_str=grammar_str, sample=row["left_phonetic"]),
        axis=1,
    )
    df["json"] = df.apply(
        lambda row: ChatCompletionResponse(
            user_prompt=row["prompt"],
            max_completion_tokens=max_completion_tokens,
            n=n,
            metadata={
                "input_sentence": row["left_phonetic"],
                "output_sentence": row["right_phonetic"],
                "grammar_name": grammar_name,
                "n_words": str(n_words),
                "n_rules": str(n_rules),
                "model": model,
                "depth": str(row["depth"]),
            },
        ).to_openai_batched_json(model=model, custom_id=f"request-{row.name}"),
        axis=1,
    )

    model_pathsafe_name = model.replace("/", "_")
    batch_jsonl_filename = f"inputs_{grammar_name}_{model_pathsafe_name}.jsonl"
    batch_jsonl_path = BATCH_DIR / batch_jsonl_filename
    log.info(f"Writing batch job to {batch_jsonl_path}")

    with open(batch_jsonl_path, "w") as f:
        for j in df["json"]:
            f.write(f"{j}\n")


def generate_experiment_batchfile(
    grammar_list: str,
    prompt_type: str = "basic",
    model: str = "gpt-5-nano",
    max_completion_tokens: int | None = None,
    n: int = 1,
):
    """
    Generates a single batchfile for multiple grammars.
    """

    experiment_name = grammar_list.split(".")[0]
    grammar_list_filepath = DATA_DIR / grammar_list

    grammar_names: list[str] = []
    with open(grammar_list_filepath, "r") as f:
        for line in f:
            grammar_name = line.strip()
            if grammar_name:
                grammar_names.append(grammar_name)

    all_samples = []
    for grammar_name in grammar_names:
        samples = []
        with open(DATA_DIR / f"samples_{grammar_name}.jsonl", "r") as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)

        grammar_path = DATA_DIR / f"grammar_{grammar_name}.json"
        with open(grammar_path, "r") as f:
            grammar = json.load(f)
        grammar_str = grammar["grammar_str"]
        n_words = grammar["n_words"]
        n_rules = grammar["n_rules"]

        df = pd.DataFrame(samples)

        if prompt_type == "basic":
            prompt_func = basic_prompt
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        df["prompt"] = df.apply(
            lambda row: prompt_func(
                grammar_str=grammar_str, sample=row["left_phonetic"]
            ),
            axis=1,
        )
        df["json"] = df.apply(
            lambda row: ChatCompletionResponse(
                user_prompt=row["prompt"],
                max_completion_tokens=max_completion_tokens,
                n=n,
                metadata={
                    "input_sentence": row["left_phonetic"],
                    "output_sentence": row["right_phonetic"],
                    "grammar_name": grammar_name,
                    "n_words": str(n_words),
                    "n_rules": str(n_rules),
                    "model": model,
                    "depth": str(row["depth"]),
                } if model.startswith("gpt") else None,
            ).to_openai_batched_json(model=model, custom_id=f"{grammar_name}-request-{row.name}"),
            axis=1,
        )

        all_samples.append(df)

    all_df = pd.concat(all_samples, ignore_index=True)

    model_pathsafe_name = model.replace("/", "_")
    batch_jsonl_filename = f"inputs_{experiment_name}_{model_pathsafe_name}.jsonl"
    batch_jsonl_path = BATCH_DIR / batch_jsonl_filename
    log.info(f"Writing batch job to {batch_jsonl_path}")

    with open(batch_jsonl_path, "w") as f:
        for j in all_df["json"]:
            f.write(f"{j}\n")

def demo():
    a_params = CFGParams.english()
    b_params = CFGParams.german()
    scfg_params = SCFGParams(a_params, b_params)
    scfg = SCFG(scfg_params)
    print(scfg.as_cfg)


def demo_random():
    a_params = CFGParams()
    b_params = CFGParams(orthography="yiddish")
    scfg_params = SCFGParams(a_params, b_params)
    scfg = SCFG(scfg_params)
    print(scfg.as_cfg)


if __name__ == "__main__":
    fire.Fire(
        {
            # Core functionality
            "create_grammar": create_grammar,
            "gen_samples": generate_samples,
            "gen_batchfile": generate_batchfile,
            "gen_exp_batchfile": generate_experiment_batchfile,
            
            # Demos
            "demo": demo,
            "demo_random": demo_random,

            # Experiments
            "exp_complexity": create_complexity_data,
            "exp_large_complexity": create_large_complexity_data,
            "exp_wordorder": create_wordorder_data,
        }
    )
