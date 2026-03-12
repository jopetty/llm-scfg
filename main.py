# Generating samples from scfgs

import json
import logging
import pathlib
import random
import secrets
from hashlib import blake2b
from typing import Dict, Union

import fire
import numpy as np
import pandas as pd
import pyrootutils

from scfg.prompt import ChatCompletionResponse, basic_prompt
from scfg.scfg import SCFG, CFGParams, SCFGParams
from scfg.utils import get_logger, set_all_seeds

Path = pathlib.Path

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


def deterministic_seed(*parts: object, base_seed: int = 42, modulus: int = 10_000) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return base_seed + (int.from_bytes(digest, "big") % modulus)


def create_orthography_data(
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
):
    """
    Generates grammars which vary the target orthography.
    """

    syllable_structure: str = "C*VC"
    g_sizes: list[int] = [7500, 10000]
    target_orthographies: list[str] = [
        "latin",
        "cyrillic",
        "yiddish",
    ]
    grammar_names: list[str] = []
    for orthography in target_orthographies:
        for g_size in g_sizes:
            for _ in range(n_grammars_per_size):
                g_seed = deterministic_seed("orthography", orthography, g_size, _)
                grammar_name = create_grammar(
                    rng_seed=g_seed,
                    syllable_structure_a=syllable_structure,
                    syllable_structure_b=syllable_structure,
                    head_initial_a=True,
                    head_initial_b=True,
                    spec_initial_a=True,
                    spec_initial_b=True,
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
                    orthography_b=orthography,
                )
                log.info(
                    f"Created grammar {grammar_name} with orthography={orthography}, seed={g_seed}"
                )
                generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=g_seed,
                    min_depth=0,
                    max_depth=max_depth,
                    n_samples_per_depth=n_sentences_per_depth,
                )
                grammar_names.append(grammar_name)

    with open(DATA_DIR / "orthography_grammars.txt", "w") as f:
        for name in grammar_names:
            f.write(f"{name}\n")


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
        (False, False),
    ]

    grammar_names: list[str] = []

    for hi_b, si_b in target_head_spec_params:
        for g_size in g_sizes:
            for _ in range(n_grammars_per_size):
                g_seed = deterministic_seed("wordorder", hi_b, si_b, g_size, _)
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


def create_agreement_data(
    grammar_sizes: list[int] = [25, 50, 100, 1_000, 5_000, 7_500, 10_000],
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
):
    """
    Generates grammars which vary overt agreement marking.
    """

    syllable_structure: str = "CVC"
    exp_name = "agreement"
    exp_dir: Path = DATA_DIR / f"{exp_name}_exp"
    exp_dir.mkdir(parents=True, exist_ok=True)

    grammar_names: list[str] = []
    configurations = [
        {"agreement_enabled_a": False, "agreement_enabled_b": False},
        {"agreement_enabled_a": False, "agreement_enabled_b": True},
        {"agreement_enabled_a": True, "agreement_enabled_b": False},
        {"agreement_enabled_a": True, "agreement_enabled_b": True},
    ]

    for config in configurations:
        for g_size in grammar_sizes:
            for index in range(n_grammars_per_size):
                g_seed = deterministic_seed(
                    "agreement",
                    g_size,
                    index,
                    tuple(sorted(config.items())),
                )
                grammar_name = create_grammar(
                    rng_seed=g_seed,
                    syllable_structure_a=syllable_structure,
                    syllable_structure_b=syllable_structure,
                    head_initial_a=True,
                    head_initial_b=True,
                    spec_initial_a=True,
                    spec_initial_b=True,
                    pro_drop_a=False,
                    pro_drop_b=False,
                    n_verbs=max(2, g_size // 5),
                    n_nouns=max(2, g_size // 5),
                    n_adjectives=max(1, g_size // 10),
                    n_propns=max(2, g_size // 10),
                    n_det_def=2,
                    n_det_indef=2,
                    n_prons=6,
                    n_comps=2,
                    orthography_a="latin",
                    orthography_b="latin",
                    exp_name=exp_name,
                    agreement_enabled_a=config["agreement_enabled_a"],
                    agreement_enabled_b=config["agreement_enabled_b"],
                )
                log.info(
                    "Created grammar %s with agreement_a=%s agreement_b=%s seed=%s",
                    grammar_name,
                    config["agreement_enabled_a"],
                    config["agreement_enabled_b"],
                    g_seed,
                )
                generate_samples(
                    grammar_name=grammar_name,
                    rng_seed=g_seed,
                    min_depth=0,
                    max_depth=max_depth,
                    n_samples_per_depth=n_sentences_per_depth,
                    exp_name=exp_name,
                )
                grammar_names.append(grammar_name)

    with open(exp_dir / f"{exp_name}_grammars.txt", "w") as f:
        for name in grammar_names:
            f.write(f"{name}\n")



def create_size_data(
    grammar_sizes: list[int] = [25, 50, 100, 1_000, 5_000, 7_500, 10_000],
    max_depth: int = 5,
    n_grammars_per_size: int = 2,
    n_sentences_per_depth: int = 20,
    exp_name: str = "size",
):
    """
    Generates grammars which vary in size.
    """
    syllable_structure: str = "C*VC"
    target_head_initial: bool = False
    
    exp_dir: Path = DATA_DIR / (exp_name + "_exp")
    exp_dir.mkdir(parents=True, exist_ok=True)

    grammar_names: list[str] = []
    for g_size in grammar_sizes:
        for _ in range(n_grammars_per_size):
            g_seed = deterministic_seed(exp_name, g_size, _)
            grammar_name = create_grammar(
                rng_seed=g_seed,
                syllable_structure_a=syllable_structure,
                syllable_structure_b=syllable_structure,
                head_initial_a=True,
                head_initial_b=target_head_initial,
                spec_initial_a=True,
                spec_initial_b=True,
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
                exp_name=exp_name
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
                exp_name=exp_name,
            )
            grammar_names.append(grammar_name)
    
    with open(exp_dir / f"{exp_name}_grammars.txt", "w") as f:
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
            g_seed = deterministic_seed("complexity", g_size, _)
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
            g_seed = deterministic_seed("large_complexity", g_size, _)
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
    orthography_a: str = "latin",
    orthography_b: str = "latin",
    exp_name: str | None = None,
    agreement_enabled_a: bool = False,
    agreement_enabled_b: bool = False,
    latent_gender_a: bool = False,
    latent_gender_b: bool = False,
    realize_gender_a: bool = False,
    realize_gender_b: bool = False,
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
        orthography=orthography_a,
        agreement_enabled=agreement_enabled_a,
        latent_gender=latent_gender_a,
        realize_gender=realize_gender_a,
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
        orthography=orthography_b,
        agreement_enabled=agreement_enabled_b,
        latent_gender=latent_gender_b,
        realize_gender=realize_gender_b,
    )
    params = SCFGParams(a=a_params, b=b_params)

    out_dir: Path = DATA_DIR
    if exp_name is not None:
        out_dir /= (exp_name + "_exp")
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"grammar_{params.name}.json", "w") as f:
        json.dump(params.to_dict(), f, indent=2, ensure_ascii=False)

    log.info(f"Grammar saved to {out_dir / f'grammar_{params.name}.json'}")

    if params.name is None:
        raise ValueError("Grammar name is None")
    return str(params.name)


def generate_samples(
    grammar_name: str,
    rng_seed: int = 42,
    min_depth: int = 0,
    max_depth: int = 10,
    n_samples_per_depth: int = 2,
    exp_name: str | None = None,
):
    filepath = f"grammar_{grammar_name}.json"
    grammar_dir: Path = DATA_DIR
    if exp_name is not None:
        grammar_dir /= (exp_name + "_exp")
    with open(grammar_dir / filepath, "r") as f:
        data = json.load(f)

    log.info(f"Loaded grammar from {grammar_dir / filepath}")

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

    out_dir: Path = DATA_DIR
    if exp_name is not None:
        out_dir /= (exp_name + "_exp")
        out_dir.mkdir(parents=True, exist_ok=True)

    # Save samples to a file
    samples_filepath = out_dir / f"samples_{params.name}.jsonl"
    with open(samples_filepath, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


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
    agreement_metadata = grammar.get("agreement_metadata")
    n_words = grammar["n_words"]
    n_rules = grammar["n_rules"]

    df = pd.DataFrame(samples)

    if prompt_type == "basic":
        prompt_func = basic_prompt
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    df["prompt"] = df.apply(
        lambda row: prompt_func(
            grammar_str=grammar_str,
            sample=row["left_phonetic"],
            agreement_metadata=agreement_metadata,
        ),
        axis=1,
    )
    df["json"] = df.apply(
        lambda row: ChatCompletionResponse(
            user_prompt=row["prompt"],
            max_completion_tokens=max_completion_tokens,
            n=n,
            metadata={
                "grammar_name": grammar_name,
                "sample_id": str(row.name),
                "depth": str(row["depth"]),
            },
        ).to_openai_batched_json(
            model=model, custom_id=f"{grammar_name}-sample-{row.name}"
        ),
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
    exp: str,
    prompt_type: str = "basic",
    model: str = "gpt-5-nano",
    max_completion_tokens: int | None = None,
    n: int = 1,
    max_filesize_mb: int = 200,
):
    """
    Generates a single batchfile for multiple grammars.
    """

    # experiment_name = grammar_list.split(".")[0]

    exp_dir = PROJECT_ROOT / DATA_DIR / (exp + "_exp")
    exp_batch_dir = BATCH_DIR / (exp + "_exp")
    grammar_list_fname = exp + "_grammars.txt"
    grammar_list_filepath = exp_dir / grammar_list_fname

    exp_batch_dir.mkdir(parents=True, exist_ok=True)

    grammar_names: list[str] = []
    with open(grammar_list_filepath, "r") as f:
        for line in f:
            grammar_name = line.strip()
            if grammar_name:
                grammar_names.append(grammar_name)

    all_samples = []
    for grammar_name in grammar_names:
        samples = []
        with open(exp_dir / f"samples_{grammar_name}.jsonl", "r") as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)

        grammar_path = exp_dir / f"grammar_{grammar_name}.json"
        with open(grammar_path, "r") as f:
            grammar = json.load(f)
        grammar_str = grammar["grammar_str"]
        agreement_metadata = grammar.get("agreement_metadata")
        n_words = grammar["n_words"]
        n_rules = grammar["n_rules"]

        df = pd.DataFrame(samples)

        if prompt_type == "basic":
            prompt_func = basic_prompt
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        df["prompt"] = df.apply(
            lambda row: prompt_func(
                grammar_str=grammar_str,
                sample=row["left_phonetic"],
                agreement_metadata=agreement_metadata,
            ),
            axis=1,
        )
        df["json"] = df.apply(
            lambda row: ChatCompletionResponse(
                user_prompt=row["prompt"],
                max_completion_tokens=max_completion_tokens,
                n=n,
                metadata={
                    "grammar_name": grammar_name,
                    "sample_id": str(row.name),
                    "depth": str(row["depth"]),
                }
                if model.startswith("gpt")
                else None,
            ).to_openai_batched_json(
                model=model, custom_id=f"{grammar_name}-sample-{row.name}"
            ),
            axis=1,
        )

        all_samples.append(df)

    all_df = pd.concat(all_samples, ignore_index=True)

    # if the all_df["json"] column entries as strings exceed max_filesize_mb,
    # split into multiple files
    total_size_bytes = (
        all_df["json"].apply(lambda x: len((x + "-aaaaaa\n").encode("utf-8"))).sum()
    )

    total_size_mb = total_size_bytes / (1024 * 1024) * 1.05  # add 10% buffer

    # num_files = minimum number of files needed
    num_files = max(1, int(total_size_mb // max_filesize_mb) + 1)


    # partition all_df into num_files parts
    partitioned_dfs = np.array_split(all_df, num_files)

    for i, part_df in enumerate(partitioned_dfs):
        fname_hash: str = secrets.token_hex(3)

        # Add fname_hash to each custom_id so we can retrieve the
        # input file without metadata
        part_df["json"] = part_df["json"].apply(
            lambda x: json.loads(x)
        )  # parse json string
        part_df["json"] = part_df["json"].apply(
            lambda x: {
                **x,
                "custom_id": f"{x['custom_id'].replace('-', f'-{fname_hash}-', 1)}",
            }
        )
        part_df["json"] = part_df["json"].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

        model_pathsafe_name: str = model.replace("/", "_")
        base_fname: str = f"inputs_{exp}_{model_pathsafe_name}_part{i+1}_of_{num_files}"
        fname: str = f"{base_fname}_{fname_hash}.jsonl"
        fpath: Path = exp_batch_dir / fname
        log.info(f"Writing batch job to {fpath}")

        with open(fpath, "w") as f:
            for j in part_df["json"]:
                f.write(f"{j}\n")


def demo():
    a_params = CFGParams.english()
    b_params = CFGParams.german()
    scfg_params = SCFGParams(a_params, b_params)
    scfg = SCFG(scfg_params)
    print(scfg.as_cfg)


def demo_random():
    a_params = CFGParams()
    b_params = CFGParams(orthography="cyrillic")
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
            "exp_orthography": create_orthography_data,
            "exp_agreement": create_agreement_data,
            "exp_size": create_size_data,
            # TODO: add hash id to input files, attach it to requiest custom id
            # to make grammar identificaiton easier for gemini
        }
    )
