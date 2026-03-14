from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fire
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BATCH_DIR = PROJECT_ROOT / "batches"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "cache" / "error-analysis"

ANSWER_RE = re.compile(
    r"final\s*answer\s*(?::|-|—)?\s*(?:is\s*)?([^\n]+)", re.IGNORECASE | re.DOTALL
)
AGREEMENT_CUSTOM_ID_RE = re.compile(
    r"^(?P<grammar_name>[0-9a-f]+)-(?P<input_hash>[0-9a-f]+)-sample-(?P<sample_id>\d+)$"
)


@dataclass(frozen=True)
class ExperimentSpec:
    exp: str
    dataset: str
    batch_dir: Path
    data_dir: Path


STANDARD_EXPERIMENTS = (
    ExperimentSpec(
        exp="wordorder",
        dataset="wordorder_exp",
        batch_dir=BATCH_DIR / "word_order",
        data_dir=DATA_DIR / "wordorder_exp",
    ),
    ExperimentSpec(
        exp="size",
        dataset="size_exp",
        batch_dir=BATCH_DIR / "size_exp",
        data_dir=DATA_DIR / "size_exp",
    ),
    ExperimentSpec(
        exp="orthography",
        dataset="orthography_exp",
        batch_dir=BATCH_DIR / "orthography_exp",
        data_dir=DATA_DIR / "orthography_exp",
    ),
    ExperimentSpec(
        exp="wordorder",
        dataset="wordorder_large_exp",
        batch_dir=BATCH_DIR / "wordorder_large_exp",
        data_dir=DATA_DIR / "wordorder_large_exp",
    ),
    ExperimentSpec(
        exp="orthography",
        dataset="orthography_large_exp",
        batch_dir=BATCH_DIR / "orthography_large_exp",
        data_dir=DATA_DIR / "orthography_large_exp",
    ),
)

OUTPUT_COLUMNS = [
    "exp",
    "custom_id",
    "batch_file",
    "batch_id",
    "model",
    "fuzzy_model",
    "model_response",
    "model_answer",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]
INPUT_COLUMNS = [
    "custom_id",
    "input_file",
    "fuzzy_model",
    "grammar_name",
    "sample_id",
    "input_sentence",
    "output_sentence",
    "depth",
    "n_words",
    "n_rules",
]
SAMPLE_COLUMNS = ["grammar_name", "sample_id", "input_sentence", "output_sentence"]

CYRILLIC_RE = re.compile(r"[а-яА-Я]")
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
HEBREW_DIACRITIC_RE = re.compile(r"[\u0591-\u05C7]")


def fuzzy_model(model: str | None) -> str:
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model or "")


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    answer = re.sub(r"[^\w\s]", "", answer, flags=re.UNICODE).strip()
    return answer or None


def usage_tuple(body: dict) -> tuple[int | None, int | None, int | None]:
    usage = body.get("usage", {}) or {}
    return (
        usage.get("prompt_tokens", usage.get("promptTokens")),
        usage.get("completion_tokens", usage.get("completionTokens")),
        usage.get("total_tokens", usage.get("totalTokens")),
    )


def tokenize(text: str | None) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


def bag_equal(left: str | None, right: str | None) -> bool:
    return Counter(tokenize(left)) == Counter(tokenize(right))


def repetition_stats(text: str | None) -> tuple[float, int]:
    tokens = tokenize(text)
    if not tokens:
        return 0.0, 0

    counts = Counter(tokens)
    repeated_fraction = max(counts.values()) / len(tokens)

    longest_run = 1
    current_run = 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)

    return repeated_fraction, longest_run


def contains_latin_script(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    for char in text:
        if not char.isalpha():
            continue
        if "LATIN" in unicodedata.name(char, ""):
            return True
    return False


def detect_script(text: str | None) -> str:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    if CYRILLIC_RE.search(text):
        return "cyrillic"
    if HEBREW_RE.search(text):
        return "hebrew"
    if contains_latin_script(text):
        return "latin"
    return "other"


def strip_hebrew_diacritics(text: str | None) -> str:
    return re.sub(r"[\u0591-\u05C7]", "", text or "")


def strip_latin_diacritics(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def infer_target_orthography(sample_word: str, dataset: str) -> str:
    if CYRILLIC_RE.search(sample_word):
        return "cyrillic"
    if HEBREW_RE.search(sample_word):
        if dataset == "orthography_large_exp":
            return "hebrew" if HEBREW_DIACRITIC_RE.search(sample_word) else "hebrew_unpointed"
        return "yiddish"
    if contains_latin_script(sample_word):
        stripped = strip_latin_diacritics(sample_word)
        if stripped != sample_word:
            return "latin_diacritic"
        return "latin"
    return "unknown"


def extract_json_field(line: str, start_marker: str, end_markers: list[str]):
    start = line.index(start_marker) + len(start_marker)
    candidate_ends = [line.index(marker, start) for marker in end_markers if marker in line[start:]]
    end = min(candidate_ends) if candidate_ends else len(line)
    return json.loads(line[start:end])


def read_jsonl_prefix_until(handle, marker: bytes, chunk_size: int = 65536) -> bytes:
    buffer = bytearray()
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            return bytes(buffer)
        buffer.extend(chunk)
        marker_index = buffer.find(marker)
        if marker_index != -1:
            prefix = bytes(buffer[:marker_index])
            remainder = buffer[marker_index:]
            newline_index = remainder.find(b"\n")
            while newline_index == -1:
                chunk = handle.read(chunk_size)
                if not chunk:
                    return prefix
                newline_index = chunk.find(b"\n")
                if newline_index != -1:
                    handle.seek(newline_index - len(chunk) + 1, 1)
                    return prefix
            handle.seek(newline_index - len(remainder) + 1, 1)
            return prefix
        newline_index = buffer.find(b"\n")
        if newline_index != -1:
            return bytes(buffer[:newline_index])


def load_outputs(batch_dir: Path, exp: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(batch_dir.glob("*_output.jsonl")):
        with open(path) as handle:
            for line in handle:
                item = json.loads(line)
                body = (item.get("response") or {}).get("body") or {}
                choices = body.get("choices") or []
                message = ((choices[0] or {}).get("message") or {}).get("content") if choices else None
                prompt_tokens, completion_tokens, total_tokens = usage_tuple(body)
                row = {
                    "exp": exp,
                    "custom_id": item.get("custom_id"),
                    "batch_file": path.name,
                    "batch_id": path.name.replace("_output.jsonl", ""),
                    "model": body.get("model"),
                    "fuzzy_model": fuzzy_model(body.get("model")),
                    "model_response": message,
                    "model_answer": extract_answer(message),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
                rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def load_inputs(batch_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(batch_dir.glob("inputs_*.jsonl")):
        with open(path) as handle:
            for line in handle:
                item = json.loads(line)
                body = item["body"]
                metadata = body.get("metadata") or {}
                rows.append(
                    {
                        "custom_id": item["custom_id"],
                        "input_file": path.name,
                        "fuzzy_model": fuzzy_model(body.get("model")),
                        "grammar_name": metadata.get("grammar_name"),
                        "sample_id": metadata.get("sample_id"),
                        "input_sentence": metadata.get("input_sentence"),
                        "output_sentence": metadata.get("output_sentence"),
                        "depth": pd.to_numeric(metadata.get("depth"), errors="coerce"),
                        "n_words": pd.to_numeric(metadata.get("n_words"), errors="coerce"),
                        "n_rules": pd.to_numeric(metadata.get("n_rules"), errors="coerce"),
                    }
                )
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def merge_outputs_inputs(outputs_df: pd.DataFrame, inputs_df: pd.DataFrame) -> pd.DataFrame:
    if outputs_df.empty:
        return ensure_columns(outputs_df, INPUT_COLUMNS)
    if inputs_df.empty:
        return ensure_columns(outputs_df, INPUT_COLUMNS)

    if inputs_df["custom_id"].is_unique:
        dedup_inputs_df = inputs_df.drop(columns=["fuzzy_model"], errors="ignore")
        return ensure_columns(outputs_df.merge(dedup_inputs_df, on="custom_id", how="left"), INPUT_COLUMNS)

    merge_keys = ["custom_id", "fuzzy_model"]
    dedup_inputs_df = (
        inputs_df.sort_values(["custom_id", "fuzzy_model", "input_file"], na_position="last")
        .drop_duplicates(subset=merge_keys, keep="first")
    )
    return ensure_columns(outputs_df.merge(dedup_inputs_df, on=merge_keys, how="left"), INPUT_COLUMNS)


def load_sample_sentences(data_dir: Path, sample_index_df: pd.DataFrame) -> pd.DataFrame:
    if sample_index_df.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)

    needed_df = sample_index_df.dropna(subset=["grammar_name", "sample_id"]).copy()
    if needed_df.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)

    needed_df["sample_id"] = needed_df["sample_id"].astype(str)
    needed_ids = (
        needed_df.assign(sample_id_int=lambda df: df["sample_id"].astype(int))
        .groupby("grammar_name")["sample_id_int"]
        .agg(set)
        .to_dict()
    )

    rows: list[dict] = []
    for grammar_name, wanted_ids in needed_ids.items():
        path = data_dir / f"samples_{grammar_name}.jsonl"
        if not path.exists():
            continue
        remaining_ids = set(wanted_ids)
        with open(path) as handle:
            for sample_idx, line in enumerate(handle):
                if sample_idx not in remaining_ids:
                    if not remaining_ids:
                        break
                    continue
                sample = json.loads(line)
                rows.append(
                    {
                        "grammar_name": grammar_name,
                        "sample_id": str(sample_idx),
                        "input_sentence": sample.get("left_phonetic") or sample.get("left"),
                        "output_sentence": sample.get("right_phonetic") or sample.get("right"),
                    }
                )
                remaining_ids.remove(sample_idx)
                if not remaining_ids:
                    break
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS)


def load_grammar_metadata(exp: str, data_dir: Path, dataset: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(data_dir.glob("grammar_*.json")):
        with open(path) as handle:
            grammar = json.load(handle)
        grammar_id = path.stem.split("grammar_")[1]
        row = {
            "grammar_name": grammar_id,
            "a_words": sum(
                [grammar["a"][key] for key in ["verbs", "nouns", "propns", "prons", "adjs", "det_def", "det_indef", "comps"]],
                [],
            ),
            "b_words": sum(
                [grammar["b"][key] for key in ["verbs", "nouns", "propns", "prons", "adjs", "det_def", "det_indef", "comps"]],
                [],
            ),
            "n_words": pd.to_numeric(grammar.get("n_words"), errors="coerce"),
            "n_rules": pd.to_numeric(grammar.get("n_rules"), errors="coerce"),
        }
        if exp == "wordorder":
            share_head = grammar["a"]["head_initial"] == grammar["b"]["head_initial"]
            share_spec = grammar["a"]["spec_initial"] == grammar["b"]["spec_initial"]
            if share_head and share_spec:
                row["target_word_order"] = "SVO"
            elif share_head and not share_spec:
                row["target_word_order"] = "VOS"
            elif not share_head and share_spec:
                row["target_word_order"] = "SOV"
            else:
                row["target_word_order"] = "OVS"
        if exp == "orthography":
            sample_word = next((word for word in row["b_words"] if word), "")
            row["target_orthography"] = infer_target_orthography(sample_word, dataset)
        rows.append(row)
    return pd.DataFrame(rows)


def load_standard_experiment(spec: ExperimentSpec) -> pd.DataFrame:
    outputs_df = load_outputs(spec.batch_dir, spec.exp)
    inputs_df = load_inputs(spec.batch_dir)
    merged_df = merge_outputs_inputs(outputs_df, inputs_df)

    needs_sample_df = merged_df.loc[
        (merged_df["input_sentence"].isna() | merged_df["output_sentence"].isna())
        & merged_df["sample_id"].notna(),
        ["grammar_name", "sample_id"],
    ]
    sample_df = load_sample_sentences(spec.data_dir, needs_sample_df)
    if not sample_df.empty:
        merged_df = merged_df.merge(
            sample_df,
            on=["grammar_name", "sample_id"],
            how="left",
            suffixes=("", "_sample"),
        )
        for column in ["input_sentence", "output_sentence"]:
            sample_column = f"{column}_sample"
            if sample_column in merged_df.columns:
                merged_df[column] = merged_df[column].combine_first(merged_df[sample_column])
                merged_df = merged_df.drop(columns=[sample_column])

    grammar_df = load_grammar_metadata(spec.exp, spec.data_dir, spec.dataset)
    merged_df = merged_df.merge(grammar_df, on="grammar_name", how="left", suffixes=("", "_grammar"))
    for column in ["n_words", "n_rules"]:
        grammar_column = f"{column}_grammar"
        if grammar_column in merged_df.columns:
            primary_values = pd.to_numeric(merged_df[column], errors="coerce")
            fallback_values = pd.to_numeric(merged_df[grammar_column], errors="coerce")
            merged_df[column] = primary_values.where(primary_values.notna(), fallback_values)
            merged_df = merged_df.drop(columns=[grammar_column])
    merged_df["dataset"] = spec.dataset
    return merged_df


def load_agreement_experiment() -> pd.DataFrame:
    exp = "agreement"
    batch_dir = BATCH_DIR / "agreement_exp_compact"
    data_dir = DATA_DIR / "agreement_exp"

    input_rows: list[dict] = []
    for path in sorted(batch_dir.glob("inputs_*.jsonl")):
        with open(path) as handle:
            for line in handle:
                item = json.loads(line)
                metadata = item["body"].get("metadata") or {}
                input_rows.append(
                    {
                        "custom_id": item["custom_id"],
                        "grammar_name": metadata.get("grammar_name"),
                        "sample_id": metadata.get("sample_id"),
                        "depth": pd.to_numeric(metadata.get("depth"), errors="coerce"),
                    }
                )
    inputs_df = pd.DataFrame(input_rows).drop_duplicates(subset=["custom_id"])

    output_rows: list[dict] = []
    for path in sorted(batch_dir.glob("*_output.jsonl")):
        with open(path) as handle:
            for line in handle:
                item = json.loads(line)
                match = AGREEMENT_CUSTOM_ID_RE.match(item.get("custom_id", ""))
                if not match:
                    continue
                body = (item.get("response") or {}).get("body") or {}
                choices = body.get("choices") or []
                message = ((choices[0] or {}).get("message") or {}).get("content") if choices else None
                prompt_tokens, completion_tokens, total_tokens = usage_tuple(body)
                output_rows.append(
                    {
                        "exp": exp,
                        "custom_id": item.get("custom_id"),
                        "batch_file": path.name,
                        "batch_id": path.name.replace("_output.jsonl", ""),
                        "model": body.get("model"),
                        "fuzzy_model": fuzzy_model(body.get("model")),
                        "model_response": message,
                        "model_answer": extract_answer(message),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "sample_id": match.group("sample_id"),
                        "grammar_name": match.group("grammar_name"),
                    }
                )
    outputs_df = pd.DataFrame(output_rows).merge(
        inputs_df[["custom_id", "depth"]], on="custom_id", how="left"
    )

    needed_sample_ids = (
        outputs_df.assign(sample_id_int=lambda df: df["sample_id"].astype(int))
        .groupby("grammar_name")["sample_id_int"]
        .agg(list)
        .to_dict()
    )

    grammar_rows: list[dict] = []
    sample_rows: list[dict] = []

    with open(data_dir / "agreement_grammars.txt") as handle:
        grammar_ids = [line.strip() for line in handle if line.strip()]

    for grammar_id in grammar_ids:
        with open(data_dir / f"grammar_{grammar_id}.json") as handle:
            grammar = json.load(handle)

        agreement_metadata = grammar.get("agreement_metadata", {})
        a_enabled = bool(agreement_metadata.get("a", {}).get("config", {}).get("enabled", False))
        b_enabled = bool(agreement_metadata.get("b", {}).get("config", {}).get("enabled", False))
        grammar_rows.append(
            {
                "grammar_name": grammar_id,
                "n_words": grammar.get("n_words"),
                "n_rules": grammar.get("n_rules"),
                "agreement_enabled_a": a_enabled,
                "agreement_enabled_b": b_enabled,
                "agreement_condition": f"{'Agr' if a_enabled else 'NoAgr'} -> {'Agr' if b_enabled else 'NoAgr'}",
            }
        )

        wanted_ids = sorted(needed_sample_ids.get(grammar_id, []))
        if not wanted_ids:
            continue

        max_sample_id = max(wanted_ids)
        wanted_id_set = set(wanted_ids)
        with open(data_dir / f"samples_{grammar_id}.jsonl", "rb") as handle:
            for sample_idx in range(max_sample_id + 1):
                prefix = read_jsonl_prefix_until(handle, b', "possible_right"')
                if sample_idx > max_sample_id:
                    break
                if sample_idx not in wanted_id_set:
                    continue
                line = prefix.decode("utf-8")
                sample_rows.append(
                    {
                        "grammar_name": grammar_id,
                        "sample_id": str(sample_idx),
                        "input_sentence": extract_json_field(line, '"left_phonetic": ', [', "right":']),
                        "output_sentence": extract_json_field(
                            line, '"right_phonetic": ', [', "possible_right":', ', "left_tree":']
                        ),
                    }
                )

    samples_df = pd.DataFrame(sample_rows)
    grammars_df = pd.DataFrame(grammar_rows)
    merged_df = outputs_df.merge(samples_df, on=["grammar_name", "sample_id"], how="left").merge(
        grammars_df, on="grammar_name", how="left"
    )
    merged_df["dataset"] = data_dir.name
    return merged_df


def classify_failure(row: pd.Series) -> str:
    pred = row.get("model_answer")
    ref = row.get("output_sentence")
    src = row.get("input_sentence")

    if not isinstance(pred, str) or not pred.strip():
        return "no_answer"
    if isinstance(ref, str) and pred == ref:
        return "exact_match"
    if bag_equal(pred, ref):
        return "word_order_only"

    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    src_tokens = tokenize(src)
    pred_vocab = set(pred_tokens)
    ref_vocab = set(ref_tokens)
    src_vocab = set(src_tokens)
    b_words = row.get("b_words")
    target_vocab = set(b_words) if isinstance(b_words, list) else set()

    repeated_fraction, longest_run = repetition_stats(pred)
    src_overlap = len(pred_vocab & src_vocab) / max(1, len(pred_vocab))
    ref_overlap = len(pred_vocab & ref_vocab) / max(1, len(pred_vocab))
    oov_count = len(pred_vocab - target_vocab) if target_vocab else 0

    if longest_run >= 4 or repeated_fraction >= 0.5:
        return "repetition_loop"
    if isinstance(src, str) and pred == src:
        return "copied_source"
    if src_overlap >= 0.8 and ref_overlap < 0.5:
        return "source_lexicon_intrusion"
    if row.get("exp") == "orthography":
        target_orthography = row.get("target_orthography")
        pred_script = detect_script(pred)
        if target_orthography == "cyrillic" and pred_script != "cyrillic":
            return "wrong_script"
        if target_orthography in {"hebrew", "hebrew_unpointed", "yiddish"} and pred_script != "hebrew":
            return "wrong_script"
        if target_orthography in {"latin", "latin_diacritic"} and pred_script != "latin":
            return "wrong_script"
        if (
            target_orthography in {"hebrew", "hebrew_unpointed", "yiddish"}
            and strip_hebrew_diacritics(pred) == strip_hebrew_diacritics(ref)
            and pred != ref
        ):
            return "diacritic_drop"
        if (
            target_orthography in {"latin", "latin_diacritic"}
            and strip_latin_diacritics(pred) == strip_latin_diacritics(ref)
            and pred != ref
        ):
            return "diacritic_drop"
    if len(pred_tokens) < max(1, len(ref_tokens) // 2):
        return "too_short"
    if len(pred_tokens) > len(ref_tokens) + max(2, len(ref_tokens) // 2):
        return "too_long"
    if oov_count > 0:
        return "hallucinated_vocab"
    if pred_tokens and ref_tokens and (
        pred_tokens == ref_tokens[: len(pred_tokens)] or pred_tokens == ref_tokens[-len(pred_tokens) :]
    ):
        return "partial_span"
    if len(pred_tokens) == len(ref_tokens):
        return "same_length_substitution"
    return "mixed_other"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exact_match"] = df.apply(
        lambda row: isinstance(row.get("model_answer"), str)
        and isinstance(row.get("output_sentence"), str)
        and row["model_answer"] == row["output_sentence"],
        axis=1,
    )
    df["bow_match"] = df.apply(
        lambda row: bag_equal(row.get("model_answer"), row.get("output_sentence")), axis=1
    )
    df["failure_mode"] = df.apply(classify_failure, axis=1)
    df["pred_len"] = df["model_answer"].apply(lambda text: len(tokenize(text)))
    df["ref_len"] = df["output_sentence"].apply(lambda text: len(tokenize(text)))
    df["src_len"] = df["input_sentence"].apply(lambda text: len(tokenize(text)))
    df["length_delta"] = df["pred_len"] - df["ref_len"]
    df["pred_script"] = df["model_answer"].apply(detect_script)
    return df


def build_dataset() -> pd.DataFrame:
    parts = [load_standard_experiment(spec) for spec in STANDARD_EXPERIMENTS]
    parts.append(load_agreement_experiment())
    return enrich(pd.concat(parts, ignore_index=True, sort=False))


def write_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    row_columns = [
        "dataset",
        "exp",
        "fuzzy_model",
        "custom_id",
        "grammar_name",
        "sample_id",
        "input_sentence",
        "output_sentence",
        "model_answer",
        "exact_match",
        "bow_match",
        "failure_mode",
        "target_word_order",
        "target_orthography",
        "agreement_condition",
        "depth",
        "n_words",
        "n_rules",
        "prompt_tokens",
        "completion_tokens",
        "pred_len",
        "ref_len",
        "src_len",
        "length_delta",
        "pred_script",
    ]
    row_path = out_dir / "rows.csv"
    available_row_columns = [column for column in row_columns if column in df.columns]
    df[available_row_columns].to_csv(row_path, index=False)

    wrong_df = df[~df["exact_match"]].copy()

    summary_df = (
        wrong_df.groupby(["dataset", "exp", "fuzzy_model", "failure_mode"])
        .size()
        .rename("count")
        .reset_index()
    )
    summary_df["pct_within_model_exp"] = summary_df.groupby(["dataset", "fuzzy_model"])["count"].transform(
        lambda series: 100 * series / series.sum()
    )
    summary_df = summary_df.sort_values(
        ["dataset", "exp", "fuzzy_model", "count"], ascending=[True, True, True, False]
    )
    summary_df.to_csv(out_dir / "failure_mode_summary.csv", index=False)

    metric_df = (
        df.groupby(["dataset", "exp", "fuzzy_model"])
        .agg(
            rows=("custom_id", "size"),
            exact_match=("exact_match", "mean"),
            bow_match=("bow_match", "mean"),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
            mean_length_delta=("length_delta", "mean"),
        )
        .reset_index()
    )
    metric_df["bow_minus_exact"] = metric_df["bow_match"] - metric_df["exact_match"]
    metric_df.to_csv(out_dir / "metric_summary.csv", index=False)

    orthography_df = df[df["exp"] == "orthography"].copy()
    orthography_summary = (
        orthography_df.groupby(["dataset", "fuzzy_model", "target_orthography"])
        .agg(
            rows=("custom_id", "size"),
            exact_match=("exact_match", "mean"),
            bow_match=("bow_match", "mean"),
            wrong_script=("failure_mode", lambda s: (s == "wrong_script").mean()),
            diacritic_drop=("failure_mode", lambda s: (s == "diacritic_drop").mean()),
            too_short=("failure_mode", lambda s: (s == "too_short").mean()),
            same_length_substitution=("failure_mode", lambda s: (s == "same_length_substitution").mean()),
        )
        .reset_index()
    )
    orthography_summary.to_csv(out_dir / "orthography_summary.csv", index=False)

    example_modes = [
        "repetition_loop",
        "source_lexicon_intrusion",
        "wrong_script",
        "diacritic_drop",
        "same_length_substitution",
        "partial_span",
        "too_short",
        "hallucinated_vocab",
        "word_order_only",
    ]
    examples = {}
    example_columns = [
        "dataset",
        "exp",
        "fuzzy_model",
        "custom_id",
        "failure_mode",
        "input_sentence",
        "output_sentence",
        "model_answer",
        "target_word_order",
        "target_orthography",
        "agreement_condition",
        "depth",
        "n_words",
        "prompt_tokens",
        "completion_tokens",
    ]
    for mode in example_modes:
        subset = wrong_df[wrong_df["failure_mode"] == mode]
        examples[mode] = subset[example_columns].head(20).to_dict(orient="records")

    with open(out_dir / "example_rows.json", "w") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=False)


def main(out_dir: str = str(OUTPUT_DIR)) -> None:
    df = build_dataset()
    write_outputs(df, Path(out_dir))
    print(f"Wrote analysis outputs to {out_dir}")
    print(df.groupby(["dataset", "exp", "fuzzy_model"]).size())


if __name__ == "__main__":
    fire.Fire(main)
