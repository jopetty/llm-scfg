from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyrootutils
import sacrebleu
from tqdm.auto import tqdm

PROJECT_ROOT = pyrootutils.find_root(indicator=".project-root")
DATA_DIR = PROJECT_ROOT / "data"
BATCHES_DIR = PROJECT_ROOT / "batches"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "includes"
CACHE_DIR = PROJECT_ROOT / "notebooks" / "cache" / "results-tables"

METRIC_ORDER = [
    ("exact_match", "Exact Match"),
    ("bow_match", "Bag of Words"),
    ("bleu", r"$\BLEU$"),
    ("chrF++", r"$\chrFPP$"),
]

MODEL_DISPLAY_NAMES = {
    "gpt-5-nano": "gpt-5-nano",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5": "gpt-5",
    "google/gemma-3-1b-it": "gemma-3-1b-it",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "google/gemma-3-12b-it": "gemma-3-12b-it",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
}
MODEL_ORDER = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-1b-it",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

WORD_ORDER_ORDER = ["SVO", "SOV", "OVS"]
WORD_ORDER_LABELS = {
    "SVO": "SVO → SVO",
    "SOV": "SVO → SOV",
    "OVS": "SVO → OVS",
}
AGREEMENT_ORDER = [
    "NoAgr → NoAgr",
    "Agr → NoAgr",
    "Agr → Agr",
    "NoAgr → Agr",
]
ORTHOGRAPHY_ORDER = [
    "Latin → Latin",
    "Latin → Latin (diacritics)",
    "Latin → Cyrillic",
    "Latin → Hebrew",
    "Latin → Hebrew (pointed)",
]
ORTHOGRAPHY_TABLE_LABELS = {
    "Latin → Latin": "Latin",
    "Latin → Latin (diacritics)": "Latin + diac.",
    "Latin → Cyrillic": "Cyrillic",
    "Latin → Hebrew": "Hebrew",
    "Latin → Hebrew (pointed)": "Hebrew + points",
}

ANSWER_RE = re.compile(
    r"final\s*answer\s*(?::|-|—)?\s*(?:is\s*)?([^\n]+)",
    re.IGNORECASE | re.DOTALL,
)
SIZE_CUSTOM_ID_RE = re.compile(
    r"^(?P<grammar>[0-9a-f]+)-[0-9a-f]+-(?:request|sample)-(?P<sample_id>\d+)$"
)
CUSTOM_ID_RE = re.compile(
    r"^(?P<grammar_name>[0-9a-f]+)-(?P<input_hash>[0-9a-f]+)-sample-(?P<sample_id>\d+)$"
)
CYRILLIC_RE = re.compile(r"[а-яА-Я]")
HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def fuzzy_model(model: str | None) -> str:
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model or "")


def ordered_models(models: Iterable[str]) -> list[str]:
    models = list(dict.fromkeys(model for model in models if model))
    prioritized = [model for model in MODEL_ORDER if model in models]
    remaining = sorted(model for model in models if model not in MODEL_ORDER)
    return prioritized + remaining


def display_model(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def escape_latex(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def compact_int(value: int | float) -> str:
    return f"{int(round(float(value))):,}"


def compact_number(value: int | float) -> str:
    value = float(value)
    if value.is_integer():
        return compact_int(value)
    return f"{value:.1f}".rstrip("0").rstrip(".")


def na_cell() -> str:
    return r"\multicolumn{1}{c}{---}"


def extract_answer_unicode(model_response: str | None) -> str | None:
    if not isinstance(model_response, str):
        return None
    matches = ANSWER_RE.findall(model_response)
    if not matches:
        return None
    answer = re.sub(r"[^\w\s]", "", matches[-1], flags=re.UNICODE).strip()
    return answer or None


def extract_answer_ascii(model_response: str | None) -> str | None:
    if not isinstance(model_response, str):
        return None
    matches = ANSWER_RE.findall(model_response)
    if not matches:
        return None
    answer = re.sub(r"[^a-zA-Z\s]", "", matches[-1]).strip()
    return answer or None


def tokenize(text: str | None) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


def bag_equal(left: str | None, right: str | None) -> bool:
    return Counter(tokenize(left)) == Counter(tokenize(right))


def detect_script(text: str | None) -> str:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    if CYRILLIC_RE.search(text):
        return "cyrillic"
    if HEBREW_RE.search(text):
        return "hebrew"
    for char in text:
        if char.isalpha() and "LATIN" in unicodedata.name(char, ""):
            return "latin"
    return "other"


def count_json_string_array_items(array_text: str) -> int:
    array_text = array_text.strip()
    if array_text == "[]":
        return 0
    return array_text.count('", "') + 1


def json_string_array_contains(array_text: str, value: str | None) -> bool:
    if not value:
        return False
    needle = json.dumps(value, ensure_ascii=False)
    return any(
        token in array_text
        for token in (
            f"[{needle}]",
            f"[{needle}, ",
            f", {needle}, ",
            f", {needle}]",
        )
    )


def extract_possible_right_phonetic_array(line: str) -> str:
    start_marker = '"possible_right_phonetic": '
    end_markers = [', "agreement_ok":', ', "left_tree":', ', "grammar_name":']
    start = line.index(start_marker) + len(start_marker)
    end = min(
        line.index(marker, start) for marker in end_markers if marker in line[start:]
    )
    return line[start:end]


def compute_metrics(
    df: pd.DataFrame,
    *,
    reference_col: str,
    prediction_col: str,
) -> pd.DataFrame:
    df = df.copy()
    preds = df[prediction_col].fillna("").tolist()
    refs = df[reference_col].fillna("").tolist()
    chrf_metric = sacrebleu.metrics.CHRF(beta=2, word_order=2)

    df["exact_match"] = (
        df[prediction_col].fillna("").eq(df[reference_col].fillna(""))
        & df[prediction_col].notna()
        & df[reference_col].notna()
    )
    df["bow_match"] = [
        bag_equal(pred, ref) and pd.notna(pred) and pd.notna(ref)
        for pred, ref in zip(df[prediction_col], df[reference_col], strict=True)
    ]
    df["bleu"] = [
        sacrebleu.sentence_bleu(pred or "", [ref or ""]).score / 100.0
        for pred, ref in zip(df[prediction_col], df[reference_col], strict=True)
    ]
    df["chrF++"] = [
        chrf_metric.sentence_score(pred, [ref]).score / 100.0
        for pred, ref in zip(preds, refs, strict=True)
    ]
    return df


def usage_tuple(body: dict) -> tuple[int | None, int | None, int | None]:
    usage = body.get("usage", {}) or {}
    return (
        usage.get("prompt_tokens", usage.get("promptTokens")),
        usage.get("completion_tokens", usage.get("completionTokens")),
        usage.get("total_tokens", usage.get("totalTokens")),
    )


def add_length_midpoints(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    bins: int = 5,
) -> pd.DataFrame:
    df = df.copy()
    try:
        df[target_col] = pd.qcut(df[source_col], q=bins, duplicates="drop").apply(
            lambda interval: (interval.left + interval.right) / 2
            if pd.notna(interval)
            else float("nan")
        )
    except ValueError:
        df[target_col] = float("nan")
    return df


def build_size_table(
    df: pd.DataFrame,
    *,
    x_col: str,
    caption: str,
    label: str,
) -> str:
    models = [display_model(model) for model in ordered_models(df["model"].unique())]
    x_values = sorted(df[x_col].dropna().unique())
    values = {
        (row["model_display"], metric_key, row[x_col]): float(row[metric_key])
        for _, row in df.iterrows()
        for metric_key, _ in METRIC_ORDER
    }

    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \small",
        r"  \sisetup{table-format=1.3}",
        "  \\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X l "
        + " ".join("S" for _ in x_values)
        + "}",
        r"    \toprule",
        "    \\textbf{Model} & \\textbf{Metric} & "
        + " & ".join(f"\\textbf{{{compact_number(x_value)}}}" for x_value in x_values)
        + r" \\",
        r"    \midrule",
    ]

    for model_index, model in enumerate(models):
        for metric_index, (metric_key, metric_label) in enumerate(METRIC_ORDER):
            prefix = (
                f"    \\multirow{{{len(METRIC_ORDER)}}}{{=}}{{\\texttt{{{escape_latex(model)}}}}}"  # noqa E501
                if metric_index == 0
                else "    "
            )
            row = []
            for x_value in x_values:
                value = values.get((model, metric_key, x_value))
                row.append(na_cell() if value is None else f"{value:.3f}")
            lines.append(prefix + f" & {metric_label} & " + " & ".join(row) + r" \\")
        if model_index != len(models) - 1:
            lines.append(r"    \midrule")

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabularx}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_condition_tables(
    *,
    df: pd.DataFrame,
    condition_col: str,
    condition_order: list[str],
    condition_labels: dict[str, str],
    x_col: str,
    caption_template: str,
    label_prefix: str,
) -> str:
    x_values = sorted(df[x_col].dropna().unique())
    models = ordered_models(df["model"].dropna().unique())
    parts: list[str] = []

    for model in models:
        model_df = df[df["model"] == model].copy()
        values = {
            (row[condition_col], metric_key, row[x_col]): float(row[metric_key])
            for _, row in model_df.iterrows()
            for metric_key, _ in METRIC_ORDER
        }

        lines = [
            r"\begin{table*}[ht]",
            r"  \centering",
            r"  \small",
            r"  \sisetup{table-format=1.3}",
            "  \\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}X l "
            + " ".join("S" for _ in x_values)
            + "}",
            r"    \toprule",
            "    \\textbf{Condition} & \\textbf{Metric} & "
            + " & ".join(
                f"\\textbf{{{compact_number(x_value)}}}" for x_value in x_values
            )
            + r" \\",
            r"    \midrule",
        ]

        active_conditions = [
            cond for cond in condition_order if cond in set(model_df[condition_col])
        ]
        for cond_index, condition in enumerate(active_conditions):
            cond_label = escape_latex(condition_labels[condition])
            for metric_index, (metric_key, metric_label) in enumerate(METRIC_ORDER):
                prefix = (
                    f"    \\multirow{{{len(METRIC_ORDER)}}}{{=}}{{{cond_label}}}"
                    if metric_index == 0
                    else "    "
                )
                row = []
                for x_value in x_values:
                    value = values.get((condition, metric_key, x_value))
                    row.append(na_cell() if value is None else f"{value:.3f}")
                lines.append(
                    prefix + f" & {metric_label} & " + " & ".join(row) + r" \\"
                )
            if cond_index != len(active_conditions) - 1:
                lines.append(r"    \midrule")

        display = escape_latex(display_model(model))
        label_slug = model.replace("/", "_")
        lines.extend(
            [
                r"    \bottomrule",
                r"  \end{tabularx}",
                f"  \\caption{{{caption_template.format(model=display)}}}",
                f"  \\label{{{label_prefix}:{label_slug}}}",
                r"\end{table*}",
                "",
            ]
        )
        parts.append("\n".join(lines))

    return "\n".join(parts)


def write_output(name: str, content: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / name).write_text(content)


def cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.csv"


def load_or_compute_cached(
    name: str,
    compute_fn,
    *,
    force: bool,
) -> pd.DataFrame:
    path = cache_path(name)
    if path.exists() and not force:
        return pd.read_csv(path)

    df = compute_fn()
    df.to_csv(path, index=False)
    return df


def load_or_compute_cached_pair(
    name_prefix: str,
    compute_fn,
    *,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grammar_path = cache_path(f"{name_prefix}_grammar_size")
    length_path = cache_path(f"{name_prefix}_string_length")
    if grammar_path.exists() and length_path.exists() and not force:
        return pd.read_csv(grammar_path), pd.read_csv(length_path)

    grammar_df, length_df = compute_fn()
    grammar_df.to_csv(grammar_path, index=False)
    length_df.to_csv(length_path, index=False)
    return grammar_df, length_df


def load_size_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_dir = BATCHES_DIR / "size_exp"
    outputs_glob = sorted(exp_dir.glob("*_output.jsonl"))
    inputs_glob = sorted(
        path
        for path in exp_dir.glob("inputs_*.jsonl")
        if not path.name.endswith("_output.jsonl")
    )

    output_frames = []
    for path in tqdm(outputs_glob, desc="size outputs"):
        df = pd.read_json(path, lines=True)
        flat_df = pd.json_normalize(df.to_dict(orient="records"))
        if flat_df.empty:
            continue
        flat_df["model_response"] = flat_df["response.body.choices"].apply(
            lambda x: x[0]["message"]["content"] if isinstance(x, list) and x else None
        )
        flat_df["batch_id"] = path.name.split("_output.jsonl")[0]
        flat_df["model"] = flat_df["response.body.model"]
        flat_df["fuzzy_model"] = flat_df["model"].apply(fuzzy_model)
        output_frames.append(flat_df)

    input_frames = []
    for path in tqdm(inputs_glob, desc="size inputs"):
        df = pd.read_json(path, lines=True)
        flat_df = pd.json_normalize(df.to_dict(orient="records"))
        if flat_df.empty:
            continue
        flat_df["fuzzy_model"] = flat_df["body.model"].apply(fuzzy_model)
        flat_df["metadata_key"] = flat_df["custom_id"].apply(
            lambda custom_id: (
                f"{m.group('grammar')}-sample-{m.group('sample_id')}"
                if (m := SIZE_CUSTOM_ID_RE.match(custom_id or ""))
                else custom_id
            )
        )
        input_frames.append(flat_df)

    outputs_df = pd.concat(output_frames, ignore_index=True)
    outputs_df = outputs_df.drop(
        [col for col in outputs_df.columns if col.startswith("response")], axis=1
    ).drop(columns=["error"], errors="ignore")
    inputs_df = pd.concat(input_frames, ignore_index=True)

    metadata_reference_df = inputs_df.dropna(
        subset=["body.metadata.grammar_name"]
    ).drop_duplicates(subset=["metadata_key"], keep="first")[
        [
            "metadata_key",
            "body.metadata.input_sentence",
            "body.metadata.output_sentence",
            "body.metadata.n_rules",
            "body.metadata.n_words",
        ]
    ]
    inputs_df = inputs_df.merge(
        metadata_reference_df,
        on="metadata_key",
        how="left",
        suffixes=("", "_reference"),
    )
    for column in [
        "body.metadata.input_sentence",
        "body.metadata.output_sentence",
        "body.metadata.n_rules",
        "body.metadata.n_words",
    ]:
        ref_col = f"{column}_reference"
        inputs_df[column] = inputs_df[column].combine_first(inputs_df[ref_col])
        inputs_df = inputs_df.drop(columns=[ref_col])

    merged_df = pd.merge(
        outputs_df,
        inputs_df[
            [
                "custom_id",
                "fuzzy_model",
                "body.metadata.input_sentence",
                "body.metadata.output_sentence",
                "body.metadata.n_rules",
                "body.metadata.n_words",
            ]
        ],
        on=["custom_id", "fuzzy_model"],
        how="inner",
    ).drop_duplicates(subset=["custom_id", "batch_id"])

    merged_df["model_answer"] = merged_df["model_response"].apply(
        extract_answer_unicode
    )
    merged_df["input_sentence"] = merged_df["body.metadata.input_sentence"]
    merged_df["output_sentence"] = merged_df["body.metadata.output_sentence"]
    merged_df["size"] = pd.to_numeric(
        merged_df["body.metadata.n_rules"], errors="coerce"
    ) + pd.to_numeric(merged_df["body.metadata.n_words"], errors="coerce")
    merged_df["input_words"] = (
        merged_df["input_sentence"]
        .fillna("")
        .apply(lambda text: len(str(text).split()))
    )
    merged_df = add_length_midpoints(
        merged_df,
        source_col="input_words",
        target_col="input_words_binned_quant_num",
    )
    merged_df = compute_metrics(
        merged_df,
        reference_col="output_sentence",
        prediction_col="model_answer",
    )

    size_summary = (
        merged_df.groupby(["fuzzy_model", "size"], dropna=False)[
            [metric for metric, _ in METRIC_ORDER]
        ]
        .mean()
        .reset_index()
    )
    size_summary["model_display"] = size_summary["fuzzy_model"].map(display_model)
    size_summary = size_summary.rename(columns={"fuzzy_model": "model"})

    length_summary = (
        merged_df.dropna(subset=["input_words_binned_quant_num"])
        .groupby(["fuzzy_model", "input_words_binned_quant_num"], dropna=False)[
            [metric for metric, _ in METRIC_ORDER]
        ]
        .mean()
        .reset_index()
    )
    length_summary["model_display"] = length_summary["fuzzy_model"].map(display_model)
    length_summary = length_summary.rename(columns={"fuzzy_model": "model"})
    return size_summary, length_summary


def load_wordorder_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_exp_dir = DATA_DIR / "wordorder_large_exp"
    exp_dir = BATCHES_DIR / "wordorder_large_exp"
    grammar_ids = [
        line.strip()
        for line in (data_exp_dir / "wordorder_large_grammars.txt")
        .read_text()
        .splitlines()
        if line.strip()
    ]

    samples = []
    for grammar_id in tqdm(grammar_ids, desc="wordorder samples"):
        with (data_exp_dir / f"samples_{grammar_id}.jsonl").open() as handle:
            for sample_id, line in enumerate(handle):
                sample = json.loads(line)
                samples.append(
                    {
                        "grammar_name": grammar_id,
                        "sample_id": str(sample_id),
                        "output_sentence": sample["right_phonetic"],
                        "input_length": len(sample["left_phonetic"].split()),
                    }
                )
    samples_df = pd.DataFrame(samples)

    grammar_rows = []
    for path in tqdm(
        sorted(data_exp_dir.glob("grammar_*.json")), desc="wordorder grammars"
    ):
        grammar = json.loads(path.read_text())
        share_head = grammar["a"]["head_initial"] == grammar["b"]["head_initial"]
        share_spec = grammar["a"]["spec_initial"] == grammar["b"]["spec_initial"]
        if share_head and share_spec:
            target_word_order = "SVO"
        elif not share_head and share_spec:
            target_word_order = "SOV"
        elif not share_head and not share_spec:
            target_word_order = "OVS"
        else:
            target_word_order = "VOS"
        grammar_rows.append(
            {
                "grammar_name": path.stem.split("grammar_")[1],
                "size": int(grammar["n_rules"]) + int(grammar["n_words"]),
                "target_word_order": target_word_order,
            }
        )
    grammars_df = pd.DataFrame(grammar_rows)

    rows = []
    for path in tqdm(sorted(exp_dir.glob("*_output.jsonl")), desc="wordorder outputs"):
        with path.open() as handle:
            for line in handle:
                item = json.loads(line)
                body = (item.get("response") or {}).get("body") or {}
                choices = body.get("choices") or []
                message = (
                    ((choices[0] or {}).get("message") or {}).get("content")
                    if choices
                    else None
                )
                match = CUSTOM_ID_RE.match(item.get("custom_id", ""))
                if not match:
                    continue
                rows.append(
                    {
                        "batch_id": path.name.replace("_output.jsonl", ""),
                        "custom_id": item.get("custom_id"),
                        "grammar_name": match.group("grammar_name"),
                        "sample_id": match.group("sample_id"),
                        "model": fuzzy_model(body.get("model")),
                        "model_answer": extract_answer_ascii(message),
                    }
                )
    merged_df = pd.DataFrame(rows).drop_duplicates(subset=["batch_id", "custom_id"])
    merged_df = merged_df.merge(
        samples_df,
        on=["grammar_name", "sample_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        grammars_df,
        on="grammar_name",
        how="left",
        validate="many_to_one",
    )
    merged_df = merged_df[merged_df["target_word_order"].isin(WORD_ORDER_ORDER)].copy()
    merged_df = add_length_midpoints(
        merged_df,
        source_col="input_length",
        target_col="input_length_quintile_mid",
    )
    merged_df = compute_metrics(
        merged_df,
        reference_col="output_sentence",
        prediction_col="model_answer",
    )
    size_summary = (
        merged_df.groupby(["model", "target_word_order", "size"], dropna=False)[
            [metric for metric, _ in METRIC_ORDER]
        ]
        .mean()
        .reset_index()
    )
    length_summary = (
        merged_df.dropna(subset=["input_length_quintile_mid"])
        .groupby(
            ["model", "target_word_order", "input_length_quintile_mid"], dropna=False
        )[[metric for metric, _ in METRIC_ORDER]]
        .mean()
        .reset_index()
    )
    return size_summary, length_summary


def load_agreement_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_data_dir = DATA_DIR / "agreement_exp"
    exp_batch_dir = BATCHES_DIR / "agreement_exp_compact"

    grammar_ids = [
        line.strip()
        for line in (exp_data_dir / "agreement_grammars.txt").read_text().splitlines()
        if line.strip()
    ]

    grammar_rows = []
    for grammar_id in tqdm(grammar_ids, desc="agreement grammars"):
        grammar = json.loads((exp_data_dir / f"grammar_{grammar_id}.json").read_text())
        meta = grammar.get("agreement_metadata", {})
        a_enabled = bool(meta.get("a", {}).get("config", {}).get("enabled", False))
        b_enabled = bool(meta.get("b", {}).get("config", {}).get("enabled", False))
        grammar_rows.append(
            {
                "grammar_name": grammar_id,
                "grammar_size": 5 * len(grammar["a"]["verbs"]),
                "agreement_condition": f"{'Agr' if a_enabled else 'NoAgr'} → {'Agr' if b_enabled else 'NoAgr'}",  # noqa E501
            }
        )
    grammars_df = pd.DataFrame(grammar_rows)

    sample_rows = []
    possible_right_map: dict[tuple[str, str], str] = {}
    right_phonetic_map: dict[tuple[str, str], str] = {}
    for grammar_id in tqdm(grammar_ids, desc="agreement samples"):
        with (exp_data_dir / f"samples_{grammar_id}.jsonl").open() as handle:
            for sample_id, line in enumerate(handle):
                parsed = json.loads(line)
                possible_right_map[(grammar_id, str(sample_id))] = (
                    extract_possible_right_phonetic_array(line)
                )
                right_phonetic_map[(grammar_id, str(sample_id))] = parsed[
                    "right_phonetic"
                ]
                sample_rows.append(
                    {
                        "grammar_name": grammar_id,
                        "sample_id": str(sample_id),
                        "right_phonetic": parsed["right_phonetic"],
                        "input_length": len(parsed["left_phonetic"].split()),
                    }
                )
    samples_df = pd.DataFrame(sample_rows).merge(
        grammars_df,
        on="grammar_name",
        how="left",
        validate="many_to_one",
    )

    chrf_metric = sacrebleu.metrics.CHRF(beta=2, word_order=2)
    output_rows = []
    for path in tqdm(
        sorted(exp_batch_dir.glob("*_output.jsonl")), desc="agreement outputs"
    ):
        with path.open() as handle:
            for line in handle:
                item = json.loads(line)
                body = (item.get("response") or {}).get("body") or {}
                choices = body.get("choices") or []
                message = (
                    ((choices[0] or {}).get("message") or {}).get("content")
                    if choices
                    else None
                )
                match = CUSTOM_ID_RE.match(item.get("custom_id", ""))
                if not match:
                    continue
                output_rows.append(
                    {
                        "batch_id": path.name.replace("_output.jsonl", ""),
                        "custom_id": item.get("custom_id"),
                        "grammar_name": match.group("grammar_name"),
                        "sample_id": match.group("sample_id"),
                        "model": fuzzy_model(body.get("model")),
                        "model_answer": extract_answer_unicode(message),
                    }
                )
    merged_df = pd.DataFrame(output_rows).drop_duplicates(
        subset=["batch_id", "custom_id"]
    )
    merged_df = merged_df.merge(
        samples_df,
        on=["grammar_name", "sample_id"],
        how="left",
        validate="many_to_one",
    )
    merged_df["right_phonetic"] = [
        right_phonetic_map.get((grammar_name, str(sample_id)))
        for grammar_name, sample_id in zip(
            merged_df["grammar_name"], merged_df["sample_id"], strict=True
        )
    ]
    merged_df["exact_match"] = [
        json_string_array_contains(
            possible_right_map.get((grammar_name, str(sample_id)), "[]"),
            pred,
        )
        for grammar_name, sample_id, pred in zip(
            merged_df["grammar_name"],
            merged_df["sample_id"],
            merged_df["model_answer"],
            strict=True,
        )
    ]
    merged_df["bow_match"] = [
        bag_equal(pred, ref) and pd.notna(pred) and pd.notna(ref)
        for pred, ref in zip(
            merged_df["model_answer"], merged_df["right_phonetic"], strict=True
        )
    ]
    merged_df["bleu"] = [
        sacrebleu.sentence_bleu(pred or "", [ref or ""]).score / 100.0
        for pred, ref in zip(
            merged_df["model_answer"], merged_df["right_phonetic"], strict=True
        )
    ]
    merged_df["chrF++"] = [
        chrf_metric.sentence_score(pred or "", [ref or ""]).score / 100.0
        for pred, ref in zip(
            merged_df["model_answer"], merged_df["right_phonetic"], strict=True
        )
    ]
    merged_df = add_length_midpoints(
        merged_df,
        source_col="input_length",
        target_col="input_length_quintile_mid",
    )
    size_summary = (
        merged_df.groupby(
            ["model", "agreement_condition", "grammar_size"], dropna=False
        )[[metric for metric, _ in METRIC_ORDER]]
        .mean()
        .reset_index()
    )
    length_summary = (
        merged_df.dropna(subset=["input_length_quintile_mid"])
        .groupby(
            ["model", "agreement_condition", "input_length_quintile_mid"], dropna=False
        )[[metric for metric, _ in METRIC_ORDER]]
        .mean()
        .reset_index()
    )
    return size_summary, length_summary


def load_orthography_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_data_dir = DATA_DIR / "orthography_large_exp"
    exp_batch_dir = BATCHES_DIR / "orthography_large_exp"

    grammar_ids = [
        line.strip()
        for line in (exp_data_dir / "orthography_large_grammars.txt")
        .read_text()
        .splitlines()
        if line.strip()
    ]

    grammar_rows = []
    orthography_labels = {
        "latin": "Latin → Latin",
        "latin_diacritic": "Latin → Latin (diacritics)",
        "cyrillic": "Latin → Cyrillic",
        "hebrew": "Latin → Hebrew (pointed)",
        "hebrew_unpointed": "Latin → Hebrew",
    }
    for grammar_id in tqdm(grammar_ids, desc="orthography grammars"):
        grammar = json.loads((exp_data_dir / f"grammar_{grammar_id}.json").read_text())
        grammar_rows.append(
            {
                "grammar_name": grammar_id,
                "grammar_size": int(grammar["n_rules"]) + int(grammar["n_words"]),
                "target_orthography": orthography_labels[grammar["b"]["orthography"]],
            }
        )
    grammars_df = pd.DataFrame(grammar_rows)

    sample_rows = []
    for grammar_id in tqdm(grammar_ids, desc="orthography samples"):
        with (exp_data_dir / f"samples_{grammar_id}.jsonl").open() as handle:
            for sample_id, line in enumerate(handle):
                sample = json.loads(line)
                sample_rows.append(
                    {
                        "grammar_name": grammar_id,
                        "sample_id": str(sample_id),
                        "input_sentence": sample.get("left_phonetic")
                        or sample.get("left"),
                        "output_sentence": sample.get("right_phonetic")
                        or sample.get("right"),
                    }
                )
    samples_df = pd.DataFrame(sample_rows).merge(
        grammars_df,
        on="grammar_name",
        how="left",
        validate="many_to_one",
    )

    output_rows = []
    for path in tqdm(
        sorted(exp_batch_dir.glob("*_output.jsonl")), desc="orthography outputs"
    ):
        with path.open() as handle:
            for line in handle:
                item = json.loads(line)
                body = (item.get("response") or {}).get("body") or {}
                choices = body.get("choices") or []
                message = (
                    ((choices[0] or {}).get("message") or {}).get("content")
                    if choices
                    else None
                )
                match = CUSTOM_ID_RE.match(item.get("custom_id", ""))
                if not match:
                    continue
                output_rows.append(
                    {
                        "batch_id": path.name.replace("_output.jsonl", ""),
                        "custom_id": item.get("custom_id"),
                        "grammar_name": match.group("grammar_name"),
                        "sample_id": match.group("sample_id"),
                        "model": fuzzy_model(body.get("model")),
                        "model_answer": extract_answer_unicode(message),
                    }
                )
    merged_df = pd.DataFrame(output_rows).drop_duplicates(
        subset=["batch_id", "custom_id"]
    )
    merged_df = merged_df.merge(
        samples_df,
        on=["grammar_name", "sample_id"],
        how="left",
        validate="many_to_one",
    )
    merged_df["input_length"] = merged_df["input_sentence"].map(
        lambda text: len(tokenize(text))
    )
    merged_df = add_length_midpoints(
        merged_df,
        source_col="input_length",
        target_col="input_length_quintile_mid",
    )
    merged_df = compute_metrics(
        merged_df,
        reference_col="output_sentence",
        prediction_col="model_answer",
    )
    size_summary = (
        merged_df.groupby(
            ["model", "target_orthography", "grammar_size"], dropna=False
        )[[metric for metric, _ in METRIC_ORDER]]
        .mean()
        .reset_index()
    )
    length_summary = (
        merged_df.dropna(subset=["input_length_quintile_mid"])
        .groupby(
            ["model", "target_orthography", "input_length_quintile_mid"], dropna=False
        )[[metric for metric, _ in METRIC_ORDER]]
        .mean()
        .reset_index()
    )
    return size_summary, length_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached table values instead of loading them from notebooks/cache/results-tables.",  # noqa E501
    )
    args = parser.parse_args()

    size_size_df, size_length_df = load_or_compute_cached_pair(
        "size_values", load_size_results, force=args.force
    )
    write_output(
        "results_tables_size.tex",
        build_size_table(
            size_size_df,
            x_col="size",
            caption="Mean results by grammar size for all models in the size experiment.",  # noqa E501
            label="tab:results-size",
        )
        + build_size_table(
            size_length_df,
            x_col="input_words_binned_quant_num",
            caption="Mean results by input string length for all models in the size experiment.",  # noqa E501
            label="tab:results-size-length",
        ),
    )

    wordorder_size_df, wordorder_length_df = load_or_compute_cached_pair(
        "wordorder_large_values", load_wordorder_results, force=args.force
    )
    write_output(
        "results_tables_wordorder_large.tex",
        build_condition_tables(
            df=wordorder_size_df,
            condition_col="target_word_order",
            condition_order=WORD_ORDER_ORDER,
            condition_labels=WORD_ORDER_LABELS,
            x_col="size",
            caption_template=(
                "Mean results for \\texttt{{{model}}} in the word order experiment, "
                "grouped by target word-order condition and grammar size."
            ),
            label_prefix="tab:results-wordorder-grammar",
        )
        + build_condition_tables(
            df=wordorder_length_df,
            condition_col="target_word_order",
            condition_order=WORD_ORDER_ORDER,
            condition_labels=WORD_ORDER_LABELS,
            x_col="input_length_quintile_mid",
            caption_template=(
                "Mean results for \\texttt{{{model}}} in the word order experiment, "
                "grouped by target word-order condition and input string length."
            ),
            label_prefix="tab:results-wordorder-length",
        ),
    )

    agreement_size_df, agreement_length_df = load_or_compute_cached_pair(
        "agreement_values", load_agreement_results, force=args.force
    )
    write_output(
        "results_tables_agreement.tex",
        build_condition_tables(
            df=agreement_size_df,
            condition_col="agreement_condition",
            condition_order=AGREEMENT_ORDER,
            condition_labels={condition: condition for condition in AGREEMENT_ORDER},
            x_col="grammar_size",
            caption_template=(
                "Mean results for \\texttt{{{model}}} in the morphology experiment, "
                "grouped by agreement condition and grammar size."
            ),
            label_prefix="tab:results-agreement-grammar",
        )
        + build_condition_tables(
            df=agreement_length_df,
            condition_col="agreement_condition",
            condition_order=AGREEMENT_ORDER,
            condition_labels={condition: condition for condition in AGREEMENT_ORDER},
            x_col="input_length_quintile_mid",
            caption_template=(
                "Mean results for \\texttt{{{model}}} in the morphology experiment, "
                "grouped by agreement condition and input string length."
            ),
            label_prefix="tab:results-agreement-length",
        ),
    )

    orthography_size_df, orthography_length_df = load_or_compute_cached_pair(
        "orthography_large_values", load_orthography_results, force=args.force
    )
    write_output(
        "results_tables_orthography_large.tex",
        build_condition_tables(
            df=orthography_size_df,
            condition_col="target_orthography",
            condition_order=ORTHOGRAPHY_ORDER,
            condition_labels=ORTHOGRAPHY_TABLE_LABELS,
            x_col="grammar_size",
            caption_template=(
                "Mean results for \\texttt{{{model}}} in the orthography experiment, "
                "grouped by target orthography and grammar size."
            ),
            label_prefix="tab:results-orthography-grammar",
        )
        + build_condition_tables(
            df=orthography_length_df,
            condition_col="target_orthography",
            condition_order=ORTHOGRAPHY_ORDER,
            condition_labels=ORTHOGRAPHY_TABLE_LABELS,
            x_col="input_length_quintile_mid",
            caption_template=(
                "Mean results for \\texttt{{{model}}} in the orthography experiment, "
                "grouped by target orthography and input string length."
            ),
            label_prefix="tab:results-orthography-length",
        ),
    )


if __name__ == "__main__":
    main()
