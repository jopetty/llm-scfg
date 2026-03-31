from __future__ import annotations

import json
import re
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import fire
import pyrootutils

PROJECT_ROOT = Path(
    pyrootutils.find_root(search_from=__file__, indicator=".project-root")
)
BATCH_DIR = PROJECT_ROOT / "batches"
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "prompt_viewer_assets"

FINAL_ANSWER_RE = re.compile(
    r"final\s*answer\s*(?::|-|\u2014)?\s*(?:is\s*)?([^\n]+)",
    re.IGNORECASE | re.DOTALL,
)
GRAMMAR_RE = re.compile(
    r"Here is the synchronous context-free grammar:\s*```(.*?)```",
    re.DOTALL,
)
INPUT_SENTENCE_RE = re.compile(
    r"Here is the input sentence:\s*`([^`]*)`",
    re.DOTALL,
)
CUSTOM_ID_RE = re.compile(
    r"^(?P<grammar_name>[0-9a-f]+)-(?P<input_hash>[0-9a-f]+)-sample-(?P<sample_id>\d+)$"
)
DATE_SUFFIX_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    matches = FINAL_ANSWER_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    answer = re.sub(r"[^\w\s]", "", answer, flags=re.UNICODE).strip()
    return answer or None


def extract_grammar(prompt: str) -> str | None:
    match = GRAMMAR_RE.search(prompt)
    if not match:
        return None
    grammar = match.group(1).strip()
    lines = grammar.splitlines()
    return "\n".join(line[4:] if line.startswith("    ") else line for line in lines)


def extract_input_sentence(prompt: str) -> str | None:
    match = INPUT_SENTENCE_RE.search(prompt)
    if not match:
        return None
    return match.group(1).strip()


def parse_custom_id(custom_id: str) -> tuple[str | None, str | None]:
    match = CUSTOM_ID_RE.match(custom_id)
    if not match:
        return None, None
    return match.group("grammar_name"), match.group("sample_id")


def fuzzy_model(model: str | None) -> str | None:
    if not model:
        return None
    return DATE_SUFFIX_RE.sub("", model)


def response_status_label(status_code: int | None) -> str:
    if status_code is None:
        return "missing"
    if 200 <= status_code < 300:
        return "ok"
    return "error"


def agreement_condition_key(
    source_marks_agreement: bool | None,
    target_marks_agreement: bool | None,
) -> str | None:
    if source_marks_agreement is None or target_marks_agreement is None:
        return None
    return (
        f"source_{'on' if source_marks_agreement else 'off'}"
        f"__target_{'on' if target_marks_agreement else 'off'}"
    )


def agreement_condition_label(
    source_marks_agreement: bool | None,
    target_marks_agreement: bool | None,
) -> str | None:
    if source_marks_agreement is None or target_marks_agreement is None:
        return None
    source_label = "Agr" if source_marks_agreement else "NoAgr"
    target_label = "Agr" if target_marks_agreement else "NoAgr"
    return f"{source_label} -> {target_label}"


def infer_default_data_dir(batch_dir: Path) -> Path | None:
    if "agreement_exp" in batch_dir.name:
        return DATA_DIR / "agreement_exp"
    return None


def load_agreement_index(data_dir: Path | None) -> dict[str, dict[str, Any]]:
    if data_dir is None or not data_dir.exists():
        return {}

    index: dict[str, dict[str, Any]] = {}
    for grammar_path in sorted(data_dir.glob("grammar_*.json")):
        payload = json.loads(grammar_path.read_text())
        grammar_name = payload.get("name") or grammar_path.stem.replace("grammar_", "")
        agreement_metadata = payload.get("agreement_metadata") or {}
        source_enabled = ((agreement_metadata.get("a") or {}).get("config") or {}).get(
            "enabled"
        )
        target_enabled = ((agreement_metadata.get("b") or {}).get("config") or {}).get(
            "enabled"
        )
        index[str(grammar_name)] = {
            "source_marks_agreement": source_enabled,
            "target_marks_agreement": target_enabled,
            "agreement_condition": agreement_condition_key(
                source_enabled, target_enabled
            ),
            "agreement_condition_label": agreement_condition_label(
                source_enabled, target_enabled
            ),
        }
    return index


@dataclass
class PromptRecord:
    custom_id: str
    input_file: str
    model: str | None = None
    fuzzy_model: str | None = None
    grammar_name: str | None = None
    sample_id: str | None = None
    depth: str | None = None
    prompt: str | None = None
    grammar: str | None = None
    input_sentence: str | None = None
    output_file: str | None = None
    status_code: int | None = None
    request_id: str | None = None
    response_model: str | None = None
    response_text: str | None = None
    final_answer: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    input_payload: dict[str, Any] | None = None
    output_payload: dict[str, Any] | None = None
    source_marks_agreement: bool | None = None
    target_marks_agreement: bool | None = None
    agreement_condition: str | None = None
    agreement_condition_label: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "custom_id": self.custom_id,
            "model": self.model,
            "fuzzy_model": self.fuzzy_model,
            "grammar_name": self.grammar_name,
            "sample_id": self.sample_id,
            "depth": self.depth,
            "input_sentence": self.input_sentence,
            "final_answer": self.final_answer,
            "status": response_status_label(self.status_code),
            "status_code": self.status_code,
            "response_model": self.response_model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error_message": self.error_message,
            "source_marks_agreement": self.source_marks_agreement,
            "target_marks_agreement": self.target_marks_agreement,
            "agreement_condition": self.agreement_condition,
            "agreement_condition_label": self.agreement_condition_label,
        }

    def detail(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "input_file": self.input_file,
            "output_file": self.output_file,
            "request_id": self.request_id,
            "prompt": self.prompt,
            "grammar": self.grammar,
            "response_text": self.response_text,
            "metadata": self.metadata,
            "input_payload": self.input_payload,
            "output_payload": self.output_payload,
        }


class BatchViewerDataset:
    def __init__(self, batch_dir: Path, data_dir: Path | None = None):
        self.batch_dir = batch_dir
        self.data_dir = data_dir
        self.agreement_index = load_agreement_index(data_dir)
        self.records_by_id = self._load_records()
        self.record_ids = sorted(
            self.records_by_id,
            key=lambda custom_id: (
                self.records_by_id[custom_id].fuzzy_model or "",
                self.records_by_id[custom_id].grammar_name or "",
                int(self.records_by_id[custom_id].sample_id or -1),
                custom_id,
            ),
        )
        self.models = sorted(
            {
                record.fuzzy_model
                for record in self.records_by_id.values()
                if record.fuzzy_model
            }
        )
        self.depths = sorted(
            {
                record.depth
                for record in self.records_by_id.values()
                if record.depth is not None
            },
            key=lambda depth: int(depth),
        )
        self.agreement_conditions = [
            {"value": key, "label": label}
            for key, label in sorted(
                {
                    (
                        record.agreement_condition,
                        record.agreement_condition_label,
                    )
                    for record in self.records_by_id.values()
                    if record.agreement_condition and record.agreement_condition_label
                },
                key=lambda item: item[0],
            )
        ]

    def _load_records(self) -> dict[str, PromptRecord]:
        records: dict[str, PromptRecord] = {}

        for input_path in sorted(self.batch_dir.glob("inputs_*.jsonl")):
            with input_path.open() as handle:
                for line in handle:
                    item = json.loads(line)
                    custom_id = str(item["custom_id"])
                    body = dict(item.get("body", {}))
                    messages = body.get("messages", [])
                    prompt = ""
                    if messages:
                        prompt = extract_text_content(messages[0].get("content"))
                    metadata = body.get("metadata") or {}
                    grammar_name, sample_id = parse_custom_id(custom_id)
                    record = PromptRecord(
                        custom_id=custom_id,
                        input_file=input_path.name,
                        model=body.get("model"),
                        fuzzy_model=fuzzy_model(body.get("model")),
                        grammar_name=metadata.get("grammar_name") or grammar_name,
                        sample_id=metadata.get("sample_id") or sample_id,
                        depth=metadata.get("depth"),
                        prompt=prompt,
                        grammar=extract_grammar(prompt),
                        input_sentence=extract_input_sentence(prompt),
                        metadata=metadata,
                        input_payload=item,
                    )
                    self._apply_grammar_metadata(record)
                    records[custom_id] = record

        for output_path in sorted(self.batch_dir.glob("*_output.jsonl")):
            with output_path.open() as handle:
                for line in handle:
                    item = json.loads(line)
                    custom_id = str(item["custom_id"])
                    record = records.get(custom_id)
                    if record is None:
                        grammar_name, sample_id = parse_custom_id(custom_id)
                        record = PromptRecord(
                            custom_id=custom_id,
                            input_file="",
                            grammar_name=grammar_name,
                            sample_id=sample_id,
                        )
                        self._apply_grammar_metadata(record)
                        records[custom_id] = record

                    response = item.get("response", {}) or {}
                    body = response.get("body", {}) or {}
                    choices = body.get("choices", []) or []
                    response_text = None
                    if choices:
                        message = choices[0].get("message", {}) or {}
                        response_text = extract_text_content(message.get("content"))
                    usage = body.get("usage", {}) or {}
                    error = item.get("error") or body.get("error") or {}

                    record.output_file = output_path.name
                    record.status_code = response.get("status_code")
                    record.request_id = response.get("request_id")
                    record.response_model = body.get("model")
                    record.response_text = response_text
                    record.final_answer = extract_final_answer(response_text)
                    record.prompt_tokens = usage.get("prompt_tokens")
                    record.completion_tokens = usage.get("completion_tokens")
                    record.total_tokens = usage.get("total_tokens")
                    record.error_message = error.get("message")
                    record.output_payload = item

        return records

    def _apply_grammar_metadata(self, record: PromptRecord) -> None:
        if not record.grammar_name:
            return
        metadata = self.agreement_index.get(record.grammar_name)
        if metadata is None:
            return
        record.source_marks_agreement = metadata["source_marks_agreement"]
        record.target_marks_agreement = metadata["target_marks_agreement"]
        record.agreement_condition = metadata["agreement_condition"]
        record.agreement_condition_label = metadata["agreement_condition_label"]

    def query_records(
        self,
        *,
        search: str = "",
        model: str = "",
        depth: str = "",
        agreement_condition: str = "",
        status: str = "",
        offset: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        normalized_search = search.strip().lower()
        filtered_ids: list[str] = []
        for custom_id in self.record_ids:
            record = self.records_by_id[custom_id]
            if model and record.fuzzy_model != model:
                continue
            if depth and record.depth != depth:
                continue
            if (
                agreement_condition
                and record.agreement_condition != agreement_condition
            ):
                continue
            if status and response_status_label(record.status_code) != status:
                continue
            if normalized_search:
                haystack = " ".join(
                    [
                        record.custom_id or "",
                        record.grammar_name or "",
                        record.input_sentence or "",
                        record.final_answer or "",
                        record.response_text or "",
                    ]
                ).lower()
                if normalized_search not in haystack:
                    continue
            filtered_ids.append(custom_id)

        page_ids = filtered_ids[offset : offset + limit]
        return {
            "batch_dir": str(self.batch_dir),
            "total_records": len(self.record_ids),
            "filtered_records": len(filtered_ids),
            "offset": offset,
            "limit": limit,
            "models": self.models,
            "depths": self.depths,
            "agreement_conditions": self.agreement_conditions,
            "records": [
                self.records_by_id[custom_id].summary() for custom_id in page_ids
            ],
        }

    def get_record(self, custom_id: str) -> dict[str, Any] | None:
        record = self.records_by_id.get(custom_id)
        if record is None:
            return None
        return record.detail()


def load_text_asset(path: Path) -> bytes:
    return path.read_bytes()


class PromptViewerHandler(BaseHTTPRequestHandler):
    dataset: BatchViewerDataset

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_bytes(
                load_text_asset(ASSETS_DIR / "index.html"),
                content_type="text/html; charset=utf-8",
            )
            return
        if parsed.path == "/app.js":
            self._send_bytes(
                load_text_asset(ASSETS_DIR / "app.js"),
                content_type="application/javascript; charset=utf-8",
            )
            return
        if parsed.path == "/styles.css":
            self._send_bytes(
                load_text_asset(ASSETS_DIR / "styles.css"),
                content_type="text/css; charset=utf-8",
            )
            return
        if parsed.path == "/api/records":
            query = parse_qs(parsed.query)
            payload = self.dataset.query_records(
                search=query.get("search", [""])[0],
                model=query.get("model", [""])[0],
                depth=query.get("depth", [""])[0],
                agreement_condition=query.get("agreement_condition", [""])[0],
                status=query.get("status", [""])[0],
                offset=int(query.get("offset", ["0"])[0]),
                limit=min(int(query.get("limit", ["200"])[0]), 500),
            )
            self._send_json(payload)
            return
        if parsed.path == "/api/record":
            query = parse_qs(parsed.query)
            custom_id = query.get("id", [""])[0]
            payload = self.dataset.get_record(custom_id)
            if payload is None:
                self.send_error(HTTPStatus.NOT_FOUND, "record not found")
                return
            self._send_json(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, content_type="application/json; charset=utf-8")

    def _send_bytes(self, data: bytes, *, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(
    batch_dir: str = "agreement_exp_compact",
    data_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8123,
) -> None:
    resolved_batch_dir = Path(batch_dir)
    if not resolved_batch_dir.is_absolute():
        resolved_batch_dir = BATCH_DIR / resolved_batch_dir
    if not resolved_batch_dir.exists():
        raise FileNotFoundError(f"Batch directory does not exist: {resolved_batch_dir}")

    resolved_data_dir: Path | None
    if data_dir is None:
        resolved_data_dir = infer_default_data_dir(resolved_batch_dir)
    else:
        resolved_data_dir = Path(data_dir)
        if not resolved_data_dir.is_absolute():
            resolved_data_dir = PROJECT_ROOT / resolved_data_dir

    PromptViewerHandler.dataset = BatchViewerDataset(
        resolved_batch_dir,
        resolved_data_dir,
    )
    server = ThreadingHTTPServer((host, port), PromptViewerHandler)
    print(f"Prompt viewer serving {resolved_batch_dir} at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    fire.Fire({"serve": serve})
