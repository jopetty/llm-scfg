"""Run OpenAI-style batch JSONL files through Vertex AI Batch Prediction.

This is the asynchronous, discounted counterpart to ``open_weights.py``'s
online ``--auth=gcp`` mode. The workflow mirrors the OpenAI Batch API:

1. ``submit``  converts an ``inputs_*.jsonl`` file to Gemini's request format,
   uploads it to GCS, creates a batch job, and writes a small manifest next to
   the input file recording the job name and GCS paths.
2. ``status``  polls the job(s) recorded in one or more manifests.
3. ``collect`` downloads finished predictions and writes the sibling
   ``*_output.jsonl`` in the same OpenAI-like shape the analysis code reads.

Authentication uses Application Default Credentials for both Vertex AI and
Cloud Storage.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import dotenv
import fire
import pyrootutils

from open_weights import (
    GCP_MODEL_PREFIX,
    BatchRequest,
    build_error_record,
    build_success_record,
    default_output_path,
    load_batch_requests,
)
from scfg.utils import get_logger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

PROJECT_ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

dotenv.load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_LOCATION = "global"
MANIFEST_SUFFIX = "_gemini_batch.json"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

# Gemini finish reasons -> OpenAI finish reasons.
FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
    "OTHER": "stop",
}


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #


def resolve_project(gcp_project: str | None) -> str:
    project = (
        gcp_project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    )
    if not project:
        import google.auth

        _, project = google.auth.default()
    if not project:
        raise ValueError(
            "No GCP project found; pass --gcp_project or set GOOGLE_CLOUD_PROJECT"
        )
    return str(project)


def resolve_bucket(gcs_bucket: str | None, project: str) -> str:
    bucket = gcs_bucket or os.getenv("GEMINI_BATCH_BUCKET") or f"{project}-llm-scfg"
    return bucket.removeprefix("gs://").rstrip("/")


def resolve_location(gcp_location: str | None) -> str:
    return gcp_location or os.getenv("GEMINI_BATCH_LOCATION") or DEFAULT_LOCATION


def genai_client(project: str, location: str) -> Any:
    from google import genai

    return genai.Client(vertexai=True, project=project, location=location)


def strip_model_prefix(model: str) -> str:
    return model.removeprefix(GCP_MODEL_PREFIX)


def manifest_path(input_file: str | Path) -> Path:
    input_path = Path(input_file)
    return input_path.with_name(f"{input_path.stem}{MANIFEST_SUFFIX}")


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


# --------------------------------------------------------------------------- #
# GCS helpers
# --------------------------------------------------------------------------- #


def split_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got {gcs_uri}")
    bucket, _, blob = gcs_uri.removeprefix("gs://").partition("/")
    return bucket, blob


def storage_client(project: str) -> Any:
    from google.cloud import storage

    return storage.Client(project=project)


def gcs_upload(client: Any, local_path: Path, gcs_uri: str) -> None:
    bucket_name, blob_name = split_gcs_uri(gcs_uri)
    log.info("Uploading %s -> %s", local_path, gcs_uri)
    blob = client.bucket(bucket_name).blob(blob_name)
    # Checksums are verified server-side; large files upload in resumable
    # chunks automatically.
    blob.upload_from_filename(str(local_path), checksum="crc32c", timeout=600)


def gcs_list(client: Any, gcs_prefix: str) -> list[str]:
    bucket_name, prefix = split_gcs_uri(gcs_prefix)
    return [
        f"gs://{bucket_name}/{blob.name}"
        for blob in client.list_blobs(bucket_name, prefix=prefix)
    ]


def gcs_download(client: Any, gcs_uri: str, local_path: Path) -> None:
    bucket_name, blob_name = split_gcs_uri(gcs_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", gcs_uri, local_path)
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.download_to_filename(str(local_path), checksum="crc32c", timeout=600)


# --------------------------------------------------------------------------- #
# Request conversion: OpenAI chat body -> Gemini GenerateContent request
# --------------------------------------------------------------------------- #


def openai_messages_to_gemini(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        else:
            text = str(content or "")
        if role in {"system", "developer"}:
            system_parts.append({"text": text})
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    system_instruction = {"parts": system_parts} if system_parts else None
    return contents, system_instruction


def openai_body_to_gemini_request(body: dict[str, Any]) -> dict[str, Any]:
    contents, system_instruction = openai_messages_to_gemini(body.get("messages", []))
    request: dict[str, Any] = {"contents": contents}
    if system_instruction is not None:
        request["systemInstruction"] = system_instruction

    generation_config: dict[str, Any] = {}
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = int(max_tokens)
    if body.get("n") not in (None, 1):
        generation_config["candidateCount"] = int(body["n"])
    for openai_key, gemini_key in (
        ("temperature", "temperature"),
        ("top_p", "topP"),
        ("seed", "seed"),
    ):
        if body.get(openai_key) is not None:
            generation_config[gemini_key] = body[openai_key]
    if generation_config:
        request["generationConfig"] = generation_config
    return request


def request_text_key(gemini_request: dict[str, Any]) -> str:
    """Concatenate the prompt text of a Gemini request, for matching outputs."""
    pieces = []
    for content in gemini_request.get("contents") or []:
        for part in content.get("parts") or []:
            if isinstance(part, dict) and "text" in part:
                pieces.append(str(part["text"]))
    return "\n".join(pieces)


def build_vertex_input_lines(requests: list[BatchRequest]) -> list[str]:
    lines = []
    for request in requests:
        # ``custom_id`` is a passthrough field; if Vertex drops it, ``collect``
        # falls back to matching on the echoed request text.
        record = {
            "custom_id": request.custom_id,
            "request": openai_body_to_gemini_request(request.body),
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


# --------------------------------------------------------------------------- #
# Response conversion: Gemini prediction line -> OpenAI-like output record
# --------------------------------------------------------------------------- #


def gemini_response_to_openai_body(
    response: dict[str, Any], *, model: str
) -> dict[str, Any]:
    usage_meta = response.get("usageMetadata") or {}
    prompt_tokens = int(usage_meta.get("promptTokenCount") or 0)
    completion_tokens = int(usage_meta.get("candidatesTokenCount") or 0)
    reasoning_tokens = int(usage_meta.get("thoughtsTokenCount") or 0)
    total_tokens = int(
        usage_meta.get("totalTokenCount")
        or prompt_tokens + completion_tokens + reasoning_tokens
    )

    choices = []
    for index, candidate in enumerate(response.get("candidates") or []):
        parts = (candidate.get("content") or {}).get("parts") or []
        # Skip thought parts (only present when include_thoughts is on).
        text = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and not part.get("thought")
        )
        finish_reason = FINISH_REASON_MAP.get(
            str(candidate.get("finishReason") or "STOP"), "stop"
        )
        choices.append(
            {
                "index": index,
                "finish_reason": finish_reason,
                "logprobs": None,
                "message": {"role": "assistant", "content": text},
            }
        )

    return {
        "id": response.get("responseId") or f"vertex-batch-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
        "system_fingerprint": response.get("modelVersion") or "",
    }


def convert_prediction_line(
    line: dict[str, Any],
    *,
    model: str,
    custom_id_by_text: dict[str, str] | None = None,
) -> dict[str, Any]:
    custom_id = str(line.get("custom_id") or "")
    if not custom_id and custom_id_by_text:
        request = line.get("request")
        if isinstance(request, dict):
            custom_id = custom_id_by_text.get(request_text_key(request), "")
    status = line.get("status") or ""
    response = line.get("response")
    if status or not isinstance(response, dict):
        return build_error_record(
            custom_id,
            model=model,
            message=str(status or "missing response"),
        )
    if not response.get("candidates"):
        block_reason = (response.get("promptFeedback") or {}).get("blockReason")
        return build_error_record(
            custom_id,
            model=model,
            message=f"no candidates returned (blockReason={block_reason})",
            status_code=400,
        )
    body = gemini_response_to_openai_body(response, model=model)
    return build_success_record(
        custom_id,
        body,
        request_id=f"vertex-batch-{body['id']}",
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def prepare(
    input_file: str,
    output_file: str | None = None,
) -> str:
    """Convert an OpenAI batch JSONL into Vertex batch-prediction JSONL."""
    requests = load_batch_requests(input_file)
    if not requests:
        raise ValueError(f"No requests found in {input_file}")
    input_path = Path(input_file)
    resolved_output = (
        Path(output_file)
        if output_file is not None
        else input_path.with_name(f"{input_path.stem}_vertex.jsonl")
    )
    lines = build_vertex_input_lines(requests)
    with open(resolved_output, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    log.info("Wrote %s Vertex requests to %s", len(lines), resolved_output)
    return str(resolved_output)


def submit(
    input_file: str,
    model: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
    gcs_bucket: str | None = None,
    display_name: str | None = None,
    overwrite: bool = False,
) -> str:
    """Upload one batch file and create a Vertex AI batch prediction job."""
    input_path = Path(input_file)
    manifest_file = manifest_path(input_path)
    if manifest_file.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_file} already exists; pass overwrite=True to resubmit"
        )

    requests = load_batch_requests(input_path)
    if not requests:
        raise ValueError(f"No requests found in {input_file}")
    resolved_model = strip_model_prefix(
        model or str(requests[0].body.get("model") or "")
    )
    if not resolved_model:
        raise ValueError("No model given and none found in the batch file")

    project = resolve_project(gcp_project)
    location = resolve_location(gcp_location)
    bucket = resolve_bucket(gcs_bucket, project)
    job_tag = f"{input_path.stem}_{uuid.uuid4().hex[:6]}"
    gcs_input = f"gs://{bucket}/inputs/{job_tag}.jsonl"
    gcs_output_prefix = f"gs://{bucket}/outputs/{job_tag}"

    vertex_file = input_path.with_name(f"{input_path.stem}_vertex.jsonl")
    prepare(str(input_path), output_file=str(vertex_file))
    gcs_upload(storage_client(project), vertex_file, gcs_input)
    vertex_file.unlink()

    client = genai_client(project, location)
    job = client.batches.create(
        model=resolved_model,
        src=gcs_input,
        config={
            "dest": gcs_output_prefix,
            "display_name": display_name or job_tag,
        },
    )
    log.info("Created batch job %s (state=%s)", job.name, job_state_name(job))

    manifest = {
        "input_file": str(input_path),
        "n_requests": len(requests),
        "custom_ids": [request.custom_id for request in requests],
        "model": resolved_model,
        "project": project,
        "location": location,
        "gcs_input": gcs_input,
        "gcs_output_prefix": gcs_output_prefix,
        "job_name": job.name,
        "job_state": job_state_name(job),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    save_manifest(manifest_file, manifest)
    log.info("Wrote manifest to %s", manifest_file)
    return str(manifest_file)


def submit_dir(
    batch_dir: str,
    input_glob: str = "inputs_*.jsonl",
    model: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
    gcs_bucket: str | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Submit every matching input file under ``batch_dir`` as its own job."""
    batch_path = Path(batch_dir)
    input_files = sorted(
        path
        for path in batch_path.glob(input_glob)
        if not path.name.endswith("_output.jsonl")
        and not path.name.endswith("_vertex.jsonl")
    )
    if not input_files:
        raise FileNotFoundError(
            f"No input files matching {input_glob} under {batch_path}"
        )
    manifests = []
    for input_file in input_files:
        if manifest_path(input_file).exists() and not overwrite:
            log.info("Skipping %s; manifest already exists", input_file.name)
            continue
        manifests.append(
            submit(
                str(input_file),
                model=model,
                gcp_project=gcp_project,
                gcp_location=gcp_location,
                gcs_bucket=gcs_bucket,
                overwrite=overwrite,
            )
        )
    return manifests


def _find_manifests(target: str) -> list[Path]:
    path = Path(target)
    if path.is_dir():
        return sorted(path.glob(f"*{MANIFEST_SUFFIX}"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"{target} is neither a manifest nor a directory")


def job_state_name(job: Any) -> str:
    state = job.state
    return str(getattr(state, "value", state) or "JOB_STATE_UNSPECIFIED")


def _refresh_job_state(manifest: dict[str, Any]) -> Any:
    client = genai_client(manifest["project"], manifest["location"])
    job = client.batches.get(name=manifest["job_name"])
    manifest["job_state"] = job_state_name(job)
    if job.error is not None:
        manifest["job_error"] = job.error.model_dump(mode="json")
    return job


def status(target: str, refresh: bool = True) -> dict[str, str]:
    """Print the state of the job(s) for a manifest file or a batch directory."""
    states: dict[str, str] = {}
    for manifest_file in _find_manifests(target):
        manifest = load_manifest(manifest_file)
        if refresh and manifest.get("job_state") not in TERMINAL_STATES:
            _refresh_job_state(manifest)
            save_manifest(manifest_file, manifest)
        state = str(manifest.get("job_state"))
        states[manifest_file.name] = state
        error = manifest.get("job_error")
        log.info(
            "%s: %s%s",
            Path(manifest["input_file"]).name,
            state,
            f" ({error})" if error else "",
        )
    return states


def collect(
    target: str,
    output_dir: str | None = None,
    overwrite: bool = False,
    keep_raw: bool = False,
) -> list[str]:
    """Download finished predictions and write analysis-ready output JSONL."""
    outputs: list[str] = []
    for manifest_file in _find_manifests(target):
        manifest = load_manifest(manifest_file)
        input_path = Path(manifest["input_file"])
        resolved_output = default_output_path(input_path, output_dir=output_dir)
        if resolved_output.exists() and not overwrite:
            log.info("Skipping %s; output exists", resolved_output.name)
            continue

        if manifest.get("job_state") != "JOB_STATE_SUCCEEDED":
            _refresh_job_state(manifest)
            save_manifest(manifest_file, manifest)
        if manifest["job_state"] != "JOB_STATE_SUCCEEDED":
            log.warning(
                "Skipping %s; job state is %s",
                input_path.name,
                manifest["job_state"],
            )
            continue

        storage = storage_client(manifest["project"])
        prediction_uris = [
            uri
            for uri in gcs_list(storage, manifest["gcs_output_prefix"])
            if uri.endswith("predictions.jsonl")
        ]
        if not prediction_uris:
            raise FileNotFoundError(
                f"No predictions.jsonl under {manifest['gcs_output_prefix']}"
            )

        raw_dir = input_path.parent / ".vertex_raw" / input_path.stem
        model = f"{GCP_MODEL_PREFIX}{manifest['model']}"
        custom_ids = manifest.get("custom_ids") or []
        custom_id_by_text = {
            request_text_key(openai_body_to_gemini_request(request.body)): (
                request.custom_id
            )
            for request in load_batch_requests(input_path)
        }
        records_by_id: dict[str, dict[str, Any]] = {}
        unmatched: list[dict[str, Any]] = []
        for index, uri in enumerate(prediction_uris):
            local_raw = raw_dir / f"predictions_{index:03d}.jsonl"
            gcs_download(storage, uri, local_raw)
            with open(local_raw) as handle:
                for line_number, line in enumerate(handle):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    record = convert_prediction_line(
                        item, model=model, custom_id_by_text=custom_id_by_text
                    )
                    if record["custom_id"]:
                        records_by_id[record["custom_id"]] = record
                    else:
                        unmatched.append(record)
            if not keep_raw:
                local_raw.unlink()
        if not keep_raw:
            for directory in (raw_dir, raw_dir.parent):
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()

        if unmatched:
            log.warning(
                "%s prediction lines had no custom_id; they are written with "
                "empty custom_id at the end of %s",
                len(unmatched),
                resolved_output.name,
            )

        # Preserve input order and surface anything Vertex did not return.
        ordered: list[dict[str, Any]] = []
        missing = 0
        for custom_id in custom_ids:
            record = records_by_id.pop(custom_id, None)
            if record is None:
                missing += 1
                record = build_error_record(
                    custom_id,
                    model=model,
                    message="no prediction returned by Vertex batch job",
                    status_code=404,
                )
            ordered.append(record)
        ordered.extend(records_by_id.values())
        ordered.extend(unmatched)
        if missing:
            log.warning("%s requests missing from Vertex output", missing)

        n_ok = sum(1 for r in ordered if r["response"]["status_code"] == 200)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_output, "w") as handle:
            for record in ordered:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info(
            "Wrote %s records (%s ok) to %s",
            len(ordered),
            n_ok,
            resolved_output,
        )
        manifest["output_file"] = str(resolved_output)
        manifest["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save_manifest(manifest_file, manifest)
        outputs.append(str(resolved_output))
    return outputs


def wait(
    target: str,
    poll_seconds: float = 120.0,
    collect_when_done: bool = True,
) -> list[str]:
    """Block until every job for ``target`` is terminal, then collect."""
    while True:
        states = status(target)
        if all(state in TERMINAL_STATES for state in states.values()):
            break
        time.sleep(poll_seconds)
    return collect(target) if collect_when_done else []


if __name__ == "__main__":
    fire.Fire(
        {
            "prepare": prepare,
            "submit": submit,
            "submit_dir": submit_dir,
            "status": status,
            "collect": collect,
            "wait": wait,
        }
    )
