import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dotenv
import fire
import pyrootutils
from openai import AsyncOpenAI

from scfg.utils import get_logger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

PROJECT_ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
BATCH_DIR = PROJECT_ROOT / "batches"

dotenv.load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class BatchRequest:
    custom_id: str
    body: dict[str, Any]


def pathsafe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def normalize_chat_body(
    body: dict[str, Any],
    *,
    model_override: str | None = None,
) -> dict[str, Any]:
    normalized = dict(body)
    if model_override:
        normalized["model"] = model_override
    if "max_completion_tokens" in normalized and "max_tokens" not in normalized:
        normalized["max_tokens"] = normalized.pop("max_completion_tokens")
    else:
        normalized.pop("max_completion_tokens", None)

    # These are useful for provider batch uploads but are not accepted by most
    # local OpenAI-compatible servers.
    normalized.pop("metadata", None)
    normalized.pop("store", None)

    return {key: value for key, value in normalized.items() if value is not None}


def load_batch_requests(input_file: str | Path) -> list[BatchRequest]:
    requests: list[BatchRequest] = []
    with open(input_file) as handle:
        for line_number, line in enumerate(handle, start=1):
            item = json.loads(line)
            if item.get("url") != "/v1/chat/completions":
                raise ValueError(
                    f"{input_file}:{line_number} uses unsupported url={item.get('url')}"
                )
            requests.append(
                BatchRequest(
                    custom_id=str(item["custom_id"]),
                    body=dict(item["body"]),
                )
            )
    return requests


def build_success_record(
    custom_id: str,
    response_body: dict[str, Any],
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": request_id or f"local-{uuid.uuid4().hex}",
            "body": response_body,
        },
    }


def build_error_record(
    custom_id: str,
    *,
    model: str | None,
    message: str,
    status_code: int = 500,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "request_id": f"local-{uuid.uuid4().hex}",
            "body": {
                "model": model,
                "choices": [],
                "usage": {},
                "error": {"message": message},
            },
        },
    }


async def invoke_request(
    client: AsyncOpenAI,
    request: BatchRequest,
    *,
    model_override: str | None,
    max_retries: int,
    retry_backoff_seconds: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    request_body = normalize_chat_body(request.body, model_override=model_override)
    model_name = request_body.get("model")

    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(**request_body)
                response_body = response.model_dump(mode="json")
                return build_success_record(
                    request.custom_id,
                    response_body,
                    request_id=getattr(response, "_request_id", None),
                )
            except Exception as exc:
                if attempt >= max_retries:
                    return build_error_record(
                        request.custom_id,
                        model=str(model_name) if model_name is not None else None,
                        message=str(exc),
                    )
                await asyncio.sleep(retry_backoff_seconds * (attempt + 1))

    raise RuntimeError("unreachable")


async def run_batch_requests(
    requests: list[BatchRequest],
    *,
    base_url: str,
    api_key: str,
    model_override: str | None,
    concurrency: int,
    max_retries: int,
    retry_backoff_seconds: float,
    request_timeout_seconds: float,
) -> list[dict[str, Any]]:
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=request_timeout_seconds,
    )
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        invoke_request(
            client,
            request,
            model_override=model_override,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            semaphore=semaphore,
        )
        for request in requests
    ]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        await client.close()


def default_output_path(
    input_file: str | Path,
    *,
    output_dir: str | Path | None = None,
    model_override: str | None = None,
) -> Path:
    input_path = Path(input_file)
    parent = Path(output_dir) if output_dir is not None else input_path.parent
    suffix = (
        f"_{pathsafe_model_name(model_override)}" if model_override is not None else ""
    )
    return parent / f"{input_path.stem}{suffix}_output.jsonl"


def run_batch_file(
    input_file: str,
    output_file: str | None = None,
    output_dir: str | None = None,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    model_override: str | None = None,
    concurrency: int = 32,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    request_timeout_seconds: float = 600.0,
    overwrite: bool = False,
) -> str:
    requests = load_batch_requests(input_file)
    if not requests:
        raise ValueError(f"No requests found in {input_file}")

    resolved_output = (
        Path(output_file)
        if output_file is not None
        else default_output_path(
            input_file,
            output_dir=output_dir,
            model_override=model_override,
        )
    )
    if resolved_output.exists() and not overwrite:
        raise FileExistsError(
            f"{resolved_output} already exists; pass overwrite=True to replace it"
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_api_key = api_key or os.getenv(api_key_env) or "EMPTY"

    started_at = time.time()
    records = asyncio.run(
        run_batch_requests(
            requests,
            base_url=base_url,
            api_key=resolved_api_key,
            model_override=model_override,
            concurrency=concurrency,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
    )

    with open(resolved_output, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - started_at
    log.info(
        "Wrote %s responses to %s in %.1fs",
        len(records),
        resolved_output,
        elapsed,
    )
    return str(resolved_output)


def run_batch_dir(
    batch_dir: str,
    input_glob: str = "inputs_*.jsonl",
    output_dir: str | None = None,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    model_override: str | None = None,
    concurrency: int = 32,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    request_timeout_seconds: float = 600.0,
    overwrite: bool = False,
) -> list[str]:
    batch_path = Path(batch_dir)
    input_files = sorted(batch_path.glob(input_glob))
    if not input_files:
        raise FileNotFoundError(
            f"No input files matching {input_glob} under {batch_path}"
        )

    outputs: list[str] = []
    for input_file in input_files:
        outputs.append(
            run_batch_file(
                input_file=str(input_file),
                output_dir=output_dir,
                base_url=base_url,
                api_key=api_key,
                api_key_env=api_key_env,
                model_override=model_override,
                concurrency=concurrency,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                request_timeout_seconds=request_timeout_seconds,
                overwrite=overwrite,
            )
        )
    return outputs


if __name__ == "__main__":
    fire.Fire(
        {
            "run_batch_file": run_batch_file,
            "run_batch_dir": run_batch_dir,
        }
    )
