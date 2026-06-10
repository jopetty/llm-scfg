import asyncio
import contextlib
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


@dataclass
class ProgressState:
    total_requests: int
    completed_requests: int = 0
    succeeded_requests: int = 0
    failed_requests: int = 0


def pathsafe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_wandb_project() -> str:
    return os.getenv("WANDB_PROJECT", "llm-scfg-vllm")


def should_enable_wandb(wandb_enabled: bool | None = None) -> bool:
    if wandb_enabled is not None:
        return wandb_enabled
    if os.getenv("WANDB_MODE", "").strip().lower() == "disabled":
        return False
    return bool(os.getenv("WANDB_API_KEY")) or env_flag("WANDB_ENABLE")


def maybe_init_wandb_run(
    *,
    input_file: str,
    output_file: str,
    base_url: str,
    model_name: str | None,
    concurrency: int,
    total_requests: int,
    wandb_enabled: bool | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_name: str | None,
) -> Any | None:
    if not should_enable_wandb(wandb_enabled):
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb is not installed; skipping progress logging")
        return None

    resolved_input = Path(input_file)
    resolved_output = Path(output_file)
    return wandb.init(
        project=wandb_project or default_wandb_project(),
        entity=wandb_entity or os.getenv("WANDB_ENTITY"),
        group=wandb_group or os.getenv("WANDB_RUN_GROUP"),
        name=wandb_name or os.getenv("WANDB_RUN_NAME") or resolved_input.stem,
        config={
            "input_file": str(resolved_input),
            "output_file": str(resolved_output),
            "base_url": base_url,
            "model_name": model_name,
            "concurrency": concurrency,
            "total_requests": total_requests,
        },
    )


def log_progress(
    progress: ProgressState,
    *,
    started_at: float,
    wandb_run: Any | None,
    input_file: str,
    output_file: str,
    finished: bool = False,
) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    payload = {
        "progress/completed_requests": progress.completed_requests,
        "progress/succeeded_requests": progress.succeeded_requests,
        "progress/failed_requests": progress.failed_requests,
        "progress/remaining_requests": (
            progress.total_requests - progress.completed_requests
        ),
        "progress/total_requests": progress.total_requests,
        "progress/completion_ratio": (
            progress.completed_requests / progress.total_requests
            if progress.total_requests
            else 1.0
        ),
        "progress/requests_per_second": progress.completed_requests / elapsed,
        "timing/elapsed_seconds": elapsed,
        "finished": finished,
    }
    if wandb_run is not None:
        wandb_run.log(payload)
    log.info(
        ("Progress for %s: %s/%s complete " "(ok=%s failed=%s, %.2f req/s) -> %s"),
        input_file,
        progress.completed_requests,
        progress.total_requests,
        progress.succeeded_requests,
        progress.failed_requests,
        payload["progress/requests_per_second"],
        output_file,
    )


async def monitor_progress(
    progress: ProgressState,
    *,
    started_at: float,
    wandb_run: Any | None,
    input_file: str,
    output_file: str,
    log_interval_seconds: float,
) -> None:
    while progress.completed_requests < progress.total_requests:
        await asyncio.sleep(log_interval_seconds)
        log_progress(
            progress,
            started_at=started_at,
            wandb_run=wandb_run,
            input_file=input_file,
            output_file=output_file,
        )


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
    progress: ProgressState,
) -> dict[str, Any]:
    request_body = normalize_chat_body(request.body, model_override=model_override)
    model_name = request_body.get("model")

    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(**request_body)
                response_body = response.model_dump(mode="json")
                progress.completed_requests += 1
                progress.succeeded_requests += 1
                return build_success_record(
                    request.custom_id,
                    response_body,
                    request_id=getattr(response, "_request_id", None),
                )
            except Exception as exc:
                if attempt >= max_retries:
                    progress.completed_requests += 1
                    progress.failed_requests += 1
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
    progress: ProgressState,
    started_at: float,
    input_file: str,
    output_file: str,
    wandb_run: Any | None,
    wandb_log_interval_seconds: float,
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
            progress=progress,
        )
        for request in requests
    ]
    monitor_task = asyncio.create_task(
        monitor_progress(
            progress,
            started_at=started_at,
            wandb_run=wandb_run,
            input_file=input_file,
            output_file=output_file,
            log_interval_seconds=wandb_log_interval_seconds,
        )
    )
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task
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
    wandb_enabled: bool | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
    wandb_log_interval_seconds: float = 15.0,
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
    progress = ProgressState(total_requests=len(requests))
    started_at = time.time()
    wandb_run = maybe_init_wandb_run(
        input_file=input_file,
        output_file=str(resolved_output),
        base_url=base_url,
        model_name=model_override or requests[0].body.get("model"),
        concurrency=concurrency,
        total_requests=len(requests),
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_name=wandb_name,
    )

    try:
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
                progress=progress,
                started_at=started_at,
                input_file=input_file,
                output_file=str(resolved_output),
                wandb_run=wandb_run,
                wandb_log_interval_seconds=wandb_log_interval_seconds,
            )
        )
    finally:
        log_progress(
            progress,
            started_at=started_at,
            wandb_run=wandb_run,
            input_file=input_file,
            output_file=str(resolved_output),
            finished=True,
        )
        if wandb_run is not None:
            wandb_run.finish()

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
    wandb_enabled: bool | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_name_prefix: str | None = None,
    wandb_log_interval_seconds: float = 15.0,
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
                wandb_enabled=wandb_enabled,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_group=wandb_group or batch_path.name,
                wandb_name=(
                    f"{wandb_name_prefix}-{input_file.stem}"
                    if wandb_name_prefix
                    else input_file.stem
                ),
                wandb_log_interval_seconds=wandb_log_interval_seconds,
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
