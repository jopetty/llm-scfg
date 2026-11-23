"""Run OpenAI-style batch JSONL files through the OpenAI Batch API.

This is the OpenAI counterpart to ``gemini_batch.py``'s Vertex AI runner, and
it mirrors that module's interface:

1. ``submit``   uploads an ``inputs_*.jsonl`` file, creates a batch job, and
   writes a small manifest next to the input file recording the batch id.
2. ``status``   polls the job(s) recorded in one or more manifests.
3. ``collect``  downloads finished results and writes
   ``batch_<batch_id>_output.jsonl`` into the same directory, which is the
   filename convention the analysis notebooks already glob for.

The batch files produced by ``main.py gen_exp_batchfile`` are already in the
OpenAI Batch request format, so no conversion step is needed.

Authentication uses ``OPENAI_API_KEY`` from the environment or ``.env``.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import dotenv
import fire
import pyrootutils

from scfg.utils import get_logger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

PROJECT_ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

dotenv.load_dotenv(PROJECT_ROOT / ".env")

MANIFEST_SUFFIX = "_openai_batch.json"
COMPLETION_WINDOW = "24h"
ENDPOINT = "/v1/chat/completions"

TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


def manifest_path(input_file: str | Path) -> Path:
    input_path = Path(input_file)
    return input_path.with_name(f"{input_path.stem}{MANIFEST_SUFFIX}")


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def count_requests(input_file: Path) -> int:
    with open(input_file) as handle:
        return sum(1 for line in handle if line.strip())


def _find_manifests(target: str) -> list[Path]:
    """Resolve ``target`` to a list of manifest files.

    ``target`` may be a manifest, an input JSONL whose manifest sits beside it,
    or a directory containing manifests.
    """

    path = Path(target)
    if path.is_dir():
        return sorted(path.glob(f"*{MANIFEST_SUFFIX}"))
    if path.name.endswith(MANIFEST_SUFFIX):
        return [path]
    candidate = manifest_path(path)
    if candidate.exists():
        return [candidate]
    raise FileNotFoundError(f"No batch manifest found for {target}")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def submit(
    input_file: str,
    overwrite: bool = False,
) -> str:
    """Upload one batch file and create an OpenAI batch job."""
    input_path = Path(input_file)
    manifest_file = manifest_path(input_path)
    if manifest_file.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_file} already exists; pass overwrite=True to resubmit"
        )

    n_requests = count_requests(input_path)
    if not n_requests:
        raise ValueError(f"No requests found in {input_file}")

    client = openai_client()
    log.info("Uploading %s (%s requests)", input_path, f"{n_requests:,}")
    with open(input_path, "rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata={"source_file": input_path.name},
    )
    log.info("Created batch %s (status=%s)", batch.id, batch.status)

    manifest = {
        "batch_id": batch.id,
        "input_file": str(input_path),
        "input_file_id": uploaded.id,
        "n_requests": n_requests,
        "status": batch.status,
        "submitted_at": time.time(),
    }
    save_manifest(manifest_file, manifest)
    log.info("Wrote manifest to %s", manifest_file)
    return str(manifest_file)


def submit_dir(
    batch_dir: str,
    input_glob: str = "inputs_*.jsonl",
    overwrite: bool = False,
    limit: int | None = None,
) -> list[str]:
    """Submit every batch file in a directory that has no manifest yet."""
    directory = Path(batch_dir)
    candidates = [
        path
        for path in sorted(directory.glob(input_glob))
        if not path.name.endswith("_output.jsonl")
        and not path.name.endswith("_vertex.jsonl")
    ]
    if not candidates:
        raise FileNotFoundError(f"No files matching {input_glob} in {batch_dir}")

    manifests: list[str] = []
    for path in candidates:
        if manifest_path(path).exists() and not overwrite:
            log.info("Skipping %s; already submitted", path.name)
            continue
        manifests.append(submit(str(path), overwrite=overwrite))
        if limit is not None and len(manifests) >= limit:
            break
    return manifests


def status(target: str, refresh: bool = True) -> dict[str, str]:
    """Report the status of every batch recorded for ``target``."""
    client = openai_client() if refresh else None
    states: dict[str, str] = {}
    for path in _find_manifests(target):
        manifest = load_manifest(path)
        state = manifest.get("status", "unknown")
        if client is not None:
            batch = client.batches.retrieve(manifest["batch_id"])
            state = batch.status
            manifest["status"] = state
            counts = getattr(batch, "request_counts", None)
            if counts is not None:
                manifest["request_counts"] = {
                    "total": counts.total,
                    "completed": counts.completed,
                    "failed": counts.failed,
                }
            if getattr(batch, "output_file_id", None):
                manifest["output_file_id"] = batch.output_file_id
            if getattr(batch, "error_file_id", None):
                manifest["error_file_id"] = batch.error_file_id
            save_manifest(path, manifest)
        detail = manifest.get("request_counts")
        suffix = (
            f" ({detail['completed']:,}/{detail['total']:,} done,"
            f" {detail['failed']:,} failed)"
            if isinstance(detail, dict)
            else ""
        )
        log.info("%s: %s%s", path.name, state, suffix)
        states[path.name] = state
    return states


def collect(target: str) -> list[str]:
    """Download results for every finished batch recorded for ``target``."""
    client = openai_client()
    written: list[str] = []
    for path in _find_manifests(target):
        manifest = load_manifest(path)
        batch = client.batches.retrieve(manifest["batch_id"])
        if batch.status != "completed":
            log.warning(
                "%s is %s, not completed; skipping", manifest["batch_id"], batch.status
            )
            continue
        if not batch.output_file_id:
            log.warning("%s completed with no output file", manifest["batch_id"])
            continue

        out_path = Path(manifest["input_file"]).with_name(
            f"batch_{batch.id.removeprefix('batch_')}_output.jsonl"
        )
        content = client.files.content(batch.output_file_id).text
        with open(out_path, "w") as handle:
            handle.write(content if content.endswith("\n") else content + "\n")
        log.info("Wrote %s", out_path)
        written.append(str(out_path))

        if batch.error_file_id:
            err_path = out_path.with_name(f"{out_path.stem}_errors.jsonl")
            err = client.files.content(batch.error_file_id).text
            with open(err_path, "w") as handle:
                handle.write(err if err.endswith("\n") else err + "\n")
            log.warning("Batch reported errors; wrote %s", err_path)

        manifest["status"] = batch.status
        manifest["output_path"] = str(out_path)
        save_manifest(path, manifest)
    return written


def collect_ids(
    batch_ids: str | list[str],
    out_dir: str = "batches/fewshot_exp",
) -> list[str]:
    """Download results for batches submitted outside this tool.

    Use this for jobs created by hand in the OpenAI dashboard, where there is
    no local manifest. Results land in ``out_dir`` as
    ``batch_<id>_output.jsonl``, the filename the analysis notebooks expect,
    so dashboard-submitted runs are indistinguishable downstream from ones
    submitted with :func:`submit`.
    """

    ids = [batch_ids] if isinstance(batch_ids, str) else list(batch_ids)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    client = openai_client()
    written: list[str] = []
    for batch_id in ids:
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            log.warning("%s is %s, not completed; skipping", batch_id, batch.status)
            continue
        if not batch.output_file_id:
            log.warning("%s completed with no output file", batch_id)
            continue

        out_path = directory / f"batch_{batch.id.removeprefix('batch_')}_output.jsonl"
        content = client.files.content(batch.output_file_id).text
        with open(out_path, "w") as handle:
            handle.write(content if content.endswith("\n") else content + "\n")
        log.info("Wrote %s", out_path)
        written.append(str(out_path))

        if batch.error_file_id:
            err_path = out_path.with_name(f"{out_path.stem}_errors.jsonl")
            err = client.files.content(batch.error_file_id).text
            with open(err_path, "w") as handle:
                handle.write(err if err.endswith("\n") else err + "\n")
            log.warning("Batch reported errors; wrote %s", err_path)
    return written


def list_batches(limit: int = 20) -> list[dict[str, Any]]:
    """List recent batches on the account, newest first.

    Convenience for finding the ids of jobs submitted in the dashboard so
    they can be passed to :func:`collect_ids`.
    """

    client = openai_client()
    rows: list[dict[str, Any]] = []
    for batch in client.batches.list(limit=limit).data:
        counts = getattr(batch, "request_counts", None)
        row = {
            "id": batch.id,
            "status": batch.status,
            "total": getattr(counts, "total", None),
            "completed": getattr(counts, "completed", None),
            "failed": getattr(counts, "failed", None),
            "source_file": (batch.metadata or {}).get("source_file"),
        }
        rows.append(row)
        log.info(
            "%s  %-10s  %s/%s done, %s failed  %s",
            row["id"],
            row["status"],
            row["completed"],
            row["total"],
            row["failed"],
            row["source_file"] or "",
        )
    return rows


def wait(
    target: str,
    poll_seconds: float = 120.0,
    collect_when_done: bool = True,
) -> list[str]:
    """Block until every batch for ``target`` is terminal, then collect."""
    while True:
        states = status(target)
        if all(state in TERMINAL_STATES for state in states.values()):
            break
        time.sleep(poll_seconds)
    return collect(target) if collect_when_done else []


def cancel(target: str) -> dict[str, str]:
    """Cancel every non-terminal batch recorded for ``target``."""
    client = openai_client()
    results: dict[str, str] = {}
    for path in _find_manifests(target):
        manifest = load_manifest(path)
        batch = client.batches.retrieve(manifest["batch_id"])
        if batch.status in TERMINAL_STATES:
            log.info("%s already %s", batch.id, batch.status)
            results[batch.id] = batch.status
            continue
        cancelled = client.batches.cancel(batch.id)
        log.info("Cancelled %s (status=%s)", batch.id, cancelled.status)
        manifest["status"] = cancelled.status
        save_manifest(path, manifest)
        results[batch.id] = cancelled.status
    return results


if __name__ == "__main__":
    fire.Fire(
        {
            "submit": submit,
            "submit_dir": submit_dir,
            "status": status,
            "collect": collect,
            "collect_ids": collect_ids,
            "list_batches": list_batches,
            "wait": wait,
            "cancel": cancel,
        }
    )
