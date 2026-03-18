#!/usr/bin/env python3

import json
import sys
import urllib.error
import urllib.request

import dotenv
import pyrootutils

PROJECT_ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
dotenv.load_dotenv(PROJECT_ROOT / ".env")


def resolve_token() -> str | None:
    hf_token = getenv("HF_TOKEN")
    hub_token = getenv("HUGGINGFACE_HUB_TOKEN")
    return hf_token or hub_token


def getenv(name: str) -> str | None:
    value = __import__("os").environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def check_repo_access(repo_id: str, token: str) -> tuple[int, dict]:
    url = f"https://huggingface.co/api/models/{repo_id}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "llm-scfg-hf-auth-check",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
        return response.status, json.loads(payload)


def main() -> int:
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-12b-it"
    token = resolve_token()
    if token is None:
        print(
            (
                "No Hugging Face token found. "
                "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in .env."
            ),
            file=sys.stderr,
        )
        return 1

    try:
        status, payload = check_repo_access(repo_id, token)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(
            f"HF auth check failed for {repo_id}: HTTP {exc.code}\n{body}",
            file=sys.stderr,
        )
        return 1
    except urllib.error.URLError as exc:
        print(f"HF auth check failed for {repo_id}: {exc}", file=sys.stderr)
        return 1

    print(f"Authenticated access OK for {repo_id} (HTTP {status})")
    if isinstance(payload, dict):
        private = payload.get("private")
        gated = payload.get("gated")
        sha = payload.get("sha")
        print(f"private={private} gated={gated} sha={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
