#!/usr/bin/env python3

import argparse
import sys
import uuid

import dotenv
import pyrootutils
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

PROJECT_ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
dotenv.load_dotenv(PROJECT_ROOT / ".env")


def resolve_token() -> str | None:
    hf_token = getenv("HF_TOKEN")
    hub_token = getenv("HUGGINGFACE_HUB_TOKEN")
    return hf_token or hub_token


def getenv(name: str) -> str | None:
    value = __import__("os").environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Hugging Face authentication, including private dataset "
            "repo write access."
        )
    )
    parser.add_argument(
        "read_repo_id",
        nargs="?",
        default="google/gemma-3-12b-it",
        help="Existing model repo to use for the authenticated read check.",
    )
    parser.add_argument(
        "--private-test-repo",
        default=None,
        help=(
            "Dataset repo id to use for the private write check. Defaults to "
            "<authenticated-user>/llm-scfg-auth-check-<random>."
        ),
    )
    parser.add_argument(
        "--skip-private-write-check",
        action="store_true",
        help="Only verify token identity and read access.",
    )
    parser.add_argument(
        "--keep-private-test-repo",
        action="store_true",
        help="Do not delete the private test dataset repo after the check.",
    )
    return parser.parse_args()


def check_read_access(api: HfApi, repo_id: str, token: str) -> None:
    info = api.repo_info(repo_id=repo_id, repo_type="model", token=token, timeout=30)
    private = getattr(info, "private", None)
    gated = getattr(info, "gated", None)
    sha = getattr(info, "sha", None)
    print(f"Authenticated read access OK for {repo_id}")
    print(f"private={private} gated={gated} sha={sha}")


def default_private_test_repo(api: HfApi, token: str) -> str:
    profile = api.whoami(token=token)
    username = profile.get("name")
    if not isinstance(username, str) or not username:
        raise RuntimeError("Could not determine authenticated Hugging Face username.")
    suffix = uuid.uuid4().hex[:12]
    return f"{username}/llm-scfg-auth-check-{suffix}"


def check_private_dataset_write_access(
    api: HfApi,
    repo_id: str,
    token: str,
    *,
    keep_repo: bool,
) -> None:
    created = False
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            token=token,
            exist_ok=False,
        )
        created = True
        readme = b"# llm-scfg auth check\n\nTemporary private write probe.\n"
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            path_in_repo="README.md",
            path_or_fileobj=readme,
            commit_message="Verify private dataset write access",
        )
        info = api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            timeout=30,
        )
        if getattr(info, "private", None) is not True:
            raise RuntimeError(f"Created dataset repo is not private: {repo_id}")
        print(f"Private dataset write access OK for {repo_id}")
    finally:
        if created and not keep_repo:
            api.delete_repo(
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                missing_ok=True,
            )
            print(f"Deleted private test repo {repo_id}")


def main() -> int:
    args = parse_args()
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

    api = HfApi()
    try:
        check_read_access(api, args.read_repo_id, token)
        if not args.skip_private_write_check:
            private_repo_id = args.private_test_repo or default_private_test_repo(
                api, token
            )
            check_private_dataset_write_access(
                api,
                private_repo_id,
                token,
                keep_repo=args.keep_private_test_repo,
            )
    except HfHubHTTPError as exc:
        print(
            f"HF auth check failed: {exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"HF auth check failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
