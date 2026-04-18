#!/usr/bin/env python3
"""
Create (if needed) and upload this repository to a Hugging Face Space.

Environment:
  HF_TOKEN or HUGGING_FACE_HUB_TOKEN — required (needs **write** access to create repos)
  HF_SPACE_REPO — optional, default: <whoami>/qualitative-model-drafter

Usage (from repo root):
  python3 scripts/push_hf_space.py
  python3 scripts/push_hf_space.py --repo yourname/custom-space-name
  python3 scripts/push_hf_space.py --upload-only   # Space must already exist (e.g. created in the web UI)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _token() -> str:
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()


def _ignore_patterns() -> list[str]:
    return [
        ".git/**",
        "**/__pycache__/**",
        ".env",
        ".env.*",
        "venv/**",
        ".venv/**",
        ".pytest_cache/**",
        "data/chroma/**",
        "outputs/**",
        "data/processed/**",
        "data/embedding_cache/**",
        ".cursor/**",
        "*.pyc",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/upload Hugging Face Space")
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Space repo id user/name (default: $HF_SPACE_REPO or <whoami>/qualitative-model-drafter)",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip create_repo; only upload (Space must already exist, same Dockerfile/README).",
    )
    args = parser.parse_args()

    token = _token()
    if not token:
        print(
            "Missing HF_TOKEN (or HUGGING_FACE_HUB_TOKEN). Export it or add it to .env and re-run.\n"
            "Example: export HF_TOKEN=hf_… && python3 scripts/push_hf_space.py",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        print("Install huggingface_hub: pip install 'huggingface_hub>=0.26.0'", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    who = api.whoami()
    username = str(who.get("name") or "").strip()
    if not username:
        print("Could not resolve HF username from whoami(); set HF_SPACE_REPO explicitly.", file=sys.stderr)
        sys.exit(1)

    repo_id = (args.repo or os.environ.get("HF_SPACE_REPO") or "").strip()
    if not repo_id:
        repo_id = f"{username}/qualitative-model-drafter"

    root = Path(__file__).resolve().parents[1]

    if not args.upload_only:
        print(f"Ensuring Space exists: {repo_id} …")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="docker",
                private=False,
                exist_ok=True,
            )
        except HfHubHTTPError as err:
            status = getattr(err.response, "status_code", None)
            if status == 403:
                print(
                    f"\nCould not create Space {repo_id} (403 Forbidden).\n"
                    "Common fixes:\n"
                    "  • Use a token with **write** access (Settings → Access tokens → Fine-grained: Repositories write, or a classic **write** token).\n"
                    "  • Confirm your HF account can create Spaces (verified email, not blocked).\n"
                    "  • Or create the Space once in the browser, then upload only:\n"
                    "      https://huggingface.co/new-space\n"
                    "    Choose **Docker**, name it like the repo above, leave it empty, then run:\n"
                    f"      HF_SPACE_REPO={repo_id} python3 scripts/push_hf_space.py --upload-only\n",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    print(f"Uploading {root} → {repo_id} …")
    try:
        api.upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="space",
            path_in_repo=".",
            ignore_patterns=_ignore_patterns(),
        )
    except HfHubHTTPError as err:
        status = getattr(err.response, "status_code", None)
        if status == 404:
            print(
                f"\nUpload failed: Space {repo_id} does not exist.\n"
                "Create it in the UI (Docker SDK) or fix HF_SPACE_REPO, then use --upload-only.\n",
                file=sys.stderr,
            )
        raise

    print(f"Done. Open: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
