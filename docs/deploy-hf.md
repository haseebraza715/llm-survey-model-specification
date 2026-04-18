# Deploy to Hugging Face Spaces

## What you need

- A Hugging Face account and a **write** token (`hf_…`) with permission to create Spaces.
- This repository on disk (or a clean checkout).

## One-shot: create Space + upload

From the repository root:

```bash
export HF_TOKEN="hf_…"                    # or: export HUGGING_FACE_HUB_TOKEN="hf_…"
export HF_SPACE_REPO="you/qualitative-model-drafter"   # optional; default is <whoami>/qualitative-model-drafter

pip install "huggingface_hub>=0.26.0"
python3 scripts/push_hf_space.py
```

The script calls `create_repo(..., exist_ok=True)` with **`space_sdk="docker"`** (HF’s API no longer accepts `streamlit` here), then `upload_folder` while skipping `data/chroma/`, `outputs/`, venvs, and caches.

### If you get `403 Forbidden` on create

Your token must be allowed to **create** repositories in that namespace (fine-grained: **Repositories: write**, or a classic token with **write**). If your org forbids API creation, create the Space once in the UI:

1. Open [new Space](https://huggingface.co/new-space), pick **Docker**, name it `qualitative-model-drafter` (or your choice).
2. Then upload only:

```bash
export HF_TOKEN="hf_…"
export HF_SPACE_REPO="you/qualitative-model-drafter"
python3 scripts/push_hf_space.py --upload-only
```

## Space settings (HF UI)

- **SDK:** Docker (the repo ships a `Dockerfile` that runs Streamlit on port 7860).  
- **Hardware:** CPU Basic is enough for the demo (first build/install can take several minutes).  
- **Secrets:** Do **not** add `OPENROUTER_API_KEY`. BYOK stays in the browser session only.  
- **Variables:** None required for the public demo if embeddings use a public model (`sentence-transformers/all-MiniLM-L6-v2`).

## GitHub → Space sync (optional)

In your GitHub repo settings → Secrets → Actions, add:

- `HF_TOKEN` — same `hf_…` token  
- `HF_SPACE_REPO` — e.g. `you/qualitative-model-drafter`

Pushes to `main` then run [`.github/workflows/deploy-hf-space.yml`](../.github/workflows/deploy-hf-space.yml).

The workflow’s inline script also accepts `HUGGING_FACE_HUB_TOKEN` if you prefer that name.

## Thumbnail / GIF

Use `static/demo-provenance.gif` as the Space thumbnail (Settings → Display → Thumbnail), or replace it with a screen recording later.
