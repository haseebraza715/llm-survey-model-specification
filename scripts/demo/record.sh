#!/usr/bin/env bash
# Regenerate the recorded demo (assets/demo/demo.cast) from demo_body.sh.
#
#   ./scripts/demo/record.sh
#
# Requires asciinema (https://asciinema.org) on PATH. After recording, render
# the mp4 + preview gif with the host media pipeline, e.g.:
#
#   MEDIA_VENV=/path/to/media/.venv /path/to/media/mkdemo.sh \
#     assets/demo/demo.cast assets/demo
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ASCII="$(command -v asciinema || echo "$HOME/.local/bin/asciinema")"
if [[ ! -x "$ASCII" ]]; then
  echo "asciinema not found; install it (brew install asciinema) and re-run." >&2
  exit 1
fi

mkdir -p assets/demo
rm -f assets/demo/demo.cast
COLUMNS=100 LINES=30 "$ASCII" rec -q --overwrite assets/demo/demo.cast \
  --cols 100 --rows 30 -c "bash scripts/demo/demo_body.sh"
echo "recorded assets/demo/demo.cast ($(wc -c < assets/demo/demo.cast) bytes)"
