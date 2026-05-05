#!/usr/bin/env bash
# infra/serve.sh — start the chbt_nn picker (and confirm Ollama is up).
#
# Used as a non-systemd entry point per PLAN.md §3c. The systemd unit at
# infra/systemd/chbt_nn.service runs the same command.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "[serve] ollama not running on :11434 — starting in background"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 2
fi

# Vulkan / ROCm hint for AMD 6900 XT.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"

# Scan BYO models first so they appear in the picker.
python -m serve.scan || true

exec python -m serve.picker.app
