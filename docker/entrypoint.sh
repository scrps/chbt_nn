#!/bin/sh
# docker/entrypoint.sh — runtime init for the chbt_nn app image.
#
# Modes:
#   picker        (default) start the FastAPI picker
#   rag-watch     start `python -m rag.cli watch`
#   rag-ingest    one-shot ingest (useful as a one-off `docker compose run`)
#   shell         dropped intentionally; refuses unless DEBUG_SHELL=1
#
# Reads $CHBT_NN_OLLAMA_URL / $CHBT_NN_HOST / $CHBT_NN_PORT for runtime
# overrides. Configuration otherwise comes from /app/infra/serve.toml.
set -eu

mode="${1:-picker}"

# Sanity: writable state dir must exist and be ours. The compose file mounts a
# named volume here. Bail loudly if the perms are wrong rather than silently
# falling back to a bad state.
if [ ! -w "${CHBT_NN_STATE:-/var/lib/chbt_nn}" ]; then
    echo "[entrypoint] FATAL: ${CHBT_NN_STATE:-/var/lib/chbt_nn} is not writable by uid $(id -u)" >&2
    exit 64
fi

# If a Docker secret is mounted at /run/secrets/bearer_token, expose it via env
# for any code that wants to assert a token (the picker itself doesn't auth —
# Caddy does — but RAG and future tools may).
if [ -r /run/secrets/bearer_token ]; then
    CHBT_NN_BEARER_TOKEN="$(cat /run/secrets/bearer_token)"
    export CHBT_NN_BEARER_TOKEN
fi

case "$mode" in
    picker)
        # Always bind 0.0.0.0 inside the container — the *container* is on an
        # internal docker network. Host exposure is decided by compose.
        export CHBT_NN_BIND="${CHBT_NN_BIND:-0.0.0.0}"
        export CHBT_NN_PORT="${CHBT_NN_PORT:-8088}"
        export CHBT_NN_OLLAMA_URL="${CHBT_NN_OLLAMA_URL:-http://ollama:11434}"

        # Best-effort BYO model registration. Failure here is non-fatal — if
        # ollama isn't reachable yet we'll just skip and the picker boots.
        python -m serve.scan 2>&1 || true

        exec python -m serve.picker.app
        ;;
    rag-watch)
        export CHBT_NN_OLLAMA_URL="${CHBT_NN_OLLAMA_URL:-http://ollama:11434}"
        exec python -m rag.cli watch --ollama-url "${CHBT_NN_OLLAMA_URL}"
        ;;
    rag-ingest)
        export CHBT_NN_OLLAMA_URL="${CHBT_NN_OLLAMA_URL:-http://ollama:11434}"
        exec python -m rag.cli ingest --ollama-url "${CHBT_NN_OLLAMA_URL}"
        ;;
    shell)
        if [ "${DEBUG_SHELL:-0}" != "1" ]; then
            echo "[entrypoint] shell access disabled. Set DEBUG_SHELL=1 to override." >&2
            exit 1
        fi
        exec /bin/sh
        ;;
    *)
        echo "[entrypoint] unknown mode: $mode (expected: picker|rag-watch|rag-ingest|shell)" >&2
        exit 64
        ;;
esac
