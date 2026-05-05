#!/usr/bin/env bash
# docker/init-secrets.sh — generate the Docker secret(s) for the lan overlay.
#
# Idempotent: if secrets/bearer_token already exists it does nothing.
# Permissions are tightened to 0400 so the host-side file isn't world-readable.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p secrets

token_file="secrets/bearer_token"
if [[ -s "$token_file" ]]; then
    echo "[init-secrets] $token_file already exists; leaving as-is."
else
    # 256 bits of entropy, base64url, no newline.
    if command -v openssl >/dev/null 2>&1; then
        token=$(openssl rand -base64 32 | tr '+/' '-_' | tr -d '=\n')
    else
        token=$(head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=\n')
    fi
    printf '%s' "$token" > "$token_file"
    echo "[init-secrets] wrote $token_file"
    echo
    echo "Bearer token (use as: Authorization: Bearer <TOKEN>):"
    echo "  $token"
    echo
    echo "Save this somewhere safe. It won't be shown again."
fi

chmod 0400 "$token_file"

# Make sure secrets/ isn't checked in by accident.
gi="../.gitignore"
if [[ -f "$gi" ]] && ! grep -qxF 'docker/secrets/' "$gi"; then
    echo 'docker/secrets/' >> "$gi"
fi
