#!/bin/sh
# docker/healthcheck.sh — minimal HTTP healthcheck for the picker.
# Exits 0 on healthy, non-zero otherwise. No log output on success.
set -eu
PORT="${CHBT_NN_PORT:-8088}"
exec curl -fsS --max-time 4 "http://127.0.0.1:${PORT}/api/health" >/dev/null
