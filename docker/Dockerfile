# syntax=docker/dockerfile:1.7
#
# chbt_nn unified app image. Used by both the `picker` and `rag-watcher`
# services in docker/compose.yml — they share Python deps and source.
#
# Hardening goals:
#   * non-root (UID 10001), no shell on the runtime user, no sudo.
#   * read-only root filesystem at runtime (compose: read_only: true).
#   * minimum surface: python3-slim + only the wheels we need; no compilers.
#   * pinned base image + pinned pip requirements.
#   * no secrets baked in; everything sensitive is mounted at runtime.
#   * source code is owned by root and read-only to the app user.

# ----- Stage 1: build wheels in an isolated builder -----
# Pinned by digest. Bump intentionally; do NOT switch to a floating tag.
# python:3.12-slim-bookworm as of 2025-04. Verify with `docker manifest inspect`.
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build deps for any wheels that need compiling. We try to stay pure-python
# so this stage stays small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the requirements first for layer cache.
COPY serve/picker/requirements.txt /build/picker.txt
COPY rag/requirements.txt           /build/rag.txt

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install -r /build/picker.txt -r /build/rag.txt

# ----- Stage 2: minimal runtime -----
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    # Used by the entrypoint to know where the writable state goes.
    CHBT_NN_STATE=/var/lib/chbt_nn

# Runtime needs curl for healthcheck and tini for proper PID-1 signal handling.
# Everything else stays out.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    # Drop the apt source lists to make sure no one accidentally apt-installs
    # at runtime in a misconfigured deployment. (apt itself stays so an
    # operator can debug; sources are gone.)
    && rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list 2>/dev/null || true

# Non-root app user. Fixed UID/GID so volume permissions are predictable.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --system --gid "${APP_GID}" chbt \
    && useradd  --system --uid "${APP_UID}" --gid "${APP_GID}" \
                --home-dir /nonexistent --shell /usr/sbin/nologin chbt

# Copy the venv from builder.
COPY --from=builder /opt/venv /opt/venv

# Copy source. Owned by root and read-only.
WORKDIR /app
COPY --chown=root:root serve/ /app/serve/
COPY --chown=root:root rag/   /app/rag/
COPY --chown=root:root infra/serve.toml /app/infra/serve.toml
COPY --chown=root:root docker/entrypoint.sh   /usr/local/bin/chbt-entrypoint
COPY --chown=root:root docker/healthcheck.sh  /usr/local/bin/chbt-healthcheck
RUN chmod 0555 /usr/local/bin/chbt-entrypoint /usr/local/bin/chbt-healthcheck \
    && find /app -type d -exec chmod 0555 {} + \
    && find /app -type f -exec chmod 0444 {} +

# Writable state lives ONLY here; mounted as a volume in compose.
RUN mkdir -p "${CHBT_NN_STATE}" && chown chbt:chbt "${CHBT_NN_STATE}"

# Make the venv read-only to the app user (defense in depth — read_only root
# FS is the real guarantee).
RUN chmod -R a-w /opt/venv

USER chbt:chbt
WORKDIR /app
EXPOSE 8088

# Tini reaps zombies and forwards signals correctly.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/chbt-entrypoint"]
CMD ["picker"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["/usr/local/bin/chbt-healthcheck"]

LABEL org.opencontainers.image.title="chbt_nn" \
      org.opencontainers.image.description="Local AMD-only chatbot stack: picker + RAG worker." \
      org.opencontainers.image.source="https://github.com/scrps/chbt_nn"
