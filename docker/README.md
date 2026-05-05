# Docker bring-up

This is the **alternative** to running the stack natively (`infra/serve.sh`).
Both paths expose the same picker UI on `http://127.0.0.1:8088/` by default.
Use whichever you prefer; nothing about training/RAG/eval is Docker-specific.

## Quick start (CPU only, picker + ollama, loopback only)

```bash
docker compose -f docker/compose.yml up -d
# open http://127.0.0.1:8088/
```

This pulls `ollama/ollama:0.5.7` and builds the picker image locally. Models
are NOT pre-pulled; from the picker UI or the host:

```bash
docker compose -f docker/compose.yml exec ollama ollama pull llama3.1:8b-instruct-q4_K_M
docker compose -f docker/compose.yml exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
docker compose -f docker/compose.yml exec ollama ollama pull mistral:7b-instruct-q4_K_M
docker compose -f docker/compose.yml exec ollama ollama pull nomic-embed-text
```

## With AMD GPU (ROCm)

```bash
docker compose -f docker/compose.yml -f docker/compose.gpu.yml up -d
```

Requires:

* Working `amdgpu` kernel driver (`/dev/kfd`, `/dev/dri/render*` present).
* The user running `docker` is in the `video` and `render` groups, or rootful
  docker is in use.
* `compose.gpu.yml` has `HSA_OVERRIDE_GFX_VERSION=10.3.0` (RDNA2/6900 XT).
  Adjust for your card.

## With RAG worker

```bash
docker compose -f docker/compose.yml --profile rag up -d
```

The `rag-watcher` service watches `data/rag/` (mounted read-only into the
container) and re-ingests on changes. Embeddings go to a named volume
(`chbt_nn_rag_chroma`).

## With LAN exposure (Caddy + bearer token)

```bash
./docker/init-secrets.sh   # one-time: writes docker/secrets/bearer_token
docker compose \
  -f docker/compose.yml \
  -f docker/compose.lan.yml \
  --profile lan up -d
```

* Picker is **removed** from the host port.
* Caddy listens on `0.0.0.0:8443` with TLS (self-signed by Caddy's internal
  CA — trust the root CA on each client, or replace the cert).
* Every request is rejected with `401` unless it carries
  `Authorization: Bearer <token>` matching `docker/secrets/bearer_token`.
* Restrict `:8443` at the host firewall to your LAN CIDR. See
  `infra/nftables-lan.example.nft`.

## Threat model (what's actually defended)

| Risk                                          | Defense                                               |
| --------------------------------------------- | ----------------------------------------------------- |
| Container escape from picker / RAG            | non-root UID 10001, `cap_drop: ALL`, `no-new-privileges`, `read_only` rootfs, tmpfs `/tmp` mounted `nosuid,nodev,noexec`, `pids_limit`, no shell access (`shell` mode in entrypoint refuses unless `DEBUG_SHELL=1`) |
| Picker exfiltrating data to the internet      | Picker is on the `internal` docker network only — no gateway, no DNS to the public internet. Only ollama is on `egress`. |
| Ollama serving as an open relay               | `OLLAMA_HOST` only on the docker network. No host port. `OLLAMA_ORIGINS` restricted to `http://picker`. |
| Source code tampering at runtime              | `/app` and `/opt/venv` are owned by root and read-only to the app user. Compose mounts `read_only: true`. |
| Accidental LAN exposure                       | Default compose binds `127.0.0.1:8088` only. LAN requires the explicit `--profile lan` overlay AND a bearer token. |
| Token leakage via repo                        | Token is generated per-host, lives in `docker/secrets/`, which is gitignored, written `0400`. |
| Slowloris / oversized requests through Caddy  | `read_body 30s`, `read_header 10s`, `request_body max_size 8MB`. |
| Header / mixed-content attacks via UI         | CSP `default-src 'self'`, HSTS, `X-Frame-Options DENY`, `Referrer-Policy no-referrer`. |
| Caddy admin API takeover                      | Admin endpoint moved to a unix socket inside the container. |
| Supply chain (random `latest` tag)            | All images pinned to specific minor tags (`ollama/ollama:0.5.7`, `caddy:2.8-alpine`, `python:3.12-slim-bookworm`). Bump deliberately. |

### What's NOT defended

* **Host kernel exploits** — no Docker hardening defends a vulnerable host
  kernel. Keep it patched. Consider `userns-remap` in the docker daemon.
* **Hostile model weights** — if you pull a model whose weights are designed
  to exploit a parser, ollama's process is your only defense. Pin known-good
  base models via the official ollama registry.
* **Physical access** — once someone has root on the host, every defense in
  this file becomes advisory.

## File layout

```
docker/
├── Dockerfile              # multi-stage; both `picker` and `rag-watcher` run from this image
├── compose.yml             # base stack: ollama + picker (loopback only)
├── compose.gpu.yml         # AMD ROCm device passthrough overlay
├── compose.lan.yml         # Caddy + bearer-token LAN exposure overlay
├── Caddyfile               # used by the lan overlay
├── entrypoint.sh           # PID-1 init for the app image
├── healthcheck.sh          # curl-based healthcheck
├── init-secrets.sh         # one-time secret generator
└── secrets/                # gitignored; created by init-secrets.sh
```

## Useful commands

```bash
# Tail picker logs.
docker compose -f docker/compose.yml logs -f picker

# Re-ingest RAG data once (no watcher loop).
docker compose -f docker/compose.yml run --rm rag-watcher rag-ingest

# Open a shell inside the picker (refused by default — guard rail).
docker compose -f docker/compose.yml run --rm -e DEBUG_SHELL=1 picker shell

# Validate compose without starting anything.
docker compose -f docker/compose.yml config -q
docker compose -f docker/compose.yml -f docker/compose.gpu.yml \
                 -f docker/compose.lan.yml --profile rag --profile lan config -q

# Scan the built image for known CVEs (requires docker scout / trivy).
docker scout cves chbt_nn/app:latest
```

## Switching from native to docker (and back)

The two paths use **separate state** by design — Docker uses named volumes,
native uses paths under your repo / `~/.local`. To migrate a sqlite
conversations DB from native → Docker:

```bash
docker compose -f docker/compose.yml cp \
    /path/to/serve/picker/.state/conversations.db \
    picker:/var/lib/chbt_nn/conversations.db
docker compose -f docker/compose.yml restart picker
```

RAG embeddings are model-version-sensitive; re-ingest after switching paths
rather than copying the Chroma volume.
