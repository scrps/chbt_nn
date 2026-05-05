#!/usr/bin/env bash
# infra/bootstrap.sh — distro-detecting installer for the chbt_nn stack.
# PLAN.md §3c: pin everything by upstream artifact, not distro package.
#
# What this does:
#   1. Detects distro and installs minimum kernel/firmware/ROCm bits.
#   2. Installs Ollama (upstream binary) and ensures it's running on 127.0.0.1.
#   3. Pulls the four shortlisted base models at Q4_K_M.
#   4. Creates a Python venv at .venv/ and installs picker + RAG deps.
#   5. (Optional) sets up Caddy + bearer token + nftables rule when serve.toml
#      sets expose != "localhost".
#
# Usage:
#   ./infra/bootstrap.sh                  # full bring-up
#   ./infra/bootstrap.sh --no-models      # skip model pulls
#   ./infra/bootstrap.sh --no-system      # skip distro pkg installs
#   ./infra/bootstrap.sh --backend rocm   # override serve.toml backend
#
# Idempotent: re-running is safe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- args
DO_SYSTEM=1
DO_MODELS=1
BACKEND_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-system) DO_SYSTEM=0; shift ;;
    --no-models) DO_MODELS=0; shift ;;
    --backend)   BACKEND_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help)   sed -n '1,40p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

# ---------------------------------------------------------------- distro
DISTRO="unknown"
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  DISTRO="${ID:-unknown}"
fi
log "detected distro: $DISTRO"

PKG_INSTALL=""
case "$DISTRO" in
  arch|endeavouros|manjaro)        PKG_INSTALL="sudo pacman -S --needed --noconfirm" ;;
  debian|ubuntu|pop|linuxmint)     PKG_INSTALL="sudo apt-get install -y" ;;
  fedora|rocky|almalinux|centos)   PKG_INSTALL="sudo dnf install -y" ;;
  opensuse*|sles)                  PKG_INSTALL="sudo zypper install -y" ;;
  *) warn "unsupported distro: $DISTRO — system package step will be skipped" ;;
esac

# ---------------------------------------------------------------- backend
read_toml_value() {
  # crude TOML extractor: read_toml_value <section> <key>
  python3 - "$1" "$2" <<'PY' "$REPO_ROOT/infra/serve.toml"
import sys, tomllib
section, key, path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "rb") as fh:
    data = tomllib.load(fh)
print(data.get(section, {}).get(key, ""))
PY
}

if [[ -n "$BACKEND_OVERRIDE" ]]; then
  BACKEND="$BACKEND_OVERRIDE"
else
  BACKEND="$(read_toml_value ollama backend || echo vulkan)"
fi
[[ -z "$BACKEND" ]] && BACKEND="vulkan"
log "backend: $BACKEND"

EXPOSE="$(read_toml_value network expose || echo localhost)"
[[ -z "$EXPOSE" ]] && EXPOSE="localhost"
log "network expose: $EXPOSE"

# ---------------------------------------------------------------- system pkgs
if [[ $DO_SYSTEM -eq 1 && -n "$PKG_INSTALL" ]]; then
  case "$DISTRO" in
    arch|endeavouros|manjaro)
      $PKG_INSTALL python python-pip python-virtualenv git curl jq sqlite ;;
    debian|ubuntu|pop|linuxmint)
      sudo apt-get update
      $PKG_INSTALL python3 python3-pip python3-venv git curl jq sqlite3 ;;
    fedora|rocky|almalinux|centos)
      $PKG_INSTALL python3 python3-pip git curl jq sqlite ;;
  esac

  if [[ "$BACKEND" == "vulkan" ]]; then
    log "installing Vulkan runtime"
    case "$DISTRO" in
      arch|endeavouros|manjaro)      $PKG_INSTALL vulkan-icd-loader vulkan-radeon mesa ;;
      debian|ubuntu|pop|linuxmint)   $PKG_INSTALL libvulkan1 mesa-vulkan-drivers vulkan-tools ;;
      fedora|rocky|almalinux|centos) $PKG_INSTALL vulkan-loader mesa-vulkan-drivers vulkan-tools ;;
    esac
  elif [[ "$BACKEND" == "rocm" ]]; then
    warn "ROCm install is invasive and version-sensitive."
    warn "On Arch:    pacman -S rocm-hip-sdk rocm-opencl-sdk    (then reboot)"
    warn "On Ubuntu:  follow https://rocm.docs.amd.com/projects/install-on-linux/"
    warn "Then run:   export HSA_OVERRIDE_GFX_VERSION=10.3.0      (for 6900 XT)"
    warn "Skipping automatic ROCm install — install it manually, then re-run."
  fi
else
  log "skipping system package install"
fi

# ---------------------------------------------------------------- ollama
if ! command -v ollama >/dev/null 2>&1; then
  log "installing Ollama (upstream binary)"
  curl -fsSL https://ollama.com/install.sh | sh
else
  log "ollama already installed: $(ollama --version 2>/dev/null || echo unknown)"
fi

# Make sure ollama is reachable.
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  log "starting ollama (systemd if available, else background)"
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^ollama.service'; then
    sudo systemctl enable --now ollama || true
  else
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    sleep 3
  fi
fi
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null \
  || die "ollama is not responding on 127.0.0.1:11434"
log "ollama is up"

# ---------------------------------------------------------------- pull bases
if [[ $DO_MODELS -eq 1 ]]; then
  log "pulling base models (Q4_K_M)"
  # Tags chosen for permissive license + GGUF availability (PLAN.md §4a).
  for model in \
      "llama3.1:8b-instruct-q4_K_M" \
      "qwen2.5:7b-instruct-q4_K_M" \
      "mistral:7b-instruct-v0.3-q4_K_M" \
      "gemma2:9b-instruct-q4_K_M" \
      "nomic-embed-text"; do
    log "  pulling $model"
    ollama pull "$model" || warn "pull failed for $model — continuing"
  done

  log "registering Modelfiles"
  for mf in serve/modelfiles/*.Modelfile; do
    [[ -f "$mf" ]] || continue
    name="$(basename "$mf" .Modelfile)"
    log "  ollama create $name -f $mf"
    ollama create "$name" -f "$mf" || warn "create $name failed"
  done
else
  log "skipping model pulls"
fi

# ---------------------------------------------------------------- python venv
if [[ ! -d .venv ]]; then
  log "creating python venv at .venv/"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel >/dev/null
log "installing picker + rag requirements"
pip install -r serve/picker/requirements.txt
pip install -r rag/requirements.txt

# ---------------------------------------------------------------- LAN exposure
if [[ "$EXPOSE" == "lan" || "$EXPOSE" == "lan-open" ]]; then
  log "configuring LAN exposure ($EXPOSE)"
  TOKEN_FILE="$(read_toml_value network bearer_token_file)"
  TOKEN_FILE="${TOKEN_FILE:-/etc/chbt_nn/token}"
  if [[ ! -s "$TOKEN_FILE" ]]; then
    sudo mkdir -p "$(dirname "$TOKEN_FILE")"
    sudo sh -c "head -c 32 /dev/urandom | xxd -p -c 64 > '$TOKEN_FILE'"
    sudo chmod 600 "$TOKEN_FILE"
    log "generated bearer token at $TOKEN_FILE"
  fi
  if ! command -v caddy >/dev/null 2>&1; then
    case "$DISTRO" in
      arch|endeavouros|manjaro)      $PKG_INSTALL caddy ;;
      debian|ubuntu|pop|linuxmint)   $PKG_INSTALL caddy ;;
      fedora|rocky|almalinux|centos) $PKG_INSTALL caddy ;;
      *) warn "install caddy manually for LAN proxy" ;;
    esac
  fi
  log "see infra/Caddyfile.example and infra/nftables-lan.example.nft"
fi

log "done. Next:"
log "  ./infra/serve.sh                     # start picker + (re)check ollama"
log "  open http://127.0.0.1:$(read_toml_value picker port || echo 8088)/ in your browser"
