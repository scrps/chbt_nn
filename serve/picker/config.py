"""Config loader for the picker.

Reads infra/serve.toml. Falls back to sensible defaults if missing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "infra" / "serve.toml"
DEFAULT_ROUTING_PATH = REPO_ROOT / "serve" / "routing.toml"


@dataclass
class NetworkConfig:
    expose: str = "localhost"
    bind_addr: str = "auto"
    lan_cidr: str = "auto"
    bearer_token_file: str = "/etc/chbt_nn/token"
    proxy_port: int = 8080


@dataclass
class OllamaConfig:
    url: str = "http://127.0.0.1:11434"
    backend: str = "vulkan"


@dataclass
class PickerConfig:
    host: str = "127.0.0.1"
    port: int = 8088
    db_path: str = "serve/picker/.state/conversations.sqlite"
    default_model: str = "llama3.1-8b-instruct-mine"
    fallback_model: str = "llama3.1:8b-instruct-q4_K_M"


@dataclass
class RagConfig:
    enabled: bool = True
    embed_model: str = "nomic-embed-text"
    chroma_dir: str = "rag/.chroma"
    data_root: str = "data/rag"
    include_both: bool = True
    top_k: int = 6
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100


@dataclass
class RouterConfig:
    enabled: bool = False
    classifier_model: str = "mistral-7b-instruct"


@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    picker: PickerConfig = field(default_factory=PickerConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    routing_table: dict = field(default_factory=dict)
    repo_root: Path = REPO_ROOT

    @property
    def db_abspath(self) -> Path:
        p = Path(self.picker.db_path)
        return p if p.is_absolute() else self.repo_root / p

    @property
    def chroma_abspath(self) -> Path:
        p = Path(self.rag.chroma_dir)
        return p if p.is_absolute() else self.repo_root / p

    @property
    def data_root_abspath(self) -> Path:
        p = Path(self.rag.data_root)
        return p if p.is_absolute() else self.repo_root / p


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load(config_path: Path = DEFAULT_CONFIG_PATH,
         routing_path: Path = DEFAULT_ROUTING_PATH) -> Config:
    raw = _load_toml(config_path)
    cfg = Config()
    if section := raw.get("network"):
        cfg.network = NetworkConfig(**{**cfg.network.__dict__, **section})
    if section := raw.get("ollama"):
        cfg.ollama = OllamaConfig(**{**cfg.ollama.__dict__, **section})
    if section := raw.get("picker"):
        cfg.picker = PickerConfig(**{**cfg.picker.__dict__, **section})
    if section := raw.get("rag"):
        cfg.rag = RagConfig(**{**cfg.rag.__dict__, **section})
    if section := raw.get("router"):
        cfg.router = RouterConfig(**{**cfg.router.__dict__, **section})

    cfg.routing_table = _load_toml(routing_path)

    # Env-var overrides for one-off testing.
    if v := os.environ.get("CHBT_NN_HOST"):
        cfg.picker.host = v
    if v := os.environ.get("CHBT_NN_PORT"):
        cfg.picker.port = int(v)
    if v := os.environ.get("CHBT_NN_OLLAMA_URL"):
        cfg.ollama.url = v
    if v := os.environ.get("CHBT_NN_DB_PATH"):
        cfg.picker.db_path = v
    if v := os.environ.get("CHBT_NN_RAG_CHROMA_DIR"):
        cfg.rag.chroma_dir = v
    return cfg


def resolve_bind_addr(cfg: Config) -> str:
    """Resolve cfg.picker.host based on network.expose policy.

    The ``CHBT_NN_BIND`` env var (set by the docker entrypoint) wins
    unconditionally — inside a container the host binding decision is made
    by docker's port mapping, not by the picker.
    """
    forced = os.environ.get("CHBT_NN_BIND")
    if forced:
        return forced
    if cfg.network.expose == "localhost":
        return "127.0.0.1"
    # The picker stays on 127.0.0.1 even on LAN; Caddy fronts it.
    return cfg.picker.host
