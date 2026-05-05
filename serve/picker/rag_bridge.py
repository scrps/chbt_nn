"""Bridge between the picker and the rag/ package.

Kept thin so the picker doesn't have to import chroma directly, and so the
rag package can evolve independently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config


class RagBridge:
    """Lazy wrapper around rag.query.Retriever.

    If RAG dependencies aren't installed (chroma not available), the bridge
    quietly disables itself instead of crashing the picker.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._retriever: Any | None = None
        self._init_error: str | None = None
        self._available: bool | None = None

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.cfg.rag.enabled:
            self._available = False
            return False
        try:
            from rag.query import Retriever  # type: ignore
            self._retriever = Retriever(
                chroma_dir=str(self.cfg.chroma_abspath),
                embed_model=self.cfg.rag.embed_model,
                ollama_url=self.cfg.ollama.url,
            )
            self._available = True
        except Exception as e:  # pragma: no cover
            self._init_error = repr(e)
            self._available = False
        return self._available

    def list_subfolders(self) -> list[str]:
        root = self.cfg.data_root_abspath
        if not root.exists():
            return []
        return sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    async def retrieve(
        self, query: str, top_k: int | None = None, subfolders: list[str] | None = None
    ) -> list[dict]:
        if not self.available():
            return []
        assert self._retriever is not None
        k = top_k or self.cfg.rag.top_k
        return await self._retriever.aquery(query, top_k=k, subfolders=subfolders)

    @staticmethod
    def format_context(hits: list[dict]) -> str:
        """Render retrieval hits into a system-message-friendly block."""
        if not hits:
            return ""
        lines = ["The following local context was retrieved for this query.",
                 "Prefer it over your own memory and cite the source path inline.",
                 ""]
        for h in hits:
            src = h.get("source", "?")
            txt = (h.get("text") or "").strip()
            lines.append(f"--- [source: {src}] ---")
            lines.append(txt)
            lines.append("")
        return "\n".join(lines)
