"""rag.query — retrieval API used by the picker."""
from __future__ import annotations

import logging
from pathlib import Path

from .embed import Embedder
from .store import VectorStore

log = logging.getLogger("rag.query")


class Retriever:
    """Embeds the query, hits chroma, returns hits in picker-friendly form."""

    def __init__(self,
                 chroma_dir: str,
                 embed_model: str = "nomic-embed-text",
                 ollama_url: str = "http://127.0.0.1:11434"):
        self.store = VectorStore(Path(chroma_dir))
        self.embedder = Embedder(model=embed_model, ollama_url=ollama_url)

    async def aquery(self,
                     query: str,
                     top_k: int = 6,
                     subfolders: list[str] | None = None) -> list[dict]:
        if not query.strip():
            return []
        try:
            emb = await self.embedder.embed_one(query)
        except Exception as e:
            log.warning("embedding failed: %s", e)
            return []
        if not emb:
            return []
        where: dict | None = None
        if subfolders:
            if len(subfolders) == 1:
                where = {"subfolder": subfolders[0]}
            else:
                where = {"$or": [{"subfolder": s} for s in subfolders]}
        try:
            return self.store.query(emb, top_k=top_k, where=where)
        except Exception as e:
            log.warning("vector store query failed: %s", e)
            return []
