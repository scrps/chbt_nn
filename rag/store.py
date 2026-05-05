"""rag.store — Chroma persistent client wrapper."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("rag.store")

# Lazy chroma import so the picker can run without chroma installed.
def _chroma():  # pragma: no cover
    import chromadb
    return chromadb


class VectorStore:
    """Thin wrapper around a single Chroma persistent collection."""

    COLLECTION = "chbt_nn"

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        ch = _chroma()
        # Disable telemetry by default; we're an offline tool.
        self._client = ch.PersistentClient(
            path=str(self.path),
            settings=ch.Settings(anonymized_telemetry=False),
        )
        self._coll = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self,
               ids: list[str],
               documents: list[str],
               embeddings: list[list[float]],
               metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        self._coll.upsert(
            ids=ids, documents=documents,
            embeddings=embeddings, metadatas=metadatas,
        )

    def delete_where(self, where: dict[str, Any]) -> None:
        try:
            self._coll.delete(where=where)
        except Exception as e:  # pragma: no cover
            log.warning("delete failed where=%s: %s", where, e)

    def query(self,
              embedding: list[float],
              top_k: int = 6,
              where: dict[str, Any] | None = None) -> list[dict]:
        res = self._coll.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )
        out: list[dict] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            out.append({
                "id": ids[i] if i < len(ids) else "",
                "text": doc,
                "source": (meta or {}).get("source", ""),
                "subfolder": (meta or {}).get("subfolder", ""),
                "kind": (meta or {}).get("kind", ""),
                "mtime": (meta or {}).get("mtime", 0.0),
                "distance": dists[i] if i < len(dists) else None,
            })
        return out

    def stats(self) -> dict:
        try:
            count = self._coll.count()
        except Exception:
            count = -1
        return {"path": str(self.path), "collection": self.COLLECTION, "count": count}

    def list_sources(self) -> list[str]:
        try:
            res = self._coll.get(include=["metadatas"], limit=10000)
        except Exception:
            return []
        seen: set[str] = set()
        for m in res.get("metadatas") or []:
            if m and m.get("source"):
                seen.add(m["source"])
        return sorted(seen)
