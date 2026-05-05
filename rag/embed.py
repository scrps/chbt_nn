"""rag.embed — talk to Ollama's /api/embeddings."""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable

import httpx

log = logging.getLogger("rag.embed")


class Embedder:
    def __init__(self, model: str = "nomic-embed-text",
                 ollama_url: str = "http://127.0.0.1:11434",
                 timeout: float = 60.0,
                 concurrency: int = 4):
        self.model = model
        self.url = ollama_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, read=timeout))
        self._sem = asyncio.Semaphore(concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed_one(self, text: str) -> list[float]:
        async with self._sem:
            r = await self._client.post(
                f"{self.url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            r.raise_for_status()
            return r.json().get("embedding", [])

    async def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return await asyncio.gather(*(self.embed_one(t) for t in texts))

    # Sync helpers used by the CLI.
    def embed_one_sync(self, text: str) -> list[float]:
        return asyncio.run(self.embed_one(text))

    def embed_many_sync(self, texts: list[str]) -> list[list[float]]:
        async def _run():
            try:
                return await self.embed_many(texts)
            finally:
                await self.aclose()
        return asyncio.run(_run())
