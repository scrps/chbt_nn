"""Async Ollama client used by the picker.

Talks to Ollama's HTTP API at /api/tags, /api/chat, /api/embeddings.
We use streaming chat (/api/chat with stream=true) and forward each chunk
to the browser via SSE.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx


class OllamaClient:
    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, read=timeout))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[dict]:
        try:
            r = await self._client.get(f"{self.url}/api/tags")
            r.raise_for_status()
        except httpx.HTTPError:
            return []
        data = r.json()
        return data.get("models", [])

    async def show(self, name: str) -> dict | None:
        try:
            r = await self._client.post(f"{self.url}/api/show", json={"name": name})
            r.raise_for_status()
        except httpx.HTTPError:
            return None
        return r.json()

    async def embeddings(self, model: str, prompt: str) -> list[float]:
        r = await self._client.post(
            f"{self.url}/api/embeddings",
            json={"model": model, "prompt": prompt},
        )
        r.raise_for_status()
        return r.json().get("embedding", [])

    async def chat_stream(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
    ) -> AsyncIterator[dict]:
        """Yield parsed JSON objects from Ollama's streaming chat endpoint.

        Each object usually contains {"message": {"role": "assistant",
        "content": "..."}, "done": bool, ...}.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        async with self._client.stream(
            "POST", f"{self.url}/api/chat", json=payload
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
