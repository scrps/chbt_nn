"""FastAPI app — the local chat picker UI + API.

PLAN.md §3a (no Open WebUI; hand-rolled minimal local UI). Bound to 127.0.0.1
by default; LAN exposure goes through Caddy with a bearer token (see
infra/serve.toml + infra/Caddyfile.example).

Endpoints:
    GET  /                         -> static index.html
    GET  /api/health
    GET  /api/models               -> list of installed Ollama models + roles
    GET  /api/rag/subfolders       -> available RAG subfolder filters
    GET  /api/conversations
    POST /api/conversations        -> create (model, title, rag_*)
    GET  /api/conversations/{id}
    PATCH /api/conversations/{id}  -> rename, change model, toggle RAG
    DELETE /api/conversations/{id}
    GET  /api/conversations/{id}/messages
    POST /api/conversations/{id}/messages  -> SSE stream of assistant tokens
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Config, load, resolve_bind_addr
from .db import Store
from .ollama_client import OllamaClient
from .rag_bridge import RagBridge

log = logging.getLogger("chbt_nn.picker")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# ----- request models
class CreateConversation(BaseModel):
    model: str | None = None
    title: str = "New conversation"
    rag_enabled: bool = False
    rag_filter: list[str] = Field(default_factory=list)


class PatchConversation(BaseModel):
    title: str | None = None
    model: str | None = None
    rag_enabled: bool | None = None
    rag_filter: list[str] | None = None


class PostMessage(BaseModel):
    content: str
    # Per-message overrides (don't persist; conversation defaults still win
    # for future messages unless caller patches the conversation).
    model: str | None = None
    rag_enabled: bool | None = None
    rag_filter: list[str] | None = None


# ----- app factory
def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.store = Store(cfg.db_abspath)
        app.state.ollama = OllamaClient(cfg.ollama.url)
        app.state.rag = RagBridge(cfg)
        log.info("picker started: ollama=%s db=%s", cfg.ollama.url, cfg.db_abspath)
        try:
            yield
        finally:
            await app.state.ollama.aclose()

    app = FastAPI(title="chbt_nn picker", lifespan=lifespan)

    # ---- static + index
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        idx = STATIC_DIR / "index.html"
        if not idx.exists():
            raise HTTPException(500, "index.html missing")
        return FileResponse(idx)

    # ---- health
    @app.get("/api/health")
    async def health(request: Request) -> dict:
        ollama_ok = bool(await request.app.state.ollama.list_models())
        return {
            "ok": True,
            "ollama": ollama_ok,
            "rag": request.app.state.rag.available(),
            "expose": cfg.network.expose,
        }

    # ---- models
    @app.get("/api/models")
    async def list_models(request: Request) -> dict:
        models = await request.app.state.ollama.list_models()
        names = [m.get("name") or m.get("model") for m in models]
        # Roles from routing.toml [tags] (PLAN.md §4b).
        tags = (cfg.routing_table.get("tags") or {})
        roles_by_model: dict[str, list[str]] = {}
        for tag, model in tags.items():
            roles_by_model.setdefault(model, []).append(tag)
        out = []
        for m in models:
            n = m.get("name") or m.get("model") or ""
            out.append({
                "name": n,
                "size": m.get("size"),
                "details": m.get("details"),
                "roles": roles_by_model.get(n, []),
                "is_finetune": n.endswith("-mine"),
            })
        return {
            "models": out,
            "default": cfg.picker.default_model,
            "fallback": cfg.picker.fallback_model,
            "available_names": names,
        }

    # ---- rag subfolders
    @app.get("/api/rag/subfolders")
    async def rag_subfolders(request: Request) -> dict:
        rag = request.app.state.rag
        return {
            "available": rag.available(),
            "subfolders": rag.list_subfolders(),
            "embed_model": cfg.rag.embed_model,
        }

    # ---- conversations CRUD
    @app.get("/api/conversations")
    async def list_convs(request: Request) -> dict:
        return {"conversations": request.app.state.store.list_conversations()}

    @app.post("/api/conversations", status_code=201)
    async def create_conv(body: CreateConversation, request: Request) -> dict:
        store: Store = request.app.state.store
        model = body.model or _pick_default_model(cfg, await request.app.state.ollama.list_models())
        return store.create_conversation(
            model=model,
            title=body.title,
            rag_enabled=body.rag_enabled,
            rag_filter=body.rag_filter,
        )

    @app.get("/api/conversations/{cid}")
    async def get_conv(cid: str, request: Request) -> dict:
        conv = request.app.state.store.get_conversation(cid)
        if not conv:
            raise HTTPException(404, "no such conversation")
        return conv

    @app.patch("/api/conversations/{cid}")
    async def patch_conv(cid: str, body: PatchConversation, request: Request) -> dict:
        store: Store = request.app.state.store
        if not store.get_conversation(cid):
            raise HTTPException(404, "no such conversation")
        fields = {k: v for k, v in body.model_dump(exclude_none=True).items()}
        return store.update_conversation(cid, **fields)  # type: ignore[return-value]

    @app.delete("/api/conversations/{cid}", status_code=204)
    async def del_conv(cid: str, request: Request) -> None:
        request.app.state.store.delete_conversation(cid)

    @app.get("/api/conversations/{cid}/messages")
    async def list_msgs(cid: str, request: Request) -> dict:
        store: Store = request.app.state.store
        if not store.get_conversation(cid):
            raise HTTPException(404, "no such conversation")
        return {"messages": store.list_messages(cid)}

    @app.post("/api/conversations/{cid}/messages")
    async def post_msg(cid: str, body: PostMessage, request: Request) -> StreamingResponse:
        store: Store = request.app.state.store
        conv = store.get_conversation(cid)
        if not conv:
            raise HTTPException(404, "no such conversation")

        model = body.model or conv["model"]
        rag_enabled = conv["rag_enabled"] if body.rag_enabled is None else body.rag_enabled
        rag_filter = conv["rag_filter"] if body.rag_filter is None else body.rag_filter

        # Persist the user's message immediately.
        store.add_message(cid, "user", body.content)

        # Build the message list to send to Ollama: full history + (optional)
        # retrieved context as a synthetic system message.
        history = store.list_messages(cid)
        oai_messages: list[dict] = []
        sources: list[str] = []

        if rag_enabled:
            rag = request.app.state.rag
            hits = await rag.retrieve(body.content, subfolders=rag_filter or None)
            if hits:
                ctx = rag.format_context(hits)
                oai_messages.append({"role": "system", "content": ctx})
                sources = [h.get("source", "") for h in hits if h.get("source")]

        for m in history:
            if m["role"] not in {"user", "assistant", "system"}:
                continue
            oai_messages.append({"role": m["role"], "content": m["content"]})

        async def event_stream():
            full = []
            try:
                yield _sse({"event": "start", "model": model, "sources": sources})
                async for chunk in request.app.state.ollama.chat_stream(model, oai_messages):
                    msg = chunk.get("message") or {}
                    piece = msg.get("content", "")
                    if piece:
                        full.append(piece)
                        yield _sse({"event": "token", "delta": piece})
                    if chunk.get("done"):
                        yield _sse({
                            "event": "done",
                            "eval_count": chunk.get("eval_count"),
                            "eval_duration": chunk.get("eval_duration"),
                            "total_duration": chunk.get("total_duration"),
                        })
                        break
            except asyncio.CancelledError:
                # Client disconnected; persist whatever we have.
                pass
            except Exception as e:  # pragma: no cover
                log.exception("chat stream failed")
                yield _sse({"event": "error", "message": str(e)})
            finally:
                if full:
                    store.add_message(
                        cid, "assistant", "".join(full), model=model,
                        sources=sources or None,
                    )

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _pick_default_model(cfg: Config, available: list[dict]) -> str:
    names = {m.get("name") or m.get("model") for m in available}
    if cfg.picker.default_model in names:
        return cfg.picker.default_model
    if cfg.picker.fallback_model in names:
        return cfg.picker.fallback_model
    fallback_chain = (cfg.routing_table.get("fallback") or [])
    for n in fallback_chain:
        if n in names:
            return n
    if available:
        first = available[0]
        return first.get("name") or first.get("model") or cfg.picker.fallback_model
    return cfg.picker.fallback_model


# ---- entry point used by infra/serve.sh
def main() -> None:
    import uvicorn
    cfg = load()
    host = resolve_bind_addr(cfg)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info("binding %s:%s (expose=%s)", host, cfg.picker.port, cfg.network.expose)
    uvicorn.run(create_app(cfg), host=host, port=cfg.picker.port, log_level="info")


if __name__ == "__main__":
    main()
