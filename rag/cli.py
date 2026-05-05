"""rag.cli — `python -m rag.cli ingest|query|watch|stats`.

Ingestion is idempotent: every chunk gets a stable id of
    sha256(source_path)[:16] + ":" + chunk_index
so re-ingesting a changed file replaces its chunks.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
import time
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .chunk import chunk_text
from .embed import Embedder
from .ingest import iter_records
from .query import Retriever
from .store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "infra" / "serve.toml"

log = logging.getLogger("rag.cli")


def _load_cfg() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("rb") as fh:
        return tomllib.load(fh)


def _apply_manifest(records: list[dict], manifest_path: Path) -> list[dict]:
    """Apply data/manifest.toml overrides (PLAN.md §6a).

    We only honor `mode = "ignore"` here (skip these records). RAG-vs-train
    semantics are already encoded by which directory the file lives in;
    'ignore' lets the user blacklist subpaths.
    """
    if not manifest_path.exists():
        return records
    try:
        with manifest_path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:
        return records
    overrides = data.get("override") or []
    import fnmatch
    ignore_patterns = [
        o["path"] for o in overrides
        if o.get("mode") == "ignore" and o.get("path")
    ]
    if not ignore_patterns:
        return records
    out = []
    for r in records:
        src = r["source"]
        # source is relative to data/, e.g. "rag/foo/bar.md"
        src_under_data = src.removeprefix("data/")
        if any(fnmatch.fnmatch(src_under_data, p) for p in ignore_patterns):
            continue
        out.append(r)
    return out


def _stable_id(source: str, idx: int) -> str:
    h = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"{h}:{idx}"


async def _ingest(args) -> None:
    cfg = _load_cfg()
    rag = cfg.get("rag", {})
    embed_model = args.embed_model or rag.get("embed_model", "nomic-embed-text")
    ollama_url = args.ollama_url or cfg.get("ollama", {}).get("url",
                                                              "http://127.0.0.1:11434")
    chroma_dir = REPO_ROOT / (rag.get("chroma_dir") or "rag/.chroma")
    data_root = REPO_ROOT / (rag.get("data_root") or "data/rag")
    include_both = bool(rag.get("include_both", True))
    chunk_tokens = int(rag.get("chunk_size_tokens", 800))
    overlap_tokens = int(rag.get("chunk_overlap_tokens", 100))

    roots = [data_root]
    if include_both:
        roots.append(REPO_ROOT / "data" / "both")

    log.info("ingest from %s into %s with %s",
             [str(r) for r in roots], chroma_dir, embed_model)

    records = [r.asdict() for r in iter_records(roots, REPO_ROOT)]
    records = _apply_manifest(records, REPO_ROOT / "data" / "manifest.toml")
    if not records:
        log.info("no records found.")
        return
    log.info("loaded %d records", len(records))

    store = VectorStore(chroma_dir)
    embedder = Embedder(model=embed_model, ollama_url=ollama_url)

    try:
        # Process per-source so we can drop stale chunks for changed files.
        for rec in records:
            source = rec["source"]
            chunks = chunk_text(rec["text"],
                                size_tokens=chunk_tokens,
                                overlap_tokens=overlap_tokens)
            if not chunks:
                continue
            # Drop existing chunks for this source first.
            store.delete_where({"source": source})

            ids = [_stable_id(source, i) for i in range(len(chunks))]
            docs = [c.text for c in chunks]
            metas = [{
                "source": source,
                "subfolder": rec["subfolder"] or "",
                "kind": rec["kind"],
                "mtime": float(rec["mtime"]),
                "chunk_index": i,
            } for i in range(len(chunks))]
            embs = await embedder.embed_many(docs)
            store.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
            log.info("  %s -> %d chunks", source, len(chunks))
    finally:
        await embedder.aclose()

    print("ingest complete:", store.stats())


def _watch(args) -> None:
    """Tiny polling watcher — re-runs ingest when any mtime under data_root changes."""
    cfg = _load_cfg()
    rag = cfg.get("rag", {})
    data_root = REPO_ROOT / (rag.get("data_root") or "data/rag")
    include_both = bool(rag.get("include_both", True))
    roots = [data_root]
    if include_both:
        roots.append(REPO_ROOT / "data" / "both")

    def fingerprint() -> tuple:
        out = []
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if p.is_file():
                    try:
                        out.append((str(p), p.stat().st_mtime, p.stat().st_size))
                    except OSError:
                        pass
        return tuple(sorted(out))

    last = None
    while True:
        cur = fingerprint()
        if cur != last:
            log.info("change detected; re-ingesting")
            try:
                asyncio.run(_ingest(args))
            except Exception as e:
                log.exception("ingest failed: %s", e)
            last = cur
        time.sleep(args.interval)


def _query(args) -> None:
    cfg = _load_cfg()
    rag = cfg.get("rag", {})
    chroma_dir = REPO_ROOT / (rag.get("chroma_dir") or "rag/.chroma")
    embed_model = rag.get("embed_model", "nomic-embed-text")
    ollama_url = cfg.get("ollama", {}).get("url", "http://127.0.0.1:11434")
    retriever = Retriever(str(chroma_dir), embed_model, ollama_url)
    hits = asyncio.run(retriever.aquery(
        args.query, top_k=args.top_k,
        subfolders=args.subfolder or None,
    ))
    for h in hits:
        print(f"--- {h['source']}  (subfolder={h['subfolder']}, dist={h.get('distance')})")
        print((h["text"] or "")[:500].rstrip())
        print()


def _stats(args) -> None:
    cfg = _load_cfg()
    rag = cfg.get("rag", {})
    chroma_dir = REPO_ROOT / (rag.get("chroma_dir") or "rag/.chroma")
    s = VectorStore(chroma_dir)
    print(s.stats())
    for src in s.list_sources()[:20]:
        print(" -", src)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="ingest data/rag/ into the vector store")
    pi.add_argument("--embed-model", default=None)
    pi.add_argument("--ollama-url", default=None)
    pi.set_defaults(func=lambda a: asyncio.run(_ingest(a)))

    pw = sub.add_parser("watch", help="re-ingest on changes")
    pw.add_argument("--interval", type=float, default=10.0)
    pw.add_argument("--embed-model", default=None)
    pw.add_argument("--ollama-url", default=None)
    pw.set_defaults(func=_watch)

    pq = sub.add_parser("query", help="ad-hoc query against the vector store")
    pq.add_argument("query")
    pq.add_argument("--top-k", type=int, default=6)
    pq.add_argument("--subfolder", action="append", default=[])
    pq.set_defaults(func=_query)

    ps = sub.add_parser("stats", help="show vector store stats")
    ps.set_defaults(func=_stats)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
