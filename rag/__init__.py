"""rag — local retrieval-augmented generation pipeline.

Modules:
    ingest  — walk data/rag/ (and data/both/), normalize text + metadata
    chunk   — token-aware chunking with overlap
    embed   — talk to Ollama's /api/embeddings for nomic-embed-text or similar
    store   — Chroma persistent client wrapper
    query   — Retriever used by the picker
    watch   — re-embed on mtime change
    cli     — `python -m rag.cli ingest|query|watch|stats`
"""
