"""rag.chunk — token-aware chunking with overlap.

Token counting is approximate (we don't ship a tokenizer just for this — the
embedding model has its own tokenizer that we'd never match exactly anyway).
We use a 1 token ≈ 4 chars heuristic, which is conservative for English text
and good enough for chunk sizing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Approximate chars-per-token for English-ish text.
CHARS_PER_TOKEN = 4

# Split on paragraph/sentence-ish boundaries first; fall back to hard cut.
_PARA_RE = re.compile(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z(])")


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int


def approx_tokens(s: str) -> int:
    return max(1, len(s) // CHARS_PER_TOKEN)


def chunk_text(text: str,
               size_tokens: int = 800,
               overlap_tokens: int = 100) -> list[Chunk]:
    """Split text into ~size_tokens chunks with ~overlap_tokens overlap.

    Tries to break on paragraph / sentence boundaries; falls back to a hard
    character cut for very long undelimited regions (e.g. CSV, code).
    """
    size_chars = size_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    if not text:
        return []
    if len(text) <= size_chars:
        return [Chunk(text=text, start_char=0, end_char=len(text))]

    # First, build a list of natural break offsets.
    breaks = [0]
    for m in _PARA_RE.finditer(text):
        breaks.append(m.end())
    breaks.append(len(text))

    chunks: list[Chunk] = []
    cur_start = 0
    while cur_start < len(text):
        target_end = cur_start + size_chars
        if target_end >= len(text):
            chunks.append(Chunk(text=text[cur_start:].strip(),
                                start_char=cur_start, end_char=len(text)))
            break

        # Find the largest natural break <= target_end.
        candidate = cur_start
        for b in breaks:
            if cur_start < b <= target_end:
                candidate = b
        if candidate <= cur_start:
            # No good natural break — hard cut.
            candidate = target_end
        body = text[cur_start:candidate].strip()
        if body:
            chunks.append(Chunk(text=body, start_char=cur_start, end_char=candidate))
        # Move forward, leaving overlap.
        cur_start = max(candidate - overlap_chars, candidate - 1)
        if cur_start <= chunks[-1].start_char:
            cur_start = candidate  # no progress; bail safely
    return chunks
