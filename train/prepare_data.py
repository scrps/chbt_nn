"""train.prepare_data — turn data/train/ (and data/both/) into a JSONL dataset.

PLAN.md §6. Pass-through, not editorial: dedupe exact duplicates, normalize
line endings, drop empty files. No regex secret-redaction, no boilerplate
stripping. The user owns sanitization.

Output:
    train/out/dataset.jsonl   — one record per training example, in the
                                shape `{"messages": [...]}` (HF "messages"
                                column, the modern de-facto standard for SFT).

Sources understood:
    *.jsonl with a top-level "messages" key  -> passed through verbatim
    *.jsonl with "prompt"/"response" keys     -> wrapped into messages
    *.txt / *.md / *.docx / longform docs    -> turned into a single-turn
                                                 (synthetic_prompt, response)
                                                 example. The synthetic prompt
                                                 is the relative path; the
                                                 user can edit out anything
                                                 they don't want trained on.
    chat exports (whatsapp/.txt with timestamps, etc.) -> heuristic split into
                                                 multi-turn conversations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "train" / "out" / "dataset.jsonl"

log = logging.getLogger("train.prepare")

# ---- WhatsApp-style line, captured loosely.
WA_LINE = re.compile(
    r"^\s*\[?(?P<ts>\d{1,4}[/-]\d{1,2}[/-]\d{1,4}[, T]\s*\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–]?\s*"
    r"(?P<who>[^:]{1,80}):\s(?P<msg>.+)$"
)


def _load_jsonl_passthrough(p: Path) -> list[dict]:
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "messages" in obj and isinstance(obj["messages"], list):
            out.append({"messages": obj["messages"]})
        elif isinstance(obj, dict) and "prompt" in obj and "response" in obj:
            out.append({"messages": [
                {"role": "user", "content": obj["prompt"]},
                {"role": "assistant", "content": obj["response"]},
            ]})
    return out


def _wrap_longform(p: Path, text: str, source_root: Path) -> dict | None:
    text = text.strip()
    if not text:
        return None
    rel = p.relative_to(source_root).as_posix()
    # Synthetic prompt: a path-derived label. The user is free to replace this
    # with hand-written prompts; we don't fabricate content.
    prompt = f"[from {rel}]"
    return {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text},
    ]}


def _parse_chat_log(text: str) -> list[dict]:
    """Heuristic WhatsApp/iMessage-style chat → multi-turn messages.

    Two-author chats are converted into alternating user/assistant turns,
    with the *more frequent* speaker becoming "user" (representing the user
    talking to the model, since the user's voice is what we want to capture
    on the assistant side — pick the LESS frequent speaker as assistant if
    that's the user; you can flip with `--user-is-major`).

    For now: pick the lexicographically first speaker as "user". The user can
    swap roles in post if they want their voice on the assistant side.
    """
    turns: list[tuple[str, str]] = []
    cur_who, cur_buf = None, []
    for line in text.splitlines():
        m = WA_LINE.match(line)
        if m:
            if cur_who is not None and cur_buf:
                turns.append((cur_who, "\n".join(cur_buf).strip()))
            cur_who, cur_buf = m.group("who").strip(), [m.group("msg")]
        else:
            if cur_who is None:
                continue
            cur_buf.append(line)
    if cur_who is not None and cur_buf:
        turns.append((cur_who, "\n".join(cur_buf).strip()))
    if not turns:
        return []
    speakers = sorted({w for w, _ in turns})
    if len(speakers) != 2:
        return []
    user_who = speakers[0]
    msgs: list[dict] = []
    for who, body in turns:
        if not body:
            continue
        role = "user" if who == user_who else "assistant"
        msgs.append({"role": role, "content": body})
    # Convert to a single example. Long chats can blow out context; chunk by
    # rough token budget (~3000 tokens of text ≈ 12k chars) into separate
    # examples, each starting on a "user" turn.
    return _split_long_chat(msgs)


def _split_long_chat(msgs: list[dict], char_budget: int = 12000) -> list[dict]:
    out: list[dict] = []
    cur: list[dict] = []
    cur_chars = 0
    for m in msgs:
        cur.append(m)
        cur_chars += len(m["content"])
        if cur_chars >= char_budget and m["role"] == "assistant":
            out.append({"messages": cur})
            cur = []
            cur_chars = 0
    if cur and any(m["role"] == "assistant" for m in cur):
        out.append({"messages": cur})
    return out


def iter_examples(roots: list[Path], source_root: Path) -> Iterable[dict]:
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            try:
                if p.suffix.lower() == ".jsonl":
                    yield from _load_jsonl_passthrough(p)
                    continue
                if p.suffix.lower() in {".txt", ".md", ".markdown", ".rst"}:
                    text = p.read_text(encoding="utf-8", errors="replace")
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    if not text.strip():
                        continue
                    # If the text looks like a chat log, parse as such; otherwise
                    # treat as a longform single example.
                    chat = _parse_chat_log(text)
                    if chat:
                        yield from chat
                    else:
                        ex = _wrap_longform(p, text, source_root)
                        if ex:
                            yield ex
                    continue
                if p.suffix.lower() == ".docx":
                    try:
                        from docx import Document  # type: ignore
                    except ImportError:
                        log.warning("python-docx not installed; skipping %s", p)
                        continue
                    doc = Document(str(p))
                    text = "\n".join(par.text for par in doc.paragraphs)
                    ex = _wrap_longform(p, text, source_root)
                    if ex:
                        yield ex
                # other formats: skip silently — pass-through means we don't
                # try to be clever with PDFs etc. for training.
            except Exception as e:
                log.warning("failed reading %s: %s", p, e)


def dedupe(examples: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for ex in examples:
        key = hashlib.sha256(
            json.dumps(ex.get("messages", []), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    args = ap.parse_args()

    roots = [REPO_ROOT / "data" / "train", REPO_ROOT / "data" / "both"]
    examples = list(iter_examples(roots, source_root=REPO_ROOT))
    log.info("collected %d raw examples", len(examples))
    examples = dedupe(examples)
    log.info("%d after dedupe", len(examples))

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction))
    val = examples[:n_val]
    train = examples[n_val:]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for ex in train:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    val_path = out_path.with_name(out_path.stem + ".val.jsonl")
    with val_path.open("w", encoding="utf-8") as fh:
        for ex in val:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("wrote %d train -> %s", len(train), out_path)
    log.info("wrote %d val   -> %s", len(val), val_path)


if __name__ == "__main__":
    main()
