"""rag.ingest — walk data/rag/ (and data/both/), produce normalized records.

PLAN.md §6b step 1 ("ingest raw files") and §8 ("RAG layer").

Each yielded record is a dict:
    {
      "source":   <relative posix path under data/>,
      "subfolder":<top subfolder name, e.g. "specsheets">,
      "kind":     <loader kind: md|txt|pdf|html|docx|code|json|csv>,
      "mtime":    <float, posix mtime>,
      "text":     <plain text>,
    }

Per PLAN.md §6b step 2 we are pass-through: no editorial cleaning, no
secret-redaction, no boilerplate stripping. The only mechanical normalization
applied here is: utf-8 decoding (errors=replace), CRLF→LF, dropping empty
files, deduping exact (path, sha256) duplicates within a single ingest run.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

log = logging.getLogger("rag.ingest")

CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".c", ".cpp", ".h",
    ".hpp", ".java", ".kt", ".swift", ".rb", ".php", ".lua", ".sh", ".bash",
    ".zsh", ".fish", ".sql", ".toml", ".yaml", ".yml", ".ini", ".cfg",
    ".dockerfile", ".nix", ".tf",
}
TEXT_EXTS = {".md", ".markdown", ".rst", ".txt", ".log"}


@dataclass
class Record:
    source: str
    subfolder: str
    kind: str
    mtime: float
    text: str

    def asdict(self) -> dict:
        return {
            "source": self.source, "subfolder": self.subfolder,
            "kind": self.kind, "mtime": self.mtime, "text": self.text,
        }


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_text(p: Path) -> str:
    return _normalize(p.read_text(encoding="utf-8", errors="replace"))


def _read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:  # pragma: no cover
        log.warning("pypdf not installed, skipping %s", p)
        return ""
    out: list[str] = []
    try:
        reader = PdfReader(str(p))
        for page in reader.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                continue
    except Exception as e:
        log.warning("pdf parse failed for %s: %s", p, e)
        return ""
    return _normalize("\n".join(out))


def _read_html(p: Path) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:  # pragma: no cover
        return _read_text(p)
    soup = BeautifulSoup(_read_text(p), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text("\n")


def _read_docx(p: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except ImportError:  # pragma: no cover
        log.warning("python-docx not installed, skipping %s", p)
        return ""
    doc = Document(str(p))
    return "\n".join(par.text for par in doc.paragraphs)


def _read_json(p: Path) -> str:
    raw = _read_text(p)
    try:
        obj = json.loads(raw)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw


def _read_csv(p: Path) -> str:
    out = io.StringIO()
    try:
        with p.open(encoding="utf-8", errors="replace", newline="") as fh:
            r = csv.reader(fh)
            for row in r:
                out.write("\t".join(row))
                out.write("\n")
        return out.getvalue()
    except Exception:
        return _read_text(p)


def _kind_for(p: Path) -> str | None:
    suf = p.suffix.lower()
    if suf in (".md", ".markdown", ".rst"):  return "md"
    if suf in (".txt", ".log"):              return "txt"
    if suf == ".pdf":                        return "pdf"
    if suf in (".html", ".htm"):             return "html"
    if suf == ".docx":                       return "docx"
    if suf == ".json":                       return "json"
    if suf == ".csv":                        return "csv"
    if suf in CODE_EXTS:                     return "code"
    return None


def _load(p: Path, kind: str) -> str:
    if kind == "pdf":  return _read_pdf(p)
    if kind == "html": return _read_html(p)
    if kind == "docx": return _read_docx(p)
    if kind == "json": return _read_json(p)
    if kind == "csv":  return _read_csv(p)
    return _read_text(p)


def iter_records(roots: list[Path], rel_to: Path) -> Iterator[Record]:
    """Walk the given roots and yield Records.

    `roots` are absolute. `rel_to` is the path used to compute the `source`
    field (typically the data/ directory).
    """
    seen_hashes: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            kind = _kind_for(p)
            if kind is None:
                continue
            try:
                text = _load(p, kind).strip()
            except Exception as e:
                log.warning("failed to load %s: %s", p, e)
                continue
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            try:
                rel = p.relative_to(rel_to).as_posix()
            except ValueError:
                rel = p.as_posix()
            # Subfolder = first path component after the root.
            try:
                rel_to_root = p.relative_to(root)
                subfolder = rel_to_root.parts[0] if len(rel_to_root.parts) > 1 else ""
            except ValueError:
                subfolder = ""
            yield Record(
                source=rel,
                subfolder=subfolder,
                kind=kind,
                mtime=p.stat().st_mtime,
                text=text,
            )
