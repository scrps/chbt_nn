"""eval/score.py — collate human / judge scores into a summary.

Workflow:
  1. Run `python eval/run_eval.py --models X Y` -> writes eval-<label>.md
  2. Open the .md file, write `_score_: 4/5` (or `4 /5`) under each model.
  3. Run `python eval/score.py eval/results/eval-<label>.md` to get a summary.

Optionally, an LLM judge can score in batch with `--judge <model>`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCORE_RE = re.compile(r"_score_:\s*(\d)\s*/\s*5", re.IGNORECASE)
HEADER_PROMPT = re.compile(r"^##\s+(\S+)\s*·\s*(\S+)?", re.MULTILINE)
HEADER_MODEL = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


def parse_md(text: str) -> dict[tuple[str, str, str], int]:
    """Returns {(prompt_id, category, model): score}."""
    # Walk the file linearly and remember the last seen prompt header / model header.
    out: dict[tuple[str, str, str], int] = {}
    cur_pid = ""
    cur_cat = ""
    cur_model = ""
    for line in text.splitlines():
        m = re.match(r"^##\s+(\S+)\s*·\s*(\S*)", line)
        if m:
            cur_pid, cur_cat = m.group(1), (m.group(2) or "")
            cur_model = ""
            continue
        m = re.match(r"^###\s+(.+?)\s*$", line)
        if m:
            cur_model = m.group(1).strip()
            continue
        m = SCORE_RE.search(line)
        if m and cur_pid and cur_model:
            try:
                out[(cur_pid, cur_cat, cur_model)] = int(m.group(1))
            except ValueError:
                pass
    return out


def summarize(scores: dict[tuple[str, str, str], int]) -> str:
    by_model: dict[str, list[int]] = defaultdict(list)
    by_model_cat: dict[tuple[str, str], list[int]] = defaultdict(list)
    for (pid, cat, model), s in scores.items():
        by_model[model].append(s)
        by_model_cat[(model, cat)].append(s)

    out = ["# summary\n"]
    out.append("| model | n | avg |")
    out.append("|---|---:|---:|")
    for m, scs in sorted(by_model.items()):
        avg = sum(scs) / len(scs) if scs else 0
        out.append(f"| {m} | {len(scs)} | {avg:.2f} |")
    out.append("\n## by category\n")
    cats = sorted({c for (_, c) in by_model_cat.keys()})
    models = sorted(by_model.keys())
    out.append("| model | " + " | ".join(cats) + " |")
    out.append("|---|" + "---|" * len(cats))
    for m in models:
        cells = []
        for c in cats:
            scs = by_model_cat.get((m, c), [])
            cells.append(f"{(sum(scs)/len(scs)):.2f}" if scs else "—")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="eval-<label>.md file with scores filled in")
    args = ap.parse_args()
    text = Path(args.path).read_text()
    scores = parse_md(text)
    if not scores:
        print("no scores found — fill in `_score_: N/5` lines first", file=sys.stderr)
        sys.exit(1)
    print(summarize(scores))


if __name__ == "__main__":
    main()
