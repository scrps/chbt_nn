"""eval/run_eval.py — side-by-side eval harness (PLAN.md §10).

Runs every prompt in eval/prompts.jsonl against each model in --models, and
writes results as a single JSONL plus a Markdown table for human scoring.

Usage:
    python eval/run_eval.py --models llama3.1-8b-instruct mistral-7b-instruct
    python eval/run_eval.py --models llama3.1-8b-instruct llama3.1-8b-instruct-mine \
                            --label baseline_vs_mine
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = REPO_ROOT / "eval" / "prompts.jsonl"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

log = logging.getLogger("eval")


async def generate(client: httpx.AsyncClient, url: str, model: str, prompt: str,
                   num_predict: int = 1024) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    t0 = time.time()
    r = await client.post(f"{url}/api/generate", json=payload, timeout=600.0)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    return {
        "model": model,
        "prompt": prompt,
        "response": data.get("response", ""),
        "eval_count": data.get("eval_count"),
        "eval_duration_ns": data.get("eval_duration"),
        "total_duration_ns": data.get("total_duration"),
        "wall_seconds": elapsed,
    }


async def run(models: list[str], label: str, ollama_url: str,
              prompts_path: Path, num_predict: int) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = f"{label}-{ts}" if label else ts
    jsonl_out = RESULTS_DIR / f"eval-{base}.jsonl"
    md_out = RESULTS_DIR / f"eval-{base}.md"

    prompts = [json.loads(l) for l in prompts_path.read_text().splitlines() if l.strip()]
    log.info("running %d prompts × %d models", len(prompts), len(models))

    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        with jsonl_out.open("w") as fh:
            for p in prompts:
                for m in models:
                    log.info("  [%s] %s", m, p["id"])
                    try:
                        res = await generate(client, ollama_url, m, p["prompt"],
                                             num_predict=num_predict)
                    except Exception as e:
                        log.warning("    failed: %s", e)
                        res = {"model": m, "prompt": p["prompt"], "response": f"[error: {e}]"}
                    res.update({"id": p["id"], "category": p.get("category", "")})
                    results.append(res)
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fh.flush()

    _write_markdown(md_out, prompts, models, results)
    log.info("wrote %s", jsonl_out)
    log.info("wrote %s", md_out)
    return md_out


def _write_markdown(path: Path, prompts: list[dict], models: list[str],
                    results: list[dict]) -> None:
    by_pid = {}
    for r in results:
        by_pid.setdefault(r["id"], {})[r["model"]] = r
    with path.open("w") as fh:
        fh.write(f"# eval results — {path.stem}\n\n")
        fh.write(f"models: {', '.join(models)}\n\n")
        fh.write("Score each cell 1–5 (correctness, style, length adequacy). PLAN.md §10.\n\n")
        for p in prompts:
            fh.write(f"## {p['id']} · {p.get('category','')}\n\n")
            fh.write(f"> {p['prompt']}\n\n")
            for m in models:
                r = by_pid.get(p["id"], {}).get(m)
                if not r:
                    continue
                fh.write(f"### {m}\n\n")
                tps = ""
                if r.get("eval_count") and r.get("eval_duration_ns"):
                    tps = f"  ({r['eval_count']} tok in {r['eval_duration_ns']/1e9:.1f}s = {r['eval_count']/(r['eval_duration_ns']/1e9):.1f} tok/s)"
                fh.write(f"_score_: __ /5{tps}\n\n")
                fh.write("```\n" + (r.get("response", "") or "") + "\n```\n\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="ollama model names (e.g. llama3.1-8b-instruct)")
    ap.add_argument("--label", default="")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--num-predict", type=int, default=1024)
    args = ap.parse_args()
    asyncio.run(run(args.models, args.label, args.ollama_url,
                    Path(args.prompts), args.num_predict))


if __name__ == "__main__":
    main()
