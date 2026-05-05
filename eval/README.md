# eval/

PLAN.md §10. Side-by-side evaluation of base vs. fine-tuned vs. fine-tuned+RAG.

## Files

- `prompts.jsonl` — fixed prompt set with categories: factlookup, code,
  longform, story, chat, refusal, persona. Edit / extend freely.
- `run_eval.py` — runs every prompt against every `--models` argument and
  writes `results/eval-<label>-<ts>.{jsonl,md}`.
- `score.py` — parses `_score_: N/5` markers you write into the .md and
  prints a summary table.
- `results/` — gitignored.

## Typical flow

```bash
# Compare a stock base against your fine-tune of the same base.
python eval/run_eval.py --models llama3.1-8b-instruct llama3.1-8b-instruct-mine \
       --label base_vs_mine

# Open eval/results/eval-base_vs_mine-<ts>.md, fill in _score_: N/5 cells.
python eval/score.py eval/results/eval-base_vs_mine-<ts>.md
```

A fine-tune is only "shipped" (i.e. wired up as the picker default) once it
beats stock on this harness on average.
