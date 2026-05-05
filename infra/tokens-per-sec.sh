#!/usr/bin/env bash
# infra/tokens-per-sec.sh — Phase 0 benchmark (PLAN.md §12).
#
# Measures tokens/sec for each loaded model on GPU and (optionally) CPU.
# Writes results to eval/results/bench-<date>.md.
#
# Usage:
#   ./infra/tokens-per-sec.sh                    # all known models, GPU
#   ./infra/tokens-per-sec.sh --cpu              # also benchmark CPU
#   ./infra/tokens-per-sec.sh llama3.1-8b-instruct mistral-7b-instruct

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p eval/results

DO_CPU=0
MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) DO_CPU=1; shift ;;
    -h|--help) sed -n '1,15p' "$0"; exit 0 ;;
    *) MODELS+=("$1"); shift ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=( llama3.1-8b-instruct qwen2.5-7b-instruct mistral-7b-instruct gemma2-9b-instruct )
fi

PROMPT='Write a 300-word vivid description of a thunderstorm rolling across a wheat field at dusk.'
OUT="eval/results/bench-$(date -u +%Y%m%dT%H%M%SZ).md"
{
  echo "# tokens-per-sec benchmark"
  echo
  echo "- date: $(date -u --iso-8601=seconds)"
  echo "- host: $(uname -a)"
  echo
  echo "| model | backend | eval tok/s | total s | tokens |"
  echo "|-------|---------|-----------:|--------:|-------:|"
} > "$OUT"

bench_one() {
  local model="$1"
  local backend="$2"
  local extra=""
  if [[ "$backend" == "cpu" ]]; then
    extra=',"num_gpu":0'
  fi
  local resp
  resp=$(curl -fsS http://127.0.0.1:11434/api/generate -d "$(jq -nc --arg m "$model" --arg p "$PROMPT" \
    "{model:\$m, prompt:\$p, stream:false, options:{num_predict:400$extra}}")") || return 1
  local eval_count eval_dur total_dur tps
  eval_count=$(echo "$resp" | jq '.eval_count // 0')
  eval_dur=$(echo   "$resp" | jq '.eval_duration // 0')
  total_dur=$(echo  "$resp" | jq '.total_duration // 0')
  if [[ "$eval_dur" -gt 0 ]]; then
    tps=$(awk -v c="$eval_count" -v d="$eval_dur" 'BEGIN{printf "%.1f", c / (d/1e9)}')
  else
    tps="n/a"
  fi
  local total_s
  total_s=$(awk -v d="$total_dur" 'BEGIN{printf "%.1f", d/1e9}')
  printf "| %s | %s | %s | %s | %s |\n" "$model" "$backend" "$tps" "$total_s" "$eval_count" >> "$OUT"
  printf "  %-40s %-6s %s tok/s\n" "$model" "$backend" "$tps"
}

for m in "${MODELS[@]}"; do
  echo "benchmarking $m on GPU…"
  bench_one "$m" "gpu" || echo "  failed"
  if [[ $DO_CPU -eq 1 ]]; then
    echo "benchmarking $m on CPU…"
    bench_one "$m" "cpu" || echo "  failed"
  fi
done

echo
echo "results -> $OUT"
