# Runbook

Day-to-day operations for the chbt_nn box.

## First-time bring-up (Phase 0–1)

```bash
git clone <this repo> /opt/chbt_nn && cd /opt/chbt_nn
./infra/bootstrap.sh                 # installs ROCm/Vulkan deps, ollama, pulls bases, creates venv
./infra/serve.sh                     # starts the picker on http://127.0.0.1:8088/
```

Open `http://127.0.0.1:8088/` in a browser. Pick a model from the dropdown,
type a message. If you see "ollama down" in the footer, run
`systemctl status ollama` (or `pgrep -af 'ollama serve'`).

## Phase 0 benchmark

```bash
./infra/tokens-per-sec.sh            # GPU only
./infra/tokens-per-sec.sh --cpu      # also bench CPU
# results -> eval/results/bench-<ts>.md
```

## Phase 2 — RAG

```bash
# Drop content into data/rag/<subfolder>/  (e.g. data/rag/specsheets/foo.pdf)
python -m rag.cli ingest             # one-shot ingest
python -m rag.cli watch &            # continuous re-ingest on mtime change
python -m rag.cli query "what's the rated load on the X-100 PSU?"
python -m rag.cli stats
```

In the UI, toggle "RAG" on the conversation header; pick subfolders from the
multi-select to scope retrieval.

## Phase 3 — eval

```bash
python eval/run_eval.py --models llama3.1-8b-instruct mistral-7b-instruct \
       --label sanity
# Open eval/results/eval-sanity-<ts>.md, fill in `_score_: N/5` cells.
python eval/score.py eval/results/eval-sanity-<ts>.md
```

## Phase 4–5 — fine-tuning

See `train/README.md`.

## Restarting things

```bash
# Picker (foreground)
./infra/serve.sh
# Picker (systemd)
sudo systemctl restart chbt_nn
# Ollama
sudo systemctl restart ollama
```

## Logs

- Picker: stdout (foreground) or `journalctl -u chbt_nn` (systemd).
- Ollama: `journalctl -u ollama` or `/tmp/ollama.log` (foreground).
- Caddy (when LAN exposed): `/var/log/caddy/chbt_nn.log`.
