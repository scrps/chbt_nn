## BYO model drop-in directory (PLAN.md §4c)

Drop a model into this directory to make it appear in the picker.

### Accepted formats

- **GGUF**: a single `*.gguf` file (or a directory containing one). Will be
  registered with Ollama as `byo-<dirname>` via `serve/scan.py`.
- **HF safetensors directory**: a directory containing `config.json` and
  `*.safetensors` shards. Only useful if you intend to fine-tune it next; not
  served directly.

### Optional sidecar files

Place these next to the weights:

- `model.toml` — display name, suggested sampling params, role tag for the
  auto-router. Example:

  ```toml
  display_name = "My Custom 7B"
  role = "longform"          # one of: chat, longform, code, factlookup, story
  parameters = { temperature = 0.7, top_p = 0.9, num_ctx = 8192 }
  ```

- `template.jinja` — only needed if the model uses a non-standard chat
  template that GGUF metadata doesn't already encode.

### Size constraints

- ≤ ~9B parameters at fp16, or ≤ ~14B at Q4 to fit fully in the 6900 XT's
  16 GB of VRAM. Larger models are accepted but flagged "CPU/hybrid only" in
  the picker.

Run `python serve/scan.py` (or restart the picker — it scans on boot) to
register changes.
