# Local Chatbot — Plan & Methodology

This document outlines the approach for building a **local-only chatbot** that
can produce long-form replies, can be customized with personal data, runs on a
home server with an **AMD Ryzen 9 5950X (16c/32t), 64 GB system RAM, and an
AMD Radeon RX 6900 XT (16 GB VRAM)**, and is reachable only from `localhost`
or (opt-in) the local network.

No code is written yet — this PR is the design proposal. Implementation will
follow once the methodology is agreed.

> **Update (round 2):** §2, §3, §4, §6, §7, §8, §9 and §13 have been revised
> to incorporate the maintainer's feedback — full hardware spec, multi-model
> selection / BYO model, training-preferred-with-RAG-on-the-side workflow,
> hyperparameter rationale, and decided answers to all open questions.

---

## 1. High-level decision: train from scratch vs. fine-tune a base model

**Recommendation: do not train from scratch. Start from an existing
open-weight base model and adapt it.**

Reasons:

- Training a useful chat LLM from scratch requires hundreds of billions to
  trillions of tokens and clusters of high-end accelerators. A single 6900 XT
  cannot do this in any reasonable time, and the result would be far worse
  than a 7B open model that is already publicly available.
- Modern open-weight chat/instruct models (Llama 3.x, Qwen2.5, Mistral,
  Gemma 2, Phi-3, etc.) are released under licenses that allow local,
  non-commercial — and often commercial — use.
- The interesting "make it sound like me / know my data" behavior is achieved
  far more cheaply with **parameter-efficient fine-tuning (LoRA/QLoRA)** or
  **retrieval-augmented generation (RAG)** on top of a base model, both of
  which fit comfortably on a 6900 XT.

The decision tree we will follow:

1. **Do I just want it to *know* my data (docs, notes, transcripts)?**
   → Use **RAG** on top of an instruct model. No training needed. Easiest,
   fastest, easiest to update.
2. **Do I want it to *sound* a particular way / follow a particular format /
   adopt a persona / produce long structured replies reliably?**
   → Use **LoRA (or QLoRA) instruction fine-tuning** on a base instruct
   model.
3. **Both?** → Fine-tune for style/format, then layer RAG on top for facts.
   This is almost always the right answer for a personal assistant.

Full fine-tuning of all weights and pretraining from scratch are explicitly
out of scope for this hardware.

---

## 2. Hardware and software constraints

**Target machine:**

- **CPU:** AMD Ryzen 9 5950X (16 cores / 32 threads, Zen 3, AVX2)
- **RAM:** 64 GB system memory
- **GPU:** AMD Radeon RX 6900 XT, 16 GB VRAM, RDNA2 (gfx1030)
- **OS:** Arch Linux (kept distro-agnostic where possible — see §3c)

Implications:

- **No CUDA.** The two viable acceleration paths on this GPU are:
  - **ROCm** (official AMD compute stack). 6900 XT is a "consumer" RDNA2 card
    and is not on AMD's officially supported list, but it works in practice
    with `HSA_OVERRIDE_GFX_VERSION=10.3.0`. Required for PyTorch + bitsandbytes
    style training workflows.
  - **Vulkan** backend in `llama.cpp`. Doesn't need ROCm at all, very easy to
    set up, excellent for **inference** of GGUF-quantized models. Slightly
    slower than ROCm on the same card but vastly simpler.
- **VRAM budget:** 16 GB.
  - Inference of 7–8B models at 4-bit quantization: ~5–6 GB, very fast,
    plenty of room for long context.
  - Inference of 13B at 4-bit: ~8–9 GB, still comfortable.
  - Inference of ~30B at 4-bit: ~18–20 GB → won't fit fully on GPU; would
    require partial CPU offload and will be slow.
  - QLoRA fine-tuning of a 7B model: ~10–12 GB at modest sequence lengths;
    fits.
  - QLoRA fine-tuning of a 13B model: tight at 16 GB; possible with
    gradient checkpointing, small batch size, and short-ish sequences.
  - Full fine-tuning of even 7B: does **not** fit.
- **System RAM (64 GB) implications.** This is generous and unlocks two extras:
  1. **CPU-only fallback inference** of even fairly large models via
     `llama.cpp` (e.g. a 30B-class model at Q4 will fit in RAM with room to
     spare; a 70B at Q4 is ~40 GB and fits but will be slow).
  2. **Hybrid GPU+CPU offload** — `llama.cpp` can keep the hot layers on the
     6900 XT and spill the rest to RAM, which is how we'll handle anything
     that doesn't fit in 16 GB VRAM.

**Conclusion on size:** primary target is a **7B–8B instruct model** for both
training and inference, with 13B as a stretch goal for full-GPU inference and
30B+ available as a CPU/hybrid option for non-interactive tasks.

### 2a. GPU vs. CPU inference — does GPU give "higher quality"?

Short answer: **no, quality is identical for the same weights.** A model's
output quality depends on (1) the weights, (2) the quantization of those
weights, and (3) the sampling parameters. It does **not** depend on whether
the matrix multiplications happen on the GPU or the CPU. Same model + same
quant + same sampler → same logits → same answer distribution.

What the choice of backend actually changes:

- **Latency / throughput.** GPU is much faster per token. On a 6900 XT, an
  8B model at Q4_K_M will run roughly **40–80 tokens/sec**; on the 5950X
  CPU the same model runs roughly **6–12 tokens/sec**. For a 1000-token
  reply that's ~15–25 s on GPU vs. ~80–170 s on CPU.
- **Maximum model size that's practical.** GPU caps at what fits in 16 GB
  (≈13B at Q4). CPU + 64 GB RAM can host much larger models (30B–70B at
  Q4) at the cost of speed.
- **Power / heat.** GPU draws more power per token but finishes much
  sooner; CPU draws less peak power but runs longer. Total energy is
  usually a wash.

**Recommended routing** (matches the user's "flexibility and quality" goal):

| Task                                             | Backend                            |
|--------------------------------------------------|------------------------------------|
| Real-time chat                                   | GPU (latency dominates UX)         |
| Email / story / spec-sheet write-ups (long form) | GPU by default; CPU acceptable     |
| Bulk / batch jobs (re-summarizing a folder)      | CPU — frees the GPU for chat       |
| Anything > 13B that the user wants to try        | CPU or hybrid GPU+CPU offload      |

Importantly, **switching to CPU does not buy us better answers** — it only
buys us the ability to run a *bigger* model than the GPU can hold. If a
bigger model is in fact better for the task, that's an indirect quality
win; otherwise GPU is strictly better. The picker (§4) will let the user
pick model + backend per request.

---

## 3. Stack choice

Two parallel stacks, each chosen for what it is best at:

### 3a. Inference / serving stack (the chatbot itself)

Primary candidate: **`llama.cpp` with the Vulkan backend**, fronted by
**Ollama** (which wraps `llama.cpp`, manages models, and exposes an
OpenAI-compatible HTTP API).

- Pros: trivial install on AMD, GGUF quantization, OpenAI-compatible API
  makes it easy to plug any UI in front, runs as a systemd service,
  binds to `127.0.0.1` or LAN as desired.
- Alt: raw `llama.cpp` server if we want maximum control / no extra layer.
- Alt: `text-generation-webui` if we want a richer built-in chat UI.

Web UI (optional, on top of the API):
- **Open WebUI** — full-featured local chat UI with RAG, document upload,
  multi-user, runs in Docker, talks to Ollama.
- Or a minimal hand-rolled UI if we want to keep the surface area small.

### 3b. Training stack (only when we actually fine-tune)

- **PyTorch with the ROCm wheels** (`pip install torch --index-url
  https://download.pytorch.org/whl/rocm6.x`).
- **Hugging Face `transformers` + `peft` + `trl`** for LoRA / SFT.
- **`bitsandbytes`** (ROCm fork) for 4-bit base weights (QLoRA), or
  `torchao` / native bf16 if bnb proves painful on ROCm.
- **`datasets`** for data loading.
- After training: **merge LoRA into base, export to GGUF, quantize to Q4_K_M
  (or Q5_K_M)**, then load in the inference stack above.

We deliberately keep training and inference decoupled: training is a
one-shot batch job, inference is the long-running service.

### 3c. Distro / packaging note (Arch, but distro-agnostic)

The deployment target is Arch Linux, but we don't want to be locked to it.
Strategy:

- **Pin everything by upstream artifact, not distro package.** Python
  dependencies via `pip` into a project venv (or `uv`); ROCm via the
  upstream `rocm/` packages or AUR; `llama.cpp` / Ollama via their
  official binaries or built-from-source. This keeps the same setup
  reproducible on Debian/Ubuntu/Fedora if the user moves.
- **Capture the install steps as a script** (`infra/bootstrap.sh`)
  rather than as Arch-specific instructions in prose. The script
  detects the distro and chooses the right package manager / package
  names, but only does the bare minimum that has to be distro-aware
  (kernel module / firmware / `rocm` packages); everything else is
  upstream binaries or pip wheels.
- **Run the heavy things in containers** where it's painless to do so.
  Open WebUI ships as Docker; Ollama can also run in Docker on top of
  the host's ROCm. The training environment can be a single venv
  (simpler than containerizing a GPU build).
- **No systemd-only assumptions in scripts** — provide plain
  `./run.sh`-style entry points alongside the systemd units so the
  same project works on a non-systemd distro.

---

## 4. Model selection — multi-model "picker" instead of a single choice

Per the maintainer's request, we will **not** pick one model and stick with
it. Instead, the system will host **all of the shortlisted models side by
side** and let the user (or an automatic router) choose the right one for
each task. The user can also drop their own model into a directory and
have it appear in the picker, as long as it conforms to a few constraints
(see "BYO model" below).

### 4a. Initial shortlist (all of these get pulled at bring-up)

Picked for: permissive license, strong instruct tuning, good long-context
behavior, reasonable size for a 6900 XT, and active community / GGUF
availability.

| # | Model                          | Strengths                                | Suggested role         |
|---|--------------------------------|------------------------------------------|------------------------|
| 1 | **Llama 3.1 8B Instruct**      | General chat, 128k ctx, well-supported   | Default chat / persona |
| 2 | **Qwen2.5 7B Instruct**        | Reasoning, code, structured data, 128k   | Spec sheets, code, data|
| 3 | **Mistral 7B Instruct v0.3**   | Lean, fast, very solid baseline          | Fallback / quick chat  |
| 4 | **Gemma 2 9B Instruct**        | Strong quality, slightly tighter context | Long-form writing      |

Stretch / experimental (inference only, no fine-tune):

- **Qwen2.5 14B** at Q4 — fits in 16 GB with reduced context.
- **30B / 70B class** — CPU or hybrid GPU+CPU offload only (see §2a).

All four are pulled and tagged in Ollama at bring-up. Each gets its own
**Modelfile** (system prompt, sampling params, chat template confirmed)
and is also available with a "no system prompt" tag for raw use.

### 4b. Task-aware routing

There are three ways to pick a model, in increasing automation:

1. **Manual** — the picker in the UI; the user just chooses.
2. **Per-conversation default** — each conversation pins a model
   (e.g. "Code" conversation defaults to Qwen2.5; "Personal" defaults to
   the fine-tuned Llama).
3. **Auto-router** (optional, later phase) — a tiny classifier (could be a
   one-shot prompt to the *cheapest* loaded model) tags the incoming
   message as `chat | longform | code | factlookup | story` and routes
   to the model registered for that tag in `serve/routing.toml`.

We will ship (1) and (2) in early phases and treat (3) as opt-in once
the registry of fine-tuned variants stabilizes.

### 4c. BYO model

The user can drop a model into `serve/models/` and have it appear in the
picker. Constraints (kept loose deliberately):

- **Format:** GGUF (so `llama.cpp` / Ollama can load it), or HF safetensors
  in a directory if we're going to fine-tune it next.
- **Size:** ≤ ~9B parameters at fp16, or ≤ ~14B at Q4 for inference, so it
  actually fits the GPU budget. Larger models are accepted but flagged as
  "CPU/hybrid only" in the picker.
- **Chat template:** if the model has a non-standard template, the user
  drops a small `template.jinja` next to the weights; otherwise we try
  the GGUF metadata, then a small set of well-known templates.
- **Manifest:** an optional `model.toml` (display name, suggested
  sampling params, suggested role tag for the auto-router).

A `serve/scan.py` (later — not in this PR) walks `serve/models/` and
registers anything new with Ollama and the picker.

### 4d. One fine-tune per base, kept side by side

The user explicitly accepts that fine-tuning has to be repeated per base
model. The training pipeline (§7) is therefore parameterized by base
model: producing variants like

```
llama3.1-8b-instruct          (stock)
llama3.1-8b-instruct-mine     (fine-tuned on user data)
qwen2.5-7b-instruct           (stock)
qwen2.5-7b-instruct-mine      (fine-tuned on user data)
...
```

All variants are pickable. The eval harness (§10) compares stock vs.
`-mine` per base so we can tell whether the fine-tune actually helped on
that base.

---

## 5. Producing long replies

The user explicitly wants potentially long replies. Things that matter here,
in order of impact:

1. **Don't pick a model with a tiny context window.** All shortlisted models
   handle ≥ 8k tokens of context natively; Llama 3.1 / Qwen2.5 handle far
   more. We will run them with at least 8k context.
2. **Sampling parameters at inference time:**
   - Set `num_predict` / `max_tokens` to a generous value (e.g. 2048+).
   - Do not let the system prompt or chat template cut things off early.
   - Tune `repeat_penalty` carefully — too high causes premature stops on
     long outputs.
3. **Training data shape.** If we fine-tune, the dataset should *contain*
   examples of the long, structured replies we want. A model rarely
   produces longer answers than it sees during instruction tuning.
4. **System prompt.** We will define and version a system prompt that
   explicitly encourages thorough answers, structure (headings, bullets),
   and explicit "continue if more detail is helpful" behavior.

---

## 6. Data preparation (when we fine-tune)

Inputs the user is expected to bring (confirmed in §13.1): chat logs,
emails, stories, spec sheets, code, scripts, structured data, and general
documents. The pipeline is the same shape regardless of source — it just
gets a different loader per format.

**Voice / persona target** (confirmed by the maintainer): *"good
storyteller crossed with a flexible conversationalist."* The system
prompt and a hand-curated slice of the training set will be tuned to
reinforce this voice (vivid, narrative-friendly, happy to go long, but
also able to drop into precise/terse mode when the user is asking a
factual question).

### 6a. Folder convention — training vs. RAG

The maintainer wants to be able to say "this folder feeds the fine-tune,
that folder feeds RAG retrieval." So the on-disk layout is:

```
data/
├── train/        # ingested into the fine-tuning dataset
│   ├── chat/
│   ├── email/
│   ├── stories/
│   └── ...
├── rag/          # ingested into the vector store, never trained on
│   ├── specsheets/
│   ├── datasheets/
│   ├── code/
│   └── ...
└── both/         # convenience: copied into BOTH pipelines
```

A small `data/manifest.toml` lets the user override per-folder behavior
(e.g. "this folder is RAG-only even though it's under `train/`"), but the
folder names are the default and crude-but-clear classification.

### 6b. Pipeline

1. **Ingest** raw files (md, txt, pdf, docx, html, chat exports, source
   files, csv/json). One small loader per type; everything normalizes to
   plain text + metadata (source path, mtime, kind, folder-of-origin).
2. **Pass-through, not editorial.** Per the maintainer's instruction we
   do **not** substantively rewrite training content. The only automatic
   transformations applied are mechanical and reversible: deduping
   *exact* duplicates, normalizing line endings / whitespace, and dropping
   obviously-empty files. **No regex secret-redaction, no
   "boilerplate stripping", no opinionated cleaning** — the user owns
   sanitization.
3. **Shape into instruction/response pairs / multi-turn conversations.**
   This is the single most important step. Options, all preserving the
   original wording:
   - Hand-written `(prompt, response)` JSONL — best quality, lowest volume.
   - Chat / email logs converted to multi-turn conversations in the
     model's chat template format, verbatim.
   - Long-form artifacts (stories, write-ups) treated as the *response*
     side of a synthetic prompt the user reviews; the response itself is
     left untouched.
4. **Split** into train / validation (e.g. 95 / 5) with a fixed seed.
5. **Format** into the exact chat template of each target base model
   (Llama 3 / Qwen / Mistral / Gemma templates differ — getting this
   wrong silently destroys quality). Because we maintain one fine-tune
   per base (§4d), this happens once per base model from the same source
   JSONL.
6. **Store** as JSONL under `data/` (gitignored; raw personal data must
   never be committed).

Target volume for a first useful fine-tune: a few hundred to a few thousand
high-quality examples. Quality dominates quantity at this scale.

---

## 7. Training methodology

Default recipe (subject to revision once we benchmark):

- **Method:** QLoRA (4-bit base + LoRA adapters in bf16).
- **Base:** all four models in the shortlist (§4a) get the same recipe;
  the user can disable any of them in `train/targets.toml`.
- **LoRA config:** rank 16–32, alpha 32, dropout 0.05, target all linear
  projections (q/k/v/o + gate/up/down).
- **Optimizer:** paged AdamW 8-bit (or plain AdamW if bnb is troublesome on
  ROCm), cosine schedule, warmup 3%.
- **Batch:** per-device batch 1, gradient accumulation to reach effective
  batch 16–32; gradient checkpointing on.
- **Sequence length:** 2k–4k during training (long enough to teach long
  answers, short enough to fit in 16 GB).
- **Epochs:** 2–3, with early stopping on validation loss.
- **Tracking:** local TensorBoard or Weights & Biases offline mode — no
  external services required.
- **Sanity checks:** before/after generations on a fixed eval prompt set,
  diffed by hand and (optionally) judged by a stronger local model.
- **Output:** merged fp16 weights → GGUF → Q4_K_M / Q5_K_M for serving.

One full training run on ~1–3k examples should fit in well under a day on a
6900 XT.

### 7a. What those hyperparameters actually mean

The maintainer asked for an explanation of the LoRA / optimizer / batch
choices. Plain-English:

**LoRA config — `rank 16–32, alpha 32, dropout 0.05, target all linear
projections (q/k/v/o + gate/up/down)`**

- *LoRA itself.* Instead of updating the model's billions of weights
  directly (which we can't fit), LoRA freezes the original weights and
  trains **two small "adapter" matrices** next to each big matrix.
  Their product is added to the frozen weight at runtime. We end up
  training maybe ~0.5–2% as many parameters, which is what makes this
  fit on a 6900 XT.
- *Rank (r=16–32).* The "thickness" of those adapter matrices. Higher
  rank = more capacity to learn = more VRAM and more risk of
  overfitting. 16 is a safe default for stylistic fine-tunes; 32 if we
  see the model still under-fitting our voice.
- *Alpha (32).* A scaling factor on the adapter's contribution. The
  effective learning rate of the adapter is roughly `alpha / rank`, so
  `alpha=32, rank=16` gives a 2× scale; `alpha=32, rank=32` gives 1×.
  Keeping `alpha = 2 * rank` (or `= rank`) is the common, well-behaved
  recipe.
- *Dropout (0.05).* During training, randomly zero out 5% of the
  adapter's activations. Cheap regularization to reduce overfitting on
  a small personal dataset.
- *Target all linear projections (q/k/v/o + gate/up/down).* Every
  transformer block has 7 large linear layers: 4 in the attention block
  (query, key, value, output) and 3 in the MLP / feed-forward block
  (gate, up, down). Earlier LoRA papers only attached adapters to q
  and v; the modern consensus is **attach to all 7**, which gives
  noticeably better results for not much more cost.

**Optimizer — `paged AdamW 8-bit, cosine schedule, warmup 3%`**

- *AdamW.* The standard optimizer for transformer training — Adam plus
  decoupled weight decay. Robust default.
- *8-bit.* The optimizer keeps state (momentum + variance) for every
  trainable parameter, which normally takes 2× the parameter memory.
  bitsandbytes' 8-bit version stores that state in 8 bits with almost
  no quality loss — saves several GB.
- *Paged.* When VRAM gets tight, optimizer pages can be swapped to
  system RAM (which we have 64 GB of) instead of OOMing. Costs a bit
  of speed; lets us train bigger / longer.
- *Cosine schedule.* Learning rate starts at the configured peak and
  smoothly decays along a cosine curve to ~0 by the end of training.
  Empirically beats step decay for fine-tuning.
- *Warmup 3%.* For the first 3% of training steps, ramp the learning
  rate from 0 up to the peak instead of starting at full speed. Avoids
  destabilizing the model in the first few steps when gradients are
  noisy.

**Batch — `per-device batch 1, gradient accumulation to reach effective
batch 16–32; gradient checkpointing on`**

- *Per-device batch 1.* Only one example fits on the GPU at a time when
  you also have the model + optimizer state + LoRA adapters + activations
  in 16 GB.
- *Gradient accumulation to effective batch 16–32.* Run 16 (or 32)
  forward+backward passes one at a time, *summing* the gradients, then
  do a single optimizer step. The gradients you apply are mathematically
  equivalent to having processed all 16 examples in one big batch, but
  they fit in VRAM. Slower per step, same training behavior.
- *Gradient checkpointing.* Normally training keeps every layer's
  intermediate activations in VRAM so it can compute gradients on the
  way back. Checkpointing throws most of them away during the forward
  pass and **recomputes them on the backward pass** — trades ~30% more
  compute for substantially less VRAM. On a 16 GB card this is the
  difference between "fits" and "OOM".

---

## 8. RAG layer (training-preferred, but RAG is the always-fresh side channel)

The maintainer's preference is **training over RAG** for the things they
want the model to internalize (voice, style, frequently used facts), with
**RAG used as a side channel** for content that's too volatile, too
specific, or too large to be worth retraining on (spec sheets, datasheets,
fresh notes, code references).

This is reflected on disk by §6a: anything under `data/train/` feeds the
fine-tune; anything under `data/rag/` feeds only the vector store. The
two pipelines are kept fully separate so the user can move a folder from
one to the other without rebuilding both.

Implementation:

- **Embeddings:** a small local embedding model (e.g. `bge-small-en-v1.5`,
  `nomic-embed-text`) served via Ollama or `llama.cpp`.
- **Vector store:** local **Chroma** or **Qdrant** in single-node mode,
  on-disk, no external service.
- **Chunking:** 500–1000 token chunks, 10–15% overlap, preserving source
  metadata (including which `data/rag/<subfolder>` it came from).
- **Retrieval:** top-k (k≈4–8) by cosine similarity, optional reranker
  later. Subfolder filtering exposed in the UI so the user can ask
  "search only the spec sheets".
- **Prompting:** retrieved chunks injected into the system / user message
  with explicit citations the model is instructed to keep.
- **Refresh:** a watcher (or cron) re-embeds anything in `data/rag/`
  whose mtime changed. No retraining required.

RAG is also useful as a **bridge** while a fine-tune is being prepared:
the user can drop a folder into `data/rag/` and immediately get answers
about it, decide whether the model should *also* internalize it, and
later move it (or copy it via `data/both/`) to `data/train/`.

---

## 9. Serving and network exposure

- Run the inference server (Ollama / `llama.cpp` server) as a systemd
  service (with a non-systemd `./run.sh` equivalent — see §3c).
- **Default bind: `127.0.0.1`.** Localhost-only, no further setup needed,
  no surface area exposed to the LAN.
- **LAN exposure is a single opt-in flag**, kept out of the way per the
  maintainer's preference. In `infra/serve.toml`:

  ```toml
  [network]
  expose = "localhost"   # default
  # expose = "lan"       # bind to LAN IP, enable proxy + auth
  # expose = "lan-open"  # bind to LAN IP, no auth (NOT recommended)
  bind_addr = "auto"     # auto-detected LAN IP when expose != localhost
  bearer_token_file = "/etc/chbt_nn/token"
  ```

- When `expose = "lan"`, the bootstrap script automatically:
  - Puts Ollama behind a reverse proxy (Caddy by default) on the same host.
  - Generates a bearer token in the configured file if one doesn't exist.
  - Adds an `nftables`/`ufw` rule restricting the proxy port to the LAN
    subnet (auto-detected, overridable).
- Public-internet exposure is **not** offered as a config option. The
  maintainer has explicitly taken responsibility for any further
  hardening; we just provide sensible defaults.
- Single-user assumed (per §13.3). No multi-user account system.
- Logs go to journald (or stdout for the `./run.sh` path); no telemetry
  leaves the box.

---

## 10. Evaluation

We will not ship a fine-tune unless it measurably beats the untouched base
model on our use case. Evaluation harness:

1. **Fixed prompt set** (~30–50 prompts) covering: factual Qs about user
   data, long-form explanations, persona/style, refusals, edge cases.
2. **Side-by-side generations** from base vs. fine-tuned vs. fine-tuned+RAG.
3. **Scoring:** primarily human (the user) on a simple 1–5 rubric for
   correctness, style, length adequacy, and faithfulness. Optionally a
   stronger local model as a second judge.
4. **Latency / throughput** measurements on the 6900 XT at the chosen
   quantization.
5. **Regression set** kept under version control so future tweaks can be
   compared against earlier runs.

---

## 11. Repository layout (proposed, to be created in follow-up PRs)

```
.
├── PLAN.md                  # this document
├── README.md                # quick-start, links here
├── docs/                    # operational notes, runbooks
├── infra/                   # bootstrap.sh, systemd units, reverse proxy, serve.toml
├── serve/                   # inference server config
│   ├── modelfiles/          # one Ollama Modelfile per shortlisted model
│   ├── models/              # BYO model drop-in directory (gitignored)
│   ├── routing.toml         # task → model mapping (auto-router)
│   └── picker/              # UI / API for model picking
├── rag/                     # ingestion, chunking, embedding, vector store
├── train/                   # data prep, LoRA training scripts, targets.toml
├── data/                    # gitignored — personal corpora live here
│   ├── train/               # feeds the fine-tune
│   ├── rag/                 # feeds the vector store
│   ├── both/                # copied into both pipelines
│   └── manifest.toml        # per-folder overrides
└── eval/                    # prompt sets, scoring sheets, results
```

Personal data and model weights are **never** committed.

---

## 12. Phased roadmap

Each phase ends with something usable; we stop at whichever phase is "good
enough" rather than committing to all of them up front.

- **Phase 0 — Bring-up.** Install ROCm (or Vulkan) on Arch via
  `infra/bootstrap.sh`, install Ollama, pull **all four** shortlisted
  models at Q4_K_M, confirm each generates on the 6900 XT, measure
  tokens/sec per model on both GPU and CPU backends.
- **Phase 1 — Local serving + picker.** Put Ollama behind Open WebUI
  bound to `127.0.0.1`. Add the model picker (manual + per-conversation
  default). Add the LAN opt-in flag described in §9.
- **Phase 2 — RAG side channel.** Stand up the embedding model + vector
  store, wire it to `data/rag/`, expose subfolder filtering in the UI.
- **Phase 3 — Evaluation harness.** Fixed prompt set + side-by-side
  runner across all four base models. Establishes the per-model baselines
  the fine-tunes have to beat.
- **Phase 4 — Data curation.** Build the instruction dataset from
  `data/train/`, applying the pass-through rules in §6.
- **Phase 5 — QLoRA fine-tune (per base).** Run the training pipeline
  against each base in `train/targets.toml`, producing `*-mine` variants
  alongside the stock models in the picker.
- **Phase 6 — Auto-router (optional).** Add the task classifier and
  populate `serve/routing.toml`.
- **Phase 7 — Iterate.** Compare against baselines on the eval harness,
  adjust data / hyperparameters, repeat.

---

## 13. Decisions (was: open questions)

The maintainer has answered the §13 open questions from the previous
revision. Recording the decisions here so future PRs reference them
directly.

| # | Question                                  | Decision                                                                                                                      |
|---|-------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| 1 | What kind of data?                        | All of it — chat logs, emails, stories, spec sheets, code, scripts, structured data. Different bases will be tuned for different roles (Qwen for code/spec/data; Llama for chat/persona; etc. — see §4b). |
| 2 | Style vs. knowledge                       | **Both.** Fine-tune for voice/format/persona; RAG for facts that are too volatile or too specific to be worth training on (see §8). Training is preferred where a choice exists. |
| 3 | Privacy posture                           | Default **localhost only**. LAN exposure is a single config flag in `infra/serve.toml`, kept out of the way (see §9). Single user. The maintainer takes responsibility for further hardening. |
| 4 | OS on the server                          | **Arch Linux**, with a strong preference for distro-agnostic packaging in case of a future move (see §3c). |
| 5 | Acceptable latency                        | **Quality over speed**, but no "insane latency" — interactive responses should not routinely exceed ~20 s. Practically: GPU for chat (40–80 tok/s on 8B Q4); CPU acceptable for batch / very large models (see §2a). |
| 6 | Full from-scratch pretraining required?   | **No.** Base + (LoRA and/or RAG) is acceptable as long as it's effective.                                                     |

### 13a. Voice / persona target

Confirmed: *"good storyteller crossed with a flexible conversationalist."*
Reflected in the system prompt and the curated slice of training data
(see §6).

### 13b. Multi-model expectation

Confirmed: pull all shortlisted bases, expose them in a picker, allow
BYO model, accept that fine-tunes have to be repeated per base. Designed
into §4 and §7.

### 13c. Training-vs-RAG separation

Confirmed: crude folder split is fine — `data/train/`, `data/rag/`,
`data/both/`, with a `manifest.toml` for overrides. Designed into §6a.

### 13d. Data sanitization

Confirmed: the maintainer handles sanitization of training inputs. The
pipeline does **not** substantively rewrite content (no regex
secret-redaction, no boilerplate stripping). Only mechanical operations
— deduping exact duplicates, normalizing whitespace, dropping empty
files — are applied automatically. Designed into §6b step 2.

With these decisions locked in, the next PR will start at Phase 0
(bring-up + multi-model pull + tokens/sec measurements).

