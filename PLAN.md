# Local Chatbot — Plan & Methodology

This document outlines the approach for building a **local-only chatbot** that
can produce long-form replies, can be customized with personal data, runs on a
home server with an **AMD Radeon RX 6900 XT (16 GB VRAM)**, and is reachable
only from `localhost` or the local network.

No code is written yet — this PR is the design proposal. Implementation will
follow once the methodology is agreed.

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

**Target machine:** Linux server (assumed), AMD RX 6900 XT, 16 GB VRAM, RDNA2
(gfx1030).

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

**Conclusion on size:** primary target is a **7B–8B instruct model** for both
training and inference, with 13B as a stretch goal for inference only.

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

---

## 4. Model selection (initial shortlist)

Picked for: permissive license, strong instruct tuning, good long-context
behavior, reasonable size for a 6900 XT, and active community / GGUF
availability.

1. **Llama 3.1 8B Instruct** — strong general chat, 128k context, very well
   supported, good LoRA recipes available. **Default starting point.**
2. **Qwen2.5 7B Instruct** — competitive with Llama 3.1 8B, strong on
   reasoning and code, also 128k context.
3. **Mistral 7B Instruct v0.3** — leaner, fast, smaller context but very
   solid baseline.
4. **Gemma 2 9B Instruct** — strong quality, slightly tighter context.

Stretch / experimental:

- **Llama 3.1 / Qwen2.5 14B** at 4-bit for inference only, no fine-tuning.

We will benchmark 1–3 against our own prompts before committing to one for
fine-tuning.

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

Inputs the user is expected to bring: notes, chat logs, documents,
example Q&A pairs, code, or anything else they want the model to imitate
or know about.

Pipeline:

1. **Ingest** raw files (md, txt, pdf, docx, html, chat exports). One small
   loader per type; everything normalizes to plain text + metadata.
2. **Clean**: strip boilerplate, deduplicate, drop very short / very long
   outliers, redact obvious secrets (emails, keys, tokens) with a regex
   pass plus a manual review step.
3. **Shape into instruction/response pairs.** This is the single most
   important step. Options:
   - Hand-written `(prompt, response)` JSONL — best quality, lowest volume.
   - Semi-synthetic: take real long-form documents the user has written,
     and generate plausible prompts for them with a stronger local model,
     then hand-review.
   - Chat logs converted to multi-turn conversations in the model's chat
     template format.
4. **Split** into train / validation (e.g. 95 / 5) with a fixed seed.
5. **Format** into the exact chat template of the chosen base model
   (Llama 3 / Qwen / Mistral templates differ — getting this wrong silently
   destroys quality).
6. **Store** as JSONL under `data/` (gitignored; raw personal data must
   never be committed).

Target volume for a first useful fine-tune: a few hundred to a few thousand
high-quality examples. Quality dominates quantity at this scale.

---

## 7. Training methodology

Default recipe (subject to revision once we benchmark):

- **Method:** QLoRA (4-bit base + LoRA adapters in bf16).
- **Base:** Llama 3.1 8B Instruct (or whichever model wins the bake-off).
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

---

## 8. RAG layer (almost certainly wanted in addition to or instead of FT)

For "knows my data" behavior, RAG is dramatically cheaper to build, update,
and debug than fine-tuning, and Open WebUI / similar already include it.

- **Embeddings:** a small local embedding model (e.g. `bge-small-en-v1.5`,
  `nomic-embed-text`) served via Ollama or `llama.cpp`.
- **Vector store:** local **Chroma** or **Qdrant** in single-node mode,
  on-disk, no external service.
- **Chunking:** 500–1000 token chunks, 10–15% overlap, preserving source
  metadata.
- **Retrieval:** top-k (k≈4–8) by cosine similarity, optional reranker
  later.
- **Prompting:** retrieved chunks injected into the system / user message
  with explicit citations the model is instructed to keep.

This layer can be added *before* any fine-tuning is done, and may by itself
solve a large fraction of the problem.

---

## 9. Serving and network exposure

- Run the inference server (Ollama / `llama.cpp` server) as a systemd
  service.
- **Bind** to `127.0.0.1` for localhost-only, or to `0.0.0.0` (or the LAN
  IP) for LAN access. Default will be localhost; LAN exposure is opt-in.
- If LAN exposure is enabled:
  - Put it behind a reverse proxy (Caddy or nginx) on the same host.
  - Add **basic auth** or a shared bearer token at the proxy.
  - Restrict by source IP / subnet at the firewall (`ufw`/`nftables`).
  - Do **not** open the port to the public internet. No port forwarding.
- Logs go to journald; no telemetry leaves the box.

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
├── infra/                   # systemd units, reverse proxy config
├── serve/                   # inference server config (Ollama modelfiles, etc.)
├── rag/                     # ingestion, chunking, embedding, vector store
├── train/                   # data prep, LoRA training scripts, eval harness
├── data/                    # gitignored — personal corpora live here
└── eval/                    # prompt sets, scoring sheets, results
```

Personal data and model weights are **never** committed.

---

## 12. Phased roadmap

Each phase ends with something usable; we stop at whichever phase is "good
enough" rather than committing to all of them up front.

- **Phase 0 — Bring-up.** Install ROCm (or just Vulkan), install Ollama,
  pull Llama 3.1 8B Instruct Q4_K_M, confirm it generates on the 6900 XT,
  measure tokens/sec.
- **Phase 1 — Local serving.** Put it behind a minimal UI (Open WebUI) on
  localhost. Lock down LAN exposure if desired.
- **Phase 2 — RAG.** Stand up the embedding model + vector store, ingest
  the user's documents, wire retrieval into the chat flow.
- **Phase 3 — Evaluation harness.** Fixed prompt set + side-by-side runner.
  Establishes the baseline we have to beat.
- **Phase 4 — Data curation.** Build the instruction dataset from the
  user's materials.
- **Phase 5 — QLoRA fine-tune.** Train, merge, quantize, deploy as a new
  Ollama model tag alongside the baseline.
- **Phase 6 — Iterate.** Compare against baseline on the eval harness,
  adjust data / hyperparameters, repeat.

---

## 13. Open questions for the user

These will materially affect the methodology and are worth deciding before
any code is written:

1. **What kind of data?** Documents, chat logs, code, transcripts, a mix?
   This drives the ingestion pipeline and the dataset shape.
2. **Style vs. knowledge.** Is the goal mostly "knows my stuff" (→ RAG
   first) or mostly "answers like me / in a particular format" (→ fine-tune
   first), or both?
3. **Privacy posture.** Strict localhost only, or should other devices on
   the LAN be able to use it? Any multi-user requirements?
4. **OS on the server.** Confirming Linux (and which distro) so we can
   pick the right ROCm / Vulkan packaging path.
5. **Acceptable answer latency.** Roughly how many tokens/sec is "fast
   enough" — this influences quantization level and model size.
6. **Is full from-scratch pretraining a hard requirement** for some reason
   we haven't considered? (Default answer: no — see §1.)

Once these are answered, the next PR will start at Phase 0.
