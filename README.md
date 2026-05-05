# chbt_nn

A local-only chatbot project, intended to run on a home server with a
**Ryzen 9 5950X (16c/32t) + 64 GB RAM + AMD Radeon RX 6900 XT** on
**Arch Linux**, reachable from `localhost` by default and the LAN only
when explicitly enabled. Customizable with the user's own data, with a
focus on producing long, well-structured replies.

> **Status:** planning. No code yet.
>
> The full plan and methodology lives in **[PLAN.md](./PLAN.md)** — please
> review and comment there before any implementation starts.

## TL;DR of the plan

- **Don't train from scratch.** Start from open-weight 7–8B instruct
  models and adapt them.
- **Inference** via `llama.cpp` (Vulkan) fronted by Ollama, with Open
  WebUI on top. GPU for chat (~40–80 tok/s on 8B Q4), CPU as a fallback
  for very large models (the 64 GB of RAM lets us run 30B+ at Q4 on
  CPU when we want to).
- **Multi-model picker, not a single model.** Pull the four shortlisted
  bases (Llama 3.1 8B, Qwen2.5 7B, Mistral 7B v0.3, Gemma 2 9B), expose
  them in a picker, and allow the user to drop their own GGUF / HF
  model into `serve/models/`. Fine-tunes are produced per base.
- **Training preferred over RAG.** Voice / persona / format goes into a
  **QLoRA** fine-tune (per base, then merged → GGUF → Q4_K_M). Volatile
  or large reference content (spec sheets, datasheets, code) goes into
  a **RAG** side channel — separated on disk by `data/train/` vs.
  `data/rag/`.
- **16 GB VRAM** is plenty for 7–8B QLoRA training and 7–13B 4-bit
  inference; full pretraining and full fine-tuning are out of scope.
- **Network:** binds `127.0.0.1` by default; LAN exposure is a single
  opt-in flag in `infra/serve.toml`. Never public.

See [PLAN.md](./PLAN.md) for the full methodology, hardware notes, model
shortlist + picker, GPU-vs-CPU inference discussion, training recipe
with hyperparameter explanations, RAG design, evaluation harness, phased
roadmap, and the locked-in decisions from review round 2.
