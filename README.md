# chbt_nn

A local-only chatbot project, intended to run on a home server with an
**AMD Radeon RX 6900 XT** and be reachable from `localhost` or the LAN
only. Customizable with the user's own data, with a focus on producing
long, well-structured replies.

> **Status:** planning. No code yet.
>
> The full plan and methodology lives in **[PLAN.md](./PLAN.md)** — please
> review and comment there before any implementation starts.

## TL;DR of the plan

- **Don't train from scratch.** Start from an open-weight instruct model
  (Llama 3.1 8B / Qwen2.5 7B / Mistral 7B) and adapt it.
- **Inference** via `llama.cpp` (Vulkan) fronted by Ollama, with an
  optional Open WebUI on top.
- **"Knows my data"** → add a local **RAG** layer (embedding model +
  Chroma/Qdrant). Cheap, fast, easy to update.
- **"Sounds like me / answers in my format"** → **QLoRA** fine-tune of the
  7–8B base, then merge → GGUF → Q4_K_M for serving.
- **16 GB VRAM** is plenty for 7–8B QLoRA training and 7–13B 4-bit
  inference; full pretraining and full fine-tuning are out of scope.
- Bind to `127.0.0.1` by default; LAN exposure is opt-in behind a reverse
  proxy + auth + firewall rules. Never public.

See [PLAN.md](./PLAN.md) for the full methodology, hardware notes, model
shortlist, training recipe, RAG design, evaluation harness, phased roadmap,
and open questions.
