# train/

PLAN.md §6, §7. Dataset prep + QLoRA fine-tune + GGUF export.

## End-to-end flow

```bash
# 1. Drop your training corpus into data/train/ (and data/both/ if shared
#    with RAG). Subfolders are free-form (chat/, email/, stories/, ...).
#
# 2. Build the SFT dataset (pass-through; no editorial cleaning).
python -m train.prepare_data
#    -> writes train/out/dataset.jsonl + dataset.val.jsonl

# 3. Pick which bases to fine-tune in train/targets.toml.
#
# 4. Train one base. Expects ROCm PyTorch installed in the venv:
#       pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.x
#    Then:
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python -m train.train_qlora --target llama3.1
#    -> writes train/runs/llama3.1-8b-instruct-mine/

# 5. Merge LoRA into the base, convert to GGUF, quantize, register with Ollama.
#    Requires a llama.cpp checkout (built) at $LLAMA_CPP_DIR (default /opt/llama.cpp).
python -m train.merge_and_export --target llama3.1
#    -> serves as `llama3.1-8b-instruct-mine` in the picker.
```

## Hyperparameters

See `targets.toml` for the defaults. The "why" for each is in PLAN.md §7a.
Quick sanity tweaks:

- VRAM tight? Reduce `max_seq_length` (2048 → 1024) before reducing `lora_r`.
- Underfitting your voice? Bump `lora_r` (16 → 32) and `num_train_epochs`.
- Loss exploding early? Lower `learning_rate` (2e-4 → 1e-4) and check the
  chat-template render — a wrong chat template is the most common cause.

## What this does NOT do

- It does not pretrain from scratch. PLAN.md §1 explicitly excludes that.
- It does not full-fine-tune. PLAN.md §2: 16 GB VRAM can't.
- It does not edit your training data. PLAN.md §6b step 2: pass-through. If
  you don't want a sentence in the model, delete it from `data/train/`.
