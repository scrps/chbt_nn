# Training notes

See PLAN.md §6 (data prep), §7 (recipe + hyperparameter rationale).

Quick reference. The hyperparameter explanations live in PLAN.md §7a.

## Software prerequisites (one-time)

```bash
# In the project venv (.venv/, created by bootstrap.sh):
source .venv/bin/activate

# ROCm wheels for PyTorch (the only step the bootstrap script intentionally
# leaves to you, because the right --index-url depends on the ROCm minor
# version installed on the host):
pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.1

pip install -r train/requirements.txt
```

Common ROCm env var on RDNA2:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

## llama.cpp (for GGUF export)

The `train/merge_and_export.py` step shells out to llama.cpp's
`convert_hf_to_gguf.py` and `llama-quantize`. Build llama.cpp once:

```bash
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cmake -B /opt/llama.cpp/build /opt/llama.cpp -DGGML_VULKAN=ON
cmake --build /opt/llama.cpp/build --config Release -j
```

Then point the script at it:

```bash
export LLAMA_CPP_DIR=/opt/llama.cpp
python -m train.merge_and_export --target llama3.1
```

## Sanity checklist before a real run

- `python -m train.prepare_data` produced a non-empty `train/out/dataset.jsonl`.
- Spot-check a few lines of that JSONL — the `messages` look right and have
  the role labels you expect (`user`, `assistant`).
- `python -m train.train_qlora --target llama3.1 --dry-run` parses fine.
- You have at least ~12 GB of free VRAM (close other GPU users first).
- For Llama 3.1 / Gemma 2 you've accepted the model license on Hugging Face
  and `huggingface-cli login`'d.
