"""train.train_qlora — QLoRA fine-tune one base model.

PLAN.md §7. Implements the default recipe: 4-bit base + LoRA adapters in
bf16, paged AdamW 8-bit, cosine schedule, gradient checkpointing, batch
1 × accumulation 16, 2-3 epochs with eval-on-val-loss.

Imports of heavy ML libs are kept inside `main()` so the module can be
imported on a box without GPUs (e.g. the picker host) without crashing.

Usage:
    python -m train.train_qlora --target llama3.1
    python -m train.train_qlora --target qwen2.5 --hp.num_train_epochs=2

Outputs to:
    train/runs/<output_name>/        — checkpoints + final adapter
    train/out/<output_name>/         — merged fp16 weights (after merge step)

Then run:
    python -m train.merge_and_export --target llama3.1
to merge LoRA into base and export GGUF for Ollama.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = REPO_ROOT / "train" / "targets.toml"
DATA_DIR = REPO_ROOT / "train" / "out"
RUNS_DIR = REPO_ROOT / "train" / "runs"

log = logging.getLogger("train")


def load_targets() -> dict:
    with TARGETS_PATH.open("rb") as fh:
        return tomllib.load(fh)


def find_target(targets: dict, name: str) -> dict:
    for t in targets.get("targets", []):
        if t.get("name") == name:
            merged = {**(targets.get("defaults", {}).get("hp") or {}),
                      **(t.get("hp") or {})}
            t = {**t, "hp": merged}
            return t
    raise SystemExit(f"target {name!r} not found in {TARGETS_PATH}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="name from train/targets.toml")
    ap.add_argument("--dataset", default=str(DATA_DIR / "dataset.jsonl"))
    ap.add_argument("--val-dataset", default=str(DATA_DIR / "dataset.val.jsonl"))
    ap.add_argument("--dry-run", action="store_true",
                    help="parse config + check inputs, don't train")
    args, hp_overrides = ap.parse_known_args()

    targets = load_targets()
    tgt = find_target(targets, args.target)
    if not tgt.get("enabled", True):
        log.warning("target %s is disabled in targets.toml; --enable to override", args.target)
    hp = tgt["hp"]
    # very crude --hp.k=v overrides
    for raw in hp_overrides:
        if not raw.startswith("--hp."):
            log.warning("ignoring unknown arg %s", raw)
            continue
        k, _, v = raw[5:].partition("=")
        try:
            hp[k] = json.loads(v)
        except Exception:
            hp[k] = v

    out_name = tgt["output_name"]
    run_dir = RUNS_DIR / out_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("target=%s hf=%s out=%s", args.target, tgt["hf_model"], run_dir)
    log.info("hyperparameters: %s", json.dumps(hp, indent=2))

    ds_path = Path(args.dataset)
    val_path = Path(args.val_dataset)
    if not ds_path.exists():
        raise SystemExit(f"dataset missing: {ds_path}. Run `python -m train.prepare_data` first.")

    if args.dry_run:
        log.info("--dry-run: not training. dataset=%s (%d bytes)",
                 ds_path, ds_path.stat().st_size)
        return

    # --- heavy imports
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # --- tokenizer
    tok = AutoTokenizer.from_pretrained(tgt["hf_model"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- 4-bit base
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=hp["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, hp["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=hp["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=hp["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        tgt["hf_model"],
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=hp["gradient_checkpointing"]
    )

    # --- LoRA
    target_modules = hp["lora_target_modules"]
    if target_modules == "all-linear":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- dataset
    data_files = {"train": str(ds_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)
    raw = load_dataset("json", data_files=data_files)

    def to_text(batch):
        # SFTTrainer's `dataset_text_field` expects a string column. We render
        # each conversation through the tokenizer's chat template here.
        out = []
        for msgs in batch["messages"]:
            try:
                rendered = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                rendered = "\n".join(f"[{m.get('role','user')}] {m.get('content','')}" for m in msgs)
            out.append(rendered)
        return {"text": out}

    raw = raw.map(to_text, batched=True, remove_columns=raw["train"].column_names)

    # --- TrainingArguments / SFTConfig
    sft_cfg = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        gradient_checkpointing=hp["gradient_checkpointing"],
        learning_rate=hp["learning_rate"],
        lr_scheduler_type=hp["lr_scheduler_type"],
        warmup_ratio=hp["warmup_ratio"],
        weight_decay=hp["weight_decay"],
        optim=hp["optim"],
        logging_steps=hp["logging_steps"],
        save_steps=hp["save_steps"],
        eval_steps=hp["eval_steps"],
        eval_strategy=hp["eval_strategy"] if val_path.exists() else "no",
        bf16=True,
        seed=hp["seed"],
        report_to=["tensorboard"],
        max_seq_length=hp["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=raw["train"],
        eval_dataset=raw.get("validation"),
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(str(run_dir / "adapter"))
    tok.save_pretrained(str(run_dir / "adapter"))
    log.info("done. adapter -> %s", run_dir / "adapter")


if __name__ == "__main__":
    main()
