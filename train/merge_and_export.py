"""train.merge_and_export — merge LoRA into base, export GGUF, register with Ollama.

PLAN.md §7 last bullet: "merged fp16 weights → GGUF → Q4_K_M / Q5_K_M for serving."

Pipeline:
    1. Load base model in fp16.
    2. Apply the LoRA adapter from train/runs/<name>/adapter/.
    3. Save merged fp16 to train/out/<name>/.
    4. Run llama.cpp's `convert_hf_to_gguf.py` to produce a fp16 GGUF.
    5. Run llama.cpp's `quantize` to produce a Q4_K_M GGUF.
    6. Generate a Modelfile that FROMs the GGUF and `ollama create`s it as
       <output_name> (e.g. llama3.1-8b-instruct-mine).

Step 1-3 are pure Python / HF. Steps 4-6 shell out to llama.cpp tools and
ollama, since we don't want to vendor llama.cpp into this repo.

Usage:
    python -m train.merge_and_export --target llama3.1
    python -m train.merge_and_export --target llama3.1 --skip-quantize
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = REPO_ROOT / "train" / "targets.toml"
RUNS_DIR = REPO_ROOT / "train" / "runs"
OUT_DIR = REPO_ROOT / "train" / "out"

log = logging.getLogger("merge")


def _load_target(name: str) -> dict:
    with TARGETS_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    for t in data.get("targets", []):
        if t.get("name") == name:
            return t
    raise SystemExit(f"target {name} not found")


def _merge(target: dict, out_path: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_dir = RUNS_DIR / target["output_name"] / "adapter"
    if not adapter_dir.exists():
        raise SystemExit(f"adapter not found: {adapter_dir}. Train first.")

    log.info("loading base %s in bf16", target["hf_model"])
    base = AutoModelForCausalLM.from_pretrained(
        target["hf_model"], torch_dtype=torch.bfloat16, device_map="cpu"
    )
    log.info("loading adapter %s", adapter_dir)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    log.info("merging…")
    merged = model.merge_and_unload()
    out_path.mkdir(parents=True, exist_ok=True)
    log.info("saving merged weights to %s", out_path)
    merged.save_pretrained(str(out_path), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    tok.save_pretrained(str(out_path))


def _to_gguf(merged_dir: Path, out_gguf: Path, llama_cpp_dir: Path) -> None:
    convert = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit(
            f"convert_hf_to_gguf.py not found at {convert}. "
            "Pass --llama-cpp /path/to/llama.cpp or install it."
        )
    log.info("converting %s -> %s (fp16)", merged_dir, out_gguf)
    subprocess.check_call([
        sys.executable, str(convert),
        str(merged_dir), "--outfile", str(out_gguf), "--outtype", "f16",
    ])


def _quantize(in_gguf: Path, out_gguf: Path, llama_cpp_dir: Path,
              quant: str = "Q4_K_M") -> None:
    # llama.cpp's quantize binary; common locations:
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "quantize",
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "quantize",
    ]
    quantize_bin = next((p for p in candidates if p.exists()), None)
    if quantize_bin is None:
        raise SystemExit(
            f"llama-quantize binary not found in {llama_cpp_dir}. "
            "Build llama.cpp first."
        )
    log.info("quantizing %s -> %s (%s)", in_gguf, out_gguf, quant)
    subprocess.check_call([str(quantize_bin), str(in_gguf), str(out_gguf), quant])


def _ollama_register(name: str, gguf_path: Path, system_prompt_file: Path | None) -> None:
    mf_lines = [f"FROM {gguf_path}"]
    if system_prompt_file and system_prompt_file.exists():
        prompt = system_prompt_file.read_text(encoding="utf-8").strip()
        mf_lines.append('SYSTEM """\n' + prompt + '\n"""')
    mf_lines += [
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.05",
        "PARAMETER num_ctx 8192",
        "PARAMETER num_predict 2048",
    ]
    modelfile = "\n".join(mf_lines) + "\n"
    log.info("registering with Ollama as %s", name)
    proc = subprocess.run(
        ["ollama", "create", name, "-f", "-"],
        input=modelfile, text=True,
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"ollama create failed: {proc.stderr}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--llama-cpp",
                    default=os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp"),
                    help="path to a llama.cpp checkout/build")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--skip-quantize", action="store_true")
    ap.add_argument("--skip-register", action="store_true")
    args = ap.parse_args()

    target = _load_target(args.target)
    out_name = target["output_name"]
    merged_dir = OUT_DIR / out_name
    fp16_gguf = OUT_DIR / f"{out_name}.f16.gguf"
    quant_gguf = OUT_DIR / f"{out_name}.{args.quant.lower()}.gguf"
    system_file = REPO_ROOT / "serve" / "modelfiles" / "system_prompt.txt"

    if not args.skip_merge:
        _merge(target, merged_dir)
    if not args.skip_convert:
        _to_gguf(merged_dir, fp16_gguf, Path(args.llama_cpp))
    if not args.skip_quantize:
        _quantize(fp16_gguf, quant_gguf, Path(args.llama_cpp), args.quant)
    if not args.skip_register:
        _ollama_register(out_name, quant_gguf, system_file)
    log.info("done. ollama tag: %s", out_name)


if __name__ == "__main__":
    main()
