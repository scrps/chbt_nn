"""serve/scan.py — BYO model scanner (PLAN.md §4c).

Walks `serve/models/` looking for GGUF files (or directories containing one)
and registers them with Ollama as `byo-<dirname>`. Optional `model.toml` and
`template.jinja` sidecars next to the weights are honored.

Run on demand:
    python -m serve.scan

Or let `infra/serve.sh` invoke it before launching the picker.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "serve" / "models"


def _ollama(args: list[str], stdin: str | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["ollama", *args],
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _existing_models() -> set[str]:
    rc, out, _ = _ollama(["list"])
    if rc != 0:
        return set()
    names: set[str] = set()
    for line in out.splitlines()[1:]:  # skip header
        parts = line.split()
        if parts:
            names.add(parts[0])
    return names


def _find_gguf(d: Path) -> Path | None:
    if d.is_file() and d.suffix == ".gguf":
        return d
    if d.is_dir():
        for p in sorted(d.glob("*.gguf")):
            return p
    return None


def _read_sidecar(d: Path) -> dict:
    cfg_path = d / "model.toml" if d.is_dir() else d.with_suffix(".toml")
    if cfg_path.exists():
        with cfg_path.open("rb") as fh:
            return tomllib.load(fh)
    return {}


def _build_modelfile(gguf: Path, sidecar: dict, template_path: Path | None) -> str:
    lines = [f"FROM {gguf}"]
    params = sidecar.get("parameters") or {}
    for k, v in params.items():
        if isinstance(v, str):
            lines.append(f"PARAMETER {k} {shlex.quote(v)}")
        else:
            lines.append(f"PARAMETER {k} {v}")
    if template_path and template_path.exists():
        tmpl = template_path.read_text(encoding="utf-8")
        lines.append('TEMPLATE """\n' + tmpl + '\n"""')
    if system := sidecar.get("system"):
        lines.append('SYSTEM """\n' + system + '\n"""')
    return "\n".join(lines) + "\n"


def scan(verbose: bool = True) -> list[str]:
    if not MODELS_DIR.exists():
        return []

    existing = _existing_models()
    registered: list[str] = []

    for entry in sorted(MODELS_DIR.iterdir()):
        if entry.name.startswith(".") or entry.name.lower() == "readme.md":
            continue
        gguf = _find_gguf(entry)
        if gguf is None:
            if verbose:
                print(f"[scan] skip {entry.name}: no .gguf found", file=sys.stderr)
            continue

        name = f"byo-{entry.stem.lower().replace(' ', '-')}"
        if any(name == e or e.startswith(name + ":") for e in existing):
            if verbose:
                print(f"[scan] {name} already registered")
            continue

        sidecar = _read_sidecar(entry)
        tmpl_path = (entry / "template.jinja") if entry.is_dir() else None
        modelfile = _build_modelfile(gguf, sidecar, tmpl_path)

        if verbose:
            print(f"[scan] registering {name} from {gguf}")
        # `ollama create` reads the Modelfile from a path; pass via -f /dev/stdin
        # for cleanliness.
        proc = subprocess.run(
            ["ollama", "create", name, "-f", "-"],
            input=modelfile,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            print(f"[scan] failed to register {name}: {proc.stderr}", file=sys.stderr)
            continue
        registered.append(name)

    return registered


if __name__ == "__main__":
    out = scan()
    if out:
        print(f"[scan] registered: {', '.join(out)}")
    else:
        print("[scan] nothing new to register")
