"""train.templates — chat-template helpers per base.

We intentionally lean on Hugging Face's `tokenizer.apply_chat_template` for
the actual rendering when training: every modern HF tokenizer ships a chat
template, and that's the canonical source of truth.

This module is a thin abstraction over that, plus a fallback for the rare
case where a tokenizer doesn't ship a template.
"""
from __future__ import annotations

from typing import Sequence

# Last-resort fallbacks. These are only used if the tokenizer's own
# `chat_template` is None and the user hasn't supplied a `template.jinja`.
FALLBACK_TEMPLATES: dict[str, str] = {
    "llama3":  "{% for m in messages %}<|start_header_id|>{{m.role}}<|end_header_id|>\n\n{{m.content}}<|eot_id|>{% endfor %}",
    "qwen":    "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}",
    "mistral": "{% for m in messages %}{% if m.role == 'user' %}[INST] {{m.content}} [/INST]{% else %}{{m.content}}{% endif %}{% endfor %}",
    "gemma":   "{% for m in messages %}<start_of_turn>{{m.role}}\n{{m.content}}<end_of_turn>\n{% endfor %}",
}


def render(tokenizer, messages: Sequence[dict], template_name: str | None = None) -> str:
    """Render messages using the tokenizer's chat template (or a fallback)."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        if template_name and template_name in FALLBACK_TEMPLATES:
            from jinja2 import Template
            return Template(FALLBACK_TEMPLATES[template_name]).render(messages=messages)
        raise
