"""JSONL cost logging for /classify requests.

One line per request: timestamp, provider, model, estimated tokens, cost_usd.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

# (input_price_per_1M_tokens_usd, output_price_per_1M_tokens_usd)
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # Pollinations free tier proxy — no cost. The default model alias is "openai".
    "pollinations": (0.0, 0.0),
    "openai": (0.0, 0.0),  # pollinations GET proxy alias (pollinations-get default)
    # Groq hosted Llama 3.3 70B (public pricing, Oct 2026).
    "llama-3.3-70b-versatile": (0.59, 0.79),
    # OpenRouter Llama 3.3 70B Instruct (public pricing, Oct 2026).
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.30),
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token) when no tokenizer is available."""
    return max(1, len(text) // 4)


def price_for_model(model: str) -> tuple[float, float, bool]:
    """Return (input_price, output_price, known) for a model name.

    Unknown models cost $0 with known=False so the log flags them.
    """
    lowered = model.lower()
    for key, (inp, outp) in PRICE_TABLE.items():
        if key in lowered:
            return inp, outp, True
    return 0.0, 0.0, False


def cost_entry(
    *,
    provider: str,
    model: str,
    input_text: str,
    output_text: str,
    status: str = "ok",
) -> dict:
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    inp_price, outp_price, known = price_for_model(model)
    cost_usd = round(input_tokens / 1e6 * inp_price + output_tokens / 1e6 * outp_price, 8)
    entry: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "status": status,
    }
    if not known:
        entry["price_unknown"] = True
    return entry


def append_cost_log(entry: dict, data_dir: str | None = None) -> str:
    data_dir = data_dir or os.environ.get("COST_LOG_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "cost_log.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return path
