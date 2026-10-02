"""Classification logic: prompt building, JSON extraction, one repair retry."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .providers import LLMProvider, ProviderError
from .schemas import ClassifyResponse

SYSTEM_RULES = """\
You are a customer-support message classifier. Output ONLY a single JSON object —
no markdown, no code fences, no explanation, no extra text.

Schema (exact keys):
{"category": "<one of: billing | technical_issue | account | shipping | general>", "confidence": <float 0..1>, "summary": "<one-sentence summary>"}

Category rules:
- billing: charges, invoices, payments, refunds, pricing, subscriptions fees
- technical_issue: bugs, errors, crashes, broken features, things not working
- account: account settings, profile, password reset, login access, plan changes
- shipping: delivery status, tracking, lost/damaged packages, returns
- general: anything that fits none of the above
"""


def build_prompt(message: str) -> str:
    return (
        SYSTEM_RULES
        + "\nCustomer message:\n\"\"\"\n"
        + message
        + '\n"""\n\nOutput ONLY the JSON object.'
    )


def build_repair_prompt(original_output: str, error: str) -> str:
    return (
        "Your previous output was not valid. Error:\n"
        + error
        + "\n\nPrevious output:\n\"\"\"\n"
        + original_output
        + '\n"""\n\n'
        + "Output ONLY a corrected JSON object matching the schema, nothing else."
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    """Strip code fences; fall back to the outermost { ... } span."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


class ClassificationError(RuntimeError):
    """The LLM could not produce schema-valid output."""


def classify_message(message: str, provider: LLMProvider) -> tuple[ClassifyResponse, str, str]:
    """Classify ``message``. Returns (response, prompt_used, raw_output).

    Raises ProviderError on transport failure, ClassificationError when even the
    single repair retry cannot produce valid JSON.
    """
    prompt = build_prompt(message)
    raw = provider.complete(prompt)
    try:
        return ClassifyResponse.model_validate(json.loads(extract_json(raw))), prompt, raw
    except (json.JSONDecodeError, ValidationError) as first_err:
        repair = build_repair_prompt(raw, str(first_err)[:500])
        raw2 = provider.complete(repair)
        try:
            return (
                ClassifyResponse.model_validate(json.loads(extract_json(raw2))),
                prompt,
                raw2,
            )
        except (json.JSONDecodeError, ValidationError) as err:
            raise ClassificationError(
                f"LLM output failed schema validation after repair retry: {err}"
            ) from err
