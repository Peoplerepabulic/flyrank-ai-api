"""FastAPI app: POST /classify, GET /health, POST /admin/kill-switch."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import costlog
from .classifier import ClassificationError, classify_message
from .providers import ProviderError, provider_from_env
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    KillSwitchRequest,
)

app = FastAPI(title="flyrank-ai-api", version="0.1.0")

# Runtime kill switch; env var LLM_DISABLE=1 starts it disabled.
_llm_disabled: bool = os.environ.get("LLM_DISABLE", "0") == "1"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    provider = provider_from_env()
    return HealthResponse(
        status="ok",
        provider=provider.name,
        model=provider.model,
        llm_disabled=_llm_disabled,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if _llm_disabled:
        raise HTTPException(status_code=503, detail="LLM calls disabled by kill switch")
    provider = provider_from_env()
    try:
        result, prompt, raw = classify_message(req.message, provider)
    except ProviderError as exc:
        costlog.append_cost_log(
            costlog.cost_entry(
                provider=provider.name,
                model=provider.model,
                input_text=req.message,
                output_text="",
                status="provider_error",
            )
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ClassificationError as exc:
        costlog.append_cost_log(
            costlog.cost_entry(
                provider=provider.name,
                model=provider.model,
                input_text=req.message,
                output_text="",
                status="schema_error",
            )
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    costlog.append_cost_log(
        costlog.cost_entry(
            provider=provider.name,
            model=provider.model,
            input_text=prompt,
            output_text=raw,
        )
    )
    return result


@app.post("/admin/kill-switch")
def kill_switch(req: KillSwitchRequest) -> JSONResponse:
    global _llm_disabled
    _llm_disabled = req.disabled
    return JSONResponse(
        {"llm_disabled": _llm_disabled},
        status_code=200,
    )
