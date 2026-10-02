# flyrank-ai-api

A FastAPI service that classifies free-text customer support messages using an LLM,
returning clean, schema-validated JSON. Built for the FlyRank Backend AI Engineering
"Connect to an AI API" assignment.

**What it does:** `POST /classify` takes a messy support message, asks an LLM for a
judgement, and returns `{"category", "confidence", "summary"}` validated by Pydantic.

## Run it in 5 minutes

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The default provider (`pollinations-get`) needs **no API key** — it calls
`https://text.pollinations.ai/{prompt}?model=openai` with a plain HTTP GET.

## Try it

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "I was charged twice on my last invoice, can you refund one of the charges?"}'
```

Example response:

```json
{
  "category": "billing",
  "confidence": 0.97,
  "summary": "Customer was double-charged on their last invoice and requests a refund for one charge."
}
```

Health check: `curl http://localhost:8000/health`

## LLM provider abstraction

`app/providers.py` defines an `LLMProvider` interface with one method:
`complete(prompt: str) -> str`. Two providers ship with it:

| Provider (`LLM_PROVIDER`) | How it works |
|---|---|
| `pollinations-get` (default) | GET `https://text.pollinations.ai/{urlencoded_prompt}?model=openai`. No key needed. |
| `openai-compatible` | POST `{LLM_BASE_URL}/chat/completions` with `Authorization: Bearer $LLM_API_KEY`. |

Convenience mapping: if `GROQ_API_KEY` is set, `LLM_BASE_URL` defaults to
`https://api.groq.com/openai/v1`; if `OPENROUTER_API_KEY` is set, it defaults to
`https://openrouter.ai/api/v1`. Model is picked via `LLM_MODEL` (provider-specific
default if unset). The POST-based `https://text.pollinations.ai/openai` endpoint is
unreachable from some networks and is **not** used.

Every call uses a real HTTP timeout (`httpx`, `LLM_TIMEOUT_S`, default 30s) and
2 retries with exponential backoff on network errors. If the LLM output fails JSON
parsing or Pydantic validation, the classifier sends **one** repair prompt
containing the validation error and asks for corrected JSON only.

## Kill switch

- Env var `LLM_DISABLE=1` disables all LLM calls at startup.
- Admin endpoint toggles it at runtime:

```bash
curl -X POST http://localhost:8000/admin/kill-switch \
  -H "Content-Type: application/json" -d '{"disabled": true}'
```

While disabled, `POST /classify` returns `503 {"detail": "LLM calls disabled by kill switch"}`.

## Cost log

Every `/classify` request appends one JSONL line to `data/cost_log.jsonl`
(git-ignored) with timestamp, provider, model, estimated input/output tokens and
`cost_usd`, computed from a `PRICE_TABLE` in `app/costlog.py` (Pollinations: $0;
Groq `llama-3.3-70b-versatile` and an OpenRouter Llama model have realistic
per-1M-token prices; unknown models get `cost_usd: 0` with `"price_unknown": true`).

## Eval

`eval/cases.json` holds 10 labeled test cases (message → expected category).
`eval/run_eval.py` classifies each one against the **real** provider and prints
per-case PASS/FAIL plus a final PASS RATE line:

```bash
python eval/run_eval.py
```

**Eval result (measured 2026-10-02, live run against the real Pollinations provider):**
**9/10 (90.0%)**. Per-case output below; the single miss was `general-1`
("Do you offer discounts for students or nonprofits?") classified as `billing`
with confidence 0.95 — the discount/pricing wording pulled it toward billing.
Everything else passed with confidence ≥ 0.92.

```
[PASS] billing-1, billing-2, technical-1, technical-2, account-1, account-2,
       shipping-1, shipping-2, general-2
[FAIL] general-1: expected=general got=billing conf=0.95
PASS RATE: 9/10 (90.0%)
```

Reproduce with:

```bash
python eval/run_eval.py
# prints per-case PASS/FAIL and a final "PASS RATE: x/10 (y%)" line
```

Alternative right now: set `GROQ_API_KEY` (or `OPENROUTER_API_KEY`) — both have
free tiers — and the eval runs unchanged through the `openai-compatible`
provider.

## Notes

- The HTTP client honors `*_proxy` env vars and `SSL_CERT_FILE` explicitly
  (`trust_env=False`): this sandbox's egress proxy does TLS interception, and
  httpx's own env parsing crashes on bracketed IPv6 entries in `NO_PROXY`.
- The free Pollinations endpoint is rate-limited per egress IP; `eval/run_eval.py`
  waits `EVAL_DELAY_S` seconds (default 20) between cases to stay polite.

## Project layout

```
flyrank-ai-api/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI routes: POST /classify, GET /health, POST /admin/kill-switch
│   ├── schemas.py     # strict Pydantic request/response schemas
│   ├── providers.py   # LLMProvider interface + pollinations-get / openai-compatible
│   ├── classifier.py  # prompt building, JSON extraction, one repair retry
│   └── costlog.py     # JSONL cost logging + PRICE_TABLE
├── eval/
│   ├── cases.json
│   └── run_eval.py
├── data/
│   └── .gitkeep       # cost_log.jsonl lives here at runtime (git-ignored)
├── requirements.txt
└── README.md
```
