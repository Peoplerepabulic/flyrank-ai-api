"""LLM provider abstraction.

An LLMProvider has one job: turn a prompt string into a completion string.
Transport details, timeouts, and retries live here; judgement logic lives in
classifier.py so providers stay swappable.
"""

from __future__ import annotations

import os
import time
import urllib.parse
from abc import ABC, abstractmethod

import httpx


class ProviderError(RuntimeError):
    """Raised when the provider cannot produce a completion."""


class RetriableError(ProviderError):
    """A provider-side failure worth retrying (network error or HTTP 5xx/429)."""


def _http_client(timeout_s: float) -> httpx.Client:
    """Build an httpx client that honors *_proxy env vars.

    We set trust_env=False and pass proxies explicitly because httpx's
    trust_env parser crashes on bracketed IPv6 entries (e.g. "[::1]") in
    the NO_PROXY variable found on some hosts.
    """
    proxy = (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("all_proxy")
        or os.environ.get("ALL_PROXY")
    )
    # httpx>=0.28 uses `proxy=` (single URL string); older versions use `proxies=`.
    import inspect

    kwargs = {"trust_env": False, "timeout": timeout_s}
    # Honor the host CA bundle (egress proxies with TLS interception ship one
    # via SSL_CERT_FILE); fall back to httpx/certifi defaults otherwise.
    kwargs["verify"] = os.environ.get("SSL_CERT_FILE") or True
    if "proxy" in inspect.signature(httpx.Client).parameters:
        kwargs["proxy"] = proxy
    else:  # pragma: no cover - legacy httpx
        kwargs["proxies"] = {"http://": proxy, "https://": proxy} if proxy else None
    return httpx.Client(**kwargs)


class LLMProvider(ABC):
    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's raw text completion for ``prompt``."""
        raise NotImplementedError


def _with_retries(fn, *, attempts: int = 3, base_delay: float = 1.0):
    """Run ``fn`` with exponential backoff on retriable errors.

    Retries network-level errors plus HTTP 5xx / 429 from the provider
    (the free pollinations endpoint is intermittently flaky).
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except (httpx.TransportError, httpx.TimeoutException, RetriableError) as exc:
            last = exc
            if i < attempts - 1:
                time.sleep(base_delay * 2**i)
    assert last is not None
    raise ProviderError(f"failed after {attempts} attempts: {last}") from last


def _check_status(resp: httpx.Response, where: str) -> None:
    """Raise RetriableError for 5xx/429, ProviderError for other bad statuses."""
    if resp.status_code == 200:
        return
    body = resp.text[:200]
    if resp.status_code == 429 or 500 <= resp.status_code < 600:
        raise RetriableError(f"{where} returned HTTP {resp.status_code}: {body}")
    raise ProviderError(f"{where} returned HTTP {resp.status_code}: {body}")


def _timeout() -> float:
    return float(os.environ.get("LLM_TIMEOUT_S", "30"))


class PollinationsGetProvider(LLMProvider):
    """Plain GET to https://text.pollinations.ai/{prompt}?model=openai.

    Needs no API key. Note: the POST variant
    (https://text.pollinations.ai/openai) was found unreachable from some
    networks, so the GET form is the default path.
    """

    name = "pollinations-get"
    model = "openai"  # pollinations proxy model alias

    def __init__(self, model: str | None = None, timeout_s: float | None = None):
        if model:
            self.model = model
        self.timeout_s = timeout_s or _timeout()

    def complete(self, prompt: str) -> str:
        url = (
            "https://text.pollinations.ai/"
            f"{urllib.parse.quote(prompt, safe='')}"
            f"?model={urllib.parse.quote(self.model, safe='')}"
        )

        def _do() -> str:
            with _http_client(self.timeout_s) as client:
                resp = client.get(url)
            _check_status(resp, "pollinations GET")
            return resp.text

        return _with_retries(_do)


class OpenAICompatibleProvider(LLMProvider):
    """POST {base_url}/chat/completions, OpenAI-style chat messages."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s or _timeout()

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter asks for these; harmless elsewhere.
            "HTTP-Referer": "https://github.com/flyrank-ai-api",
            "X-Title": "flyrank-ai-api",
        }
        url = f"{self.base_url}/chat/completions"

        def _do() -> str:
            with _http_client(self.timeout_s) as client:
                resp = client.post(url, json=payload, headers=headers)
            _check_status(resp, "chat completions")
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError(f"unexpected chat response shape: {data!r}"[:200]) from exc

        return _with_retries(_do)


def _resolve_base_url() -> tuple[str, str]:
    """Return (base_url, default_model) using convenience env vars."""
    if os.environ.get("LLM_BASE_URL"):
        return os.environ["LLM_BASE_URL"], os.environ.get("LLM_MODEL", "")
    if os.environ.get("GROQ_API_KEY"):
        return "https://api.groq.com/openai/v1", os.environ.get(
            "LLM_MODEL", "llama-3.3-70b-versatile"
        )
    if os.environ.get("OPENROUTER_API_KEY"):
        return "https://openrouter.ai/api/v1", os.environ.get(
            "LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"
        )
    return "", os.environ.get("LLM_MODEL", "")


def _resolve_api_key() -> str:
    for var in ("LLM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    return ""


def provider_from_env() -> LLMProvider:
    """Build the configured provider. LLM_PROVIDER selects it (default pollinations-get)."""
    which = os.environ.get("LLM_PROVIDER", "pollinations-get").strip().lower()
    if which in ("pollinations-get", "pollinations"):
        return PollinationsGetProvider(model=os.environ.get("LLM_MODEL") or None)
    if which in ("openai-compatible", "openai"):
        base_url, default_model = _resolve_base_url()
        api_key = _resolve_api_key()
        if not base_url:
            raise ProviderError(
                "openai-compatible provider needs LLM_BASE_URL (or GROQ_API_KEY / "
                "OPENROUTER_API_KEY convenience vars)"
            )
        if not api_key:
            raise ProviderError(
                "openai-compatible provider needs LLM_API_KEY (or GROQ_API_KEY / "
                "OPENROUTER_API_KEY)"
            )
        model = os.environ.get("LLM_MODEL") or default_model
        if not model:
            raise ProviderError("openai-compatible provider needs LLM_MODEL")
        return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model)
    raise ProviderError(f"unknown LLM_PROVIDER={which!r}")
