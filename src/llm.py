"""
llm.py — the single, swappable LLM adapter.

ALL model-generation calls in the system go through generate() here. This is the
one place that knows about Gemini, so moving to a local model later (Ollama /
Llama 3.1 / Qwen2.5) is a change in this file + config, not a rewrite scattered
across the codebase (CLAUDE.md: swappable model layer).

Free Gemini Flash tier is rate-limited, so we retry with exponential backoff on
429 / transient errors.
"""

from __future__ import annotations

import re
import time

from config import ANTHROPIC_API_KEY, GEMINI_API_KEY, LLM_MODEL, LLM_PROVIDER


def _retry_after(msg: str) -> float | None:
    """Pull the server-suggested retry delay (seconds) out of a 429 message."""
    m = re.search(r"retry in ([\d.]+)", msg, re.IGNORECASE)
    if not m:
        m = re.search(r"retrydelay['\":\s]+(\d+)", msg, re.IGNORECASE)
    return float(m.group(1)) if m else None


class LLMError(RuntimeError):
    """Raised when generation ultimately fails (bad key, exhausted retries)."""


# --- Gemini backend --------------------------------------------------------
_gemini_client = None


def _gemini():
    """Lazily create the Gemini client (validates the key is present)."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set — add it to .env")
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


# --- Anthropic (Claude) backend ---------------------------------------------
_anthropic_client = None


def _anthropic():
    """Lazily create the Anthropic client (validates the key is present)."""
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY is not set — add it to .env")
        import anthropic
        # The SDK retries 429 / 5xx with backoff internally.
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=5)
    return _anthropic_client


# --- Public entry point -----------------------------------------------------
def generate(system: str, prompt: str, *, temperature: float = 0.0,
             max_tokens: int = 2048, max_retries: int = 5) -> str:
    """Generate a completion from the configured provider (the swappable layer).

    system: the standing instructions (grounding rules).
    prompt: the user turn + retrieved context.
    temperature: near 0 for grounded answers (Gemini only; see below).
    max_tokens: output budget — raise it for long, detailed explanations.
    """
    if LLM_PROVIDER == "gemini":
        return _generate_gemini(system, prompt, temperature, max_tokens, max_retries)
    if LLM_PROVIDER == "anthropic":
        return _generate_anthropic(system, prompt, max_tokens)
    raise LLMError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'")


def _generate_gemini(system: str, prompt: str, temperature: float,
                     max_tokens: int, max_retries: int) -> str:
    from google.genai import types
    client = _gemini()
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature,
        max_output_tokens=max_tokens,
        # Gemini 2.5 Flash has "thinking" on by default, and thinking tokens
        # are drawn from the SAME max_output_tokens budget as the visible
        # answer — with it on, a detailed answer's internal reasoning alone
        # consumed ~1,400 of a 3,500-token budget and the reply got cut off
        # mid-sentence (finish_reason=MAX_TOKENS). Grounded extraction/
        # citation doesn't need chain-of-thought, so disable it (budget=0),
        # freeing the full budget for the answer — same rationale as the
        # Anthropic backend's thinking={"type": "disabled"} below.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=LLM_MODEL, contents=prompt, config=cfg
            )
            return (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001 — classify below
            last_err = exc
            msg = str(exc).lower()
            transient = any(s in msg for s in
                            ("429", "rate", "quota", "503", "unavailable", "timeout"))
            if not transient or attempt == max_retries - 1:
                break
            wait = _retry_after(str(exc)) or delay
            time.sleep(min(wait + 1.0, 65.0))
            delay *= 2

    raise LLMError(f"Gemini generation failed: {last_err}")


def _generate_anthropic(system: str, prompt: str, max_tokens: int = 2048) -> str:
    # Note: no temperature is passed. Current Claude models (Opus 4.8, Sonnet 5)
    # reject sampling parameters with a 400; grounding is enforced by the system
    # prompt, not temperature. This keeps the backend model-agnostic.
    import anthropic
    client = _anthropic()
    try:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            # Grounded extraction/citation from provided context does not need
            # extended reasoning. Disabling adaptive thinking (on by default for
            # Sonnet 5 / Opus) keeps answers fast and cheap. Accepted on Sonnet 5,
            # Opus 4.8/4.7, Haiku; ignored gracefully elsewhere.
            thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as exc:
        raise LLMError(f"Anthropic generation failed: {exc}") from exc
    return "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    ).strip()
