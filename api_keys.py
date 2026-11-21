"""Use each provider's first API key until it runs out of credit, then its second key.

Providers and key order:
  anthropic: ANTHROPIC_API_KEY -> ANTHROPIC_API_KEY_2  (env var names) gitleaks:allow
  openai:    OPENAI_API_KEY    -> OPENAI_API_KEY_2

Only an explicit "out of credit / spend limit reached" error triggers a switch. Rate
limits, invalid keys and other errors do not: those surface as failures instead of
silently burning the second key. The active key is placed in the provider's primary
environment variable, which litellm reads at call time, so key values are never
stored on LLM objects or logged (only the 1-based slot number is).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Mapping

logger = logging.getLogger(__name__)

KEY_ENV_NAMES = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2"),  # env var names  # gitleaks:allow
    "openai": ("OPENAI_API_KEY", "OPENAI_API_KEY_2"),
}
_PLACEHOLDERS = {"REPLACE_ME", "your-api-key-here", "1234567890"}
_EXHAUSTED_MARKERS = {
    "anthropic": (
        "credit balance is too low",
        "billing_error",
        "enforced_spend_limit_reached",
        "reached your specified api usage limits",
    ),
    "openai": (
        "insufficient_quota",
        "exceeded your current quota",
        "billing_hard_limit_reached",
        "billing hard limit",
    ),
}


def is_credit_exhausted(exc: BaseException | None, provider: str) -> bool:
    """True if the error (or anything it wraps) says this provider account is out of credit."""
    markers = _EXHAUSTED_MARKERS[provider]
    seen: set[int] = set()
    stack: list[Any] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        text = f"{current!s} {getattr(current, 'code', '')} {getattr(current, 'type', '')}".lower()
        if any(marker in text for marker in markers):
            return True
        stack.extend([getattr(current, "__cause__", None), getattr(current, "__context__", None)])
        stack.extend(getattr(current, "exceptions", None) or [])  # ExceptionGroup
        last_attempt = getattr(current, "last_attempt", None)  # tenacity.RetryError
        if last_attempt is not None and hasattr(last_attempt, "exception"):
            stack.append(last_attempt.exception())
    return False


class KeyPool:
    """Ordered keys for one provider; slot numbers are 1-based and safe to log."""

    def __init__(self, provider: str, env: Mapping[str, str] | None = None, start_slot: int = 1) -> None:
        if provider not in KEY_ENV_NAMES:
            raise ValueError(f"unknown provider {provider}")
        e = os.environ if env is None else env
        self.provider = provider
        self._env_name = KEY_ENV_NAMES[provider][0]
        self._keys = []
        for name in KEY_ENV_NAMES[provider]:
            value = str(e.get(name, "") or "").strip()
            if value and value not in _PLACEHOLDERS:
                self._keys.append(value)
        self._index = min(max(start_slot, 1), max(len(self._keys), 1)) - 1
        self._lock = threading.Lock()
        self.switched_from: list[int] = []

    @property
    def size(self) -> int:
        return len(self._keys)

    @property
    def slot(self) -> int:
        return self._index + 1

    def activate(self) -> None:
        """Expose the active key to litellm via the provider's primary environment variable."""
        if self._keys:
            os.environ[self._env_name] = self._keys[self._index]

    def mark_exhausted(self, slot: int) -> bool:
        """Advance past `slot`. Returns True if another key is (now) active."""
        with self._lock:
            if slot != self.slot:
                return True  # another concurrent call already switched
            if self._index + 1 >= len(self._keys):
                return False
            self.switched_from.append(slot)
            self._index += 1
            self.activate()
            logger.warning("%s key slot %d is out of credit; switched to slot %d", self.provider, slot, self.slot)
            return True


def make_failover_llm(pool: KeyPool, **llm_kwargs):
    """GeneralLlm whose invoke() retries on the next key when the current one is out of credit."""
    from forecasting_tools import GeneralLlm

    class FailoverLlm(GeneralLlm):
        async def invoke(self, prompt, system_prompt=None):  # type: ignore[override]
            while True:
                slot = pool.slot
                try:
                    return await super().invoke(prompt, system_prompt)
                except Exception as exc:
                    if not is_credit_exhausted(exc, pool.provider) or not pool.mark_exhausted(slot):
                        raise

    return FailoverLlm(**llm_kwargs)
