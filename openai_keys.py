"""Use OPENAI_API_KEY first and switch to OPENAI_API_KEY_2 once the first runs out of credit.

Only an explicit "out of credit / quota" error triggers the switch. Rate limits, bad keys
or other errors do not: those should surface as failures instead of silently burning the
second key. The active key is placed in the OPENAI_API_KEY environment variable, which
litellm reads at call time, so key values are never stored on LLM objects or logged.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Mapping

logger = logging.getLogger(__name__)

KEY_ENV_NAMES = ("OPENAI_API_KEY", "OPENAI_API_KEY_2")
_PLACEHOLDERS = {"REPLACE_ME", "your-api-key-here", "1234567890"}
_QUOTA_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing_hard_limit_reached",
    "billing hard limit",
    "credit balance is too low",
)


def is_quota_exhausted(exc: BaseException | None) -> bool:
    """True if the error (or anything it wraps) says the account is out of credit."""
    seen: set[int] = set()
    stack: list[Any] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        text = f"{current!s} {getattr(current, 'code', '')} {getattr(current, 'type', '')}".lower()
        if any(marker in text for marker in _QUOTA_MARKERS):
            return True
        stack.extend([getattr(current, "__cause__", None), getattr(current, "__context__", None)])
        stack.extend(getattr(current, "exceptions", None) or [])  # ExceptionGroup
        last_attempt = getattr(current, "last_attempt", None)  # tenacity.RetryError
        if last_attempt is not None and hasattr(last_attempt, "exception"):
            stack.append(last_attempt.exception())
    return False


class OpenAIKeyPool:
    """Ordered list of OpenAI keys; slot numbers are 1-based and safe to log."""

    def __init__(self, env: Mapping[str, str] | None = None, start_slot: int = 1) -> None:
        e = os.environ if env is None else env
        self._keys = []
        for name in KEY_ENV_NAMES:
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
        """Expose the active key to litellm via the environment."""
        if self._keys:
            os.environ["OPENAI_API_KEY"] = self._keys[self._index]

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
            logger.warning("OpenAI key slot %d is out of credit; switched to slot %d", slot, self.slot)
            return True


def make_failover_llm(pool: OpenAIKeyPool, **llm_kwargs):
    """GeneralLlm whose invoke() retries on the next key when the current one is out of credit."""
    from forecasting_tools import GeneralLlm

    class FailoverOpenAILlm(GeneralLlm):
        async def invoke(self, prompt, system_prompt=None):  # type: ignore[override]
            while True:
                slot = pool.slot
                try:
                    return await super().invoke(prompt, system_prompt)
                except Exception as exc:
                    if not is_quota_exhausted(exc) or not pool.mark_exhausted(slot):
                        raise

    return FailoverOpenAILlm(**llm_kwargs)
