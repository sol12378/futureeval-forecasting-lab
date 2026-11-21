"""FutureEval dual-provider configuration (no secrets in source).

Claude = main forecaster/reasoner, GPT = structured-output parser and summarizer.
AskNews (free for registered bot makers) supplies fresh news when configured.
"""
from __future__ import annotations

import os

# Claude Opus 5.5 ($4 / $20 per MTok). Thinking is always on; effort defaults to medium.
DEFAULT_CLAUDE_MODEL = "anthropic/claude-opus-5-5"
# GPT-5.4 mini ($0.75 / $4.50 per MTok) for parsing text into structured predictions.
DEFAULT_GPT_MODEL = "openai/gpt-5.4-mini"

REQUIRED_SECRETS = ("METACULUS_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
_PLACEHOLDERS = {"REPLACE_ME", "your-token-here", "your-api-key-here", "1234567890"}


def _env(e, name: str) -> str:
    return str(e.get(name, "") or "").strip()


def required_credentials_present(env: dict | None = None) -> dict[str, bool]:
    e = os.environ if env is None else env
    return {name: bool(_env(e, name)) and _env(e, name) not in _PLACEHOLDERS for name in REQUIRED_SECRETS}


def has_asknews(env: dict | None = None) -> bool:
    e = os.environ if env is None else env
    return all(_env(e, k) and _env(e, k) not in _PLACEHOLDERS for k in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET"))


def config_counts(env: dict | None = None) -> tuple[int, int]:
    e = os.environ if env is None else env

    def positive(name: str, default: int, maximum: int) -> int:
        raw = _env(e, name) or str(default)
        try:
            n = int(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not 1 <= n <= maximum:
            raise ValueError(f"{name} must be 1..{maximum}")
        return n

    return positive("RESEARCH_REPORTS_PER_QUESTION", 1, 3), positive("PREDICTIONS_PER_RESEARCH_REPORT", 1, 5)


def model_names(env: dict | None = None) -> tuple[str, str]:
    e = os.environ if env is None else env
    return (_env(e, "CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL, _env(e, "GPT_MODEL") or DEFAULT_GPT_MODEL)


def unpriced_models(models: tuple[str, ...]) -> list[str]:
    """Models litellm cannot price. The budget guard would have to guess, so refuse them."""
    import litellm

    missing = []
    for model in models:
        bare = model.split("/", 1)[1] if "/" in model else model
        if model not in litellm.model_cost and bare not in litellm.model_cost:
            missing.append(model)
    return missing


def build_llm_config(env: dict | None = None):
    e = os.environ if env is None else env
    missing = [k for k, ok in required_credentials_present(e).items() if not ok]
    if missing:
        raise RuntimeError("Missing required secrets: " + ", ".join(missing))

    # Import only when the fork's uv environment has been installed.
    from forecasting_tools import GeneralLlm

    claude_model, gpt_model = model_names(e)
    return {
        # temperature=None: current Claude/GPT reasoning models reject sampling params.
        "default": GeneralLlm(model=claude_model, temperature=None, timeout=300, allowed_tries=2, max_tokens=16000),
        "parser": GeneralLlm(model=gpt_model, temperature=None, timeout=120, allowed_tries=2),
        "summarizer": GeneralLlm(model=gpt_model, temperature=None, timeout=120, allowed_tries=2),
        # Without AskNews there is intentionally NO live research (no fake "web search").
        "researcher": "asknews/news-summaries" if has_asknews(e) else "no_research",
    }
