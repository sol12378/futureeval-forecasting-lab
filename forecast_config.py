"""FutureEval dual-provider configuration (no secrets in source).

Claude = main forecaster/reasoner, GPT = structured-output parser and summarizer.
AskNews (free for registered bot makers) supplies fresh news when configured.
"""
from __future__ import annotations

import os

# Claude Opus 5.5 ($4 / $20 per MTok). Thinking is always on; effort defaults to medium.
DEFAULT_CLAUDE_MODEL = "anthropic/claude-opus-5-5"
# GPT-6.2 Sol ($2 / $10 per MTok): OpenAI's current balanced flagship, used at low
# reasoning effort to turn Claude's free-text forecast into validated structured output.
DEFAULT_GPT_MODEL = "openai/gpt-6.2-sol"
DEFAULT_GPT_REASONING_EFFORT = "low"
GPT_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")

REQUIRED_SECRETS = ("METACULUS_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
_PLACEHOLDERS = {"REPLACE_ME", "your-token-here", "your-api-key-here", "1234567890"}


def _env(e, name: str) -> str:
    return str(e.get(name, "") or "").strip()


def required_credentials_present(env: dict | None = None) -> dict[str, bool]:
    e = os.environ if env is None else env
    return {name: bool(_env(e, name)) and _env(e, name) not in _PLACEHOLDERS for name in REQUIRED_SECRETS}


def has_asknews(env: dict | None = None) -> bool:
    """AskNews is usable with ASKNEWS_API_KEY (official Metaculus setup) or CLIENT_ID + SECRET (OAuth)."""
    e = os.environ if env is None else env

    def real(k: str) -> bool:
        return bool(_env(e, k)) and _env(e, k) not in _PLACEHOLDERS

    return real("ASKNEWS_API_KEY") or (real("ASKNEWS_CLIENT_ID") and real("ASKNEWS_SECRET"))


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


def gpt_reasoning_effort(env: dict | None = None) -> str:
    e = os.environ if env is None else env
    effort = _env(e, "GPT_REASONING_EFFORT") or DEFAULT_GPT_REASONING_EFFORT
    if effort not in GPT_REASONING_EFFORTS:
        raise ValueError(f"GPT_REASONING_EFFORT must be one of {GPT_REASONING_EFFORTS}")
    return effort


def unpriced_models(models: tuple[str, ...]) -> list[str]:
    """Models litellm cannot price. The budget guard would have to guess, so refuse them."""
    import litellm

    missing = []
    for model in models:
        bare = model.split("/", 1)[1] if "/" in model else model
        if model not in litellm.model_cost and bare not in litellm.model_cost:
            missing.append(model)
    return missing


def build_llm_config(env: dict | None = None, key_pools: dict | None = None):
    e = os.environ if env is None else env
    missing = [k for k, ok in required_credentials_present(e).items() if not ok]
    if missing:
        raise RuntimeError("Missing required secrets: " + ", ".join(missing))

    # Import only when the fork's uv environment has been installed.
    from forecasting_tools import GeneralLlm

    claude_model, gpt_model = model_names(e)
    gpt_kwargs = dict(
        model=gpt_model,
        temperature=None,
        timeout=120,
        allowed_tries=2,
        reasoning_effort=gpt_reasoning_effort(e),
    )
    claude_kwargs = dict(model=claude_model, temperature=None, timeout=300, allowed_tries=2, max_tokens=16000)
    pools = key_pools or {}

    def llm(provider: str, kwargs: dict):
        # With a key pool, the LLM moves to the provider's 2nd key once the 1st is out of credit.
        if provider in pools:
            from api_keys import make_failover_llm

            return make_failover_llm(pools[provider], **kwargs)
        return GeneralLlm(**kwargs)

    parser = llm("openai", gpt_kwargs)
    summarizer = llm("openai", gpt_kwargs)
    return {
        # temperature=None: current Claude/GPT reasoning models reject sampling params.
        "default": llm("anthropic", claude_kwargs),
        "parser": parser,
        "summarizer": summarizer,
        # Without AskNews there is intentionally NO live research (no fake "web search").
        "researcher": "asknews/news-summaries" if has_asknews(e) else "no_research",
    }
