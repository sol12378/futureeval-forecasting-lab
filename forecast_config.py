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

# Ensemble + research defaults (each overridable with a repository variable of the same name).
DEFAULTS = {
    # Who makes each of the N predictions per question (median-aggregated by forecasting-tools).
    # Balanced across vendors: past winners got most from de-correlated ensemble members.
    "FORECASTER_PLAN": "claude,claude,claude,gpt,gpt,gpt",
    "CLAUDE_FORECAST_EFFORT": "xhigh",
    "GPT_FORECAST_MODEL": "openai/gpt-6.2-sol",
    "GPT_FORECAST_EFFORT": "high",
    # Research sources run in parallel; their findings are concatenated for every forecaster.
    # AskNews is opt-in (add "asknews"): its prepaid credit is nearly used up.
    "RESEARCH_SOURCES": "claude_web,gpt_web,tavily,exa,markets,data,resolution_pages",
    # After the parallel sources: one GPT web-search pass that looks for missing / conflicting facts.
    "RESEARCH_GAP_FILL": "true",
    "CLAUDE_RESEARCH_MODEL": "anthropic/claude-opus-5-5",
    "GPT_RESEARCH_MODEL": "openai/gpt-6.2-sol",
    "GPT_RESEARCH_EFFORT": "low",
}
FORECASTERS = ("claude", "gpt")
RESEARCH_SOURCES = ("claude_web", "gpt_web", "asknews", "tavily", "exa", "markets", "data", "resolution_pages")
EFFORTS = ("low", "medium", "high", "xhigh", "max")

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


def setting(env, name: str) -> str:
    return _env(env, name) or DEFAULTS[name]


def forecaster_plan(env: dict | None = None) -> list[str]:
    e = os.environ if env is None else env
    plan = [part.strip() for part in setting(e, "FORECASTER_PLAN").split(",") if part.strip()]
    if not 1 <= len(plan) <= 7 or any(p not in FORECASTERS for p in plan):
        raise ValueError(f"FORECASTER_PLAN must list 1..7 of {FORECASTERS}")
    return plan


def research_sources(env: dict | None = None) -> list[str]:
    e = os.environ if env is None else env
    sources = [part.strip() for part in setting(e, "RESEARCH_SOURCES").split(",") if part.strip()]
    if any(src not in RESEARCH_SOURCES for src in sources):
        raise ValueError(f"RESEARCH_SOURCES must be a subset of {RESEARCH_SOURCES}")
    if "asknews" in sources and not has_asknews(e):
        sources.remove("asknews")
    for name, key in (("tavily", "TAVILY_API_KEY"), ("exa", "EXA_API_KEY")):
        if name in sources and not _env(e, key):
            sources.remove(name)
    return sources


def _effort(env, name: str) -> str:
    value = setting(env, name)
    if value not in EFFORTS:
        raise ValueError(f"{name} must be one of {EFFORTS}")
    return value


def all_models(env: dict | None = None) -> tuple[str, ...]:
    e = os.environ if env is None else env
    claude, gpt = model_names(e)
    return (claude, gpt, setting(e, "GPT_FORECAST_MODEL"), setting(e, "CLAUDE_RESEARCH_MODEL"), setting(e, "GPT_RESEARCH_MODEL"))


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


def gap_fill_enabled(env: dict | None = None) -> bool:
    e = os.environ if env is None else env
    value = setting(e, "RESEARCH_GAP_FILL").strip().lower()
    if value not in ("true", "false"):
        raise ValueError("RESEARCH_GAP_FILL must be true or false")
    return value == "true"


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

    claude_kwargs["reasoning_effort"] = _effort(e, "CLAUDE_FORECAST_EFFORT")  # -> output_config.effort
    gpt_forecast_kwargs = dict(
        model=setting(e, "GPT_FORECAST_MODEL"), temperature=None, timeout=300, allowed_tries=2,
        reasoning_effort=_effort(e, "GPT_FORECAST_EFFORT"),
    )
    claude_research_kwargs = dict(
        model=setting(e, "CLAUDE_RESEARCH_MODEL"), temperature=None, timeout=300, allowed_tries=2, max_tokens=8000,
        reasoning_effort="medium",
        web_search_options={"search_context_size": "medium"},  # Anthropic server-side web search
    )
    gpt_research_kwargs = dict(
        model=setting(e, "GPT_RESEARCH_MODEL"), temperature=None, timeout=300, allowed_tries=2,
        reasoning_effort=_effort(e, "GPT_RESEARCH_EFFORT"), responses_api=True, tools=[{"type": "web_search"}],
    )

    parser = llm("openai", gpt_kwargs)
    summarizer = llm("openai", gpt_kwargs)
    return {
        # temperature=None: current Claude/GPT reasoning models reject sampling params.
        "default": llm("anthropic", claude_kwargs),
        "gpt_forecaster": llm("openai", gpt_forecast_kwargs),
        "claude_researcher": llm("anthropic", claude_research_kwargs),
        "gpt_researcher": llm("openai", gpt_research_kwargs),
        "parser": parser,
        "summarizer": summarizer,
        # Without AskNews this source is skipped; the web-search researchers still run.
        "researcher": "asknews/news-summaries" if has_asknews(e) else "no_research",
    }
