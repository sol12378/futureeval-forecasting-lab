"""FutureEval dual-provider launch configuration (no secrets in source)."""
from __future__ import annotations

import os

DEFAULT_CLAUDE_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_GPT_MODEL = "openai/gpt-5.4-mini"


def required_credentials_present(env: dict | None = None) -> dict[str, bool]:
    e = os.environ if env is None else env
    placeholders = {"REPLACE_ME", "your-token-here", "your-api-key-here", "1234567890"}
    return {name: bool(str(e.get(name, "")).strip()) and str(e.get(name, "")).strip() not in placeholders for name in (
        "METACULUS_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"
    )}


def config_counts(env: dict | None = None) -> tuple[int, int]:
    e = os.environ if env is None else env
    def positive(name: str, default: int, maximum: int) -> int:
        try:
            n = int(e.get(name, str(default)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not 1 <= n <= maximum:
            raise ValueError(f"{name} must be 1..{maximum}")
        return n
    return positive("RESEARCH_REPORTS_PER_QUESTION", 1, 3), positive("PREDICTIONS_PER_RESEARCH_REPORT", 1, 5)


def build_llm_config(env: dict | None = None):
    """Claude for forecasting, GPT for structured parsing. AskNews optional for real-time evidence."""
    e = os.environ if env is None else env
    missing = [k for k, ok in required_credentials_present(e).items() if not ok]
    if missing:
        raise RuntimeError("Missing required GitHub Secrets: " + ", ".join(missing))

    # Import only when the fork's uv environment has been installed.
    from forecasting_tools import GeneralLlm

    claude_model = str(e.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL)
    gpt_model = str(e.get("GPT_MODEL") or DEFAULT_GPT_MODEL)
    has_asknews = bool(e.get("ASKNEWS_CLIENT_ID")) and bool(e.get("ASKNEWS_SECRET"))
    return {
        "default": GeneralLlm(model=claude_model, timeout=120, allowed_tries=2),
        "parser": gpt_model,
        "summarizer": gpt_model,
        # IMPORTANT: without AskNews this is intentionally no live research.
        # Do not hallucinate a 'web search' from a plain LLM API call.
        "researcher": "asknews/news-summaries" if has_asknews else "no_research",
    }
