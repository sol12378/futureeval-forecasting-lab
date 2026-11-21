"""Extra evidence sources for the forecasting bot (all free tiers, no card on file).

- tavily:  news/web search (TAVILY_API_KEY, 1,000 credits/month free)
- exa:     neural web search with page text (EXA_API_KEY, $10/month free credit)
- markets: prediction-market prices from Polymarket and Manifold (no key). Metaculus rules allow
           using publicly available forecasts from other platforms; they are given to the model as a
           reference prior, never copied.
- data:    official time series: FRED (FRED_API_KEY) and CoinGecko (no key), summarised as
           latest value + 1y range, chosen by a cheap LLM extraction step.

Every function returns plain text (possibly empty) and raises on transport errors; the caller
treats any failure as "no evidence from this source". Nothing here logs keys or URLs with keys.
"""
from __future__ import annotations

import json
import logging
import os
import re
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import httpx

# httpx logs every request URL at INFO. FRED only accepts its key as a query parameter and the
# Actions logs of this repository are public, so request logging is silenced for good.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

TIMEOUT = httpx.Timeout(25.0, connect=10.0)


class SourceError(RuntimeError):
    """HTTP failure with the URL (and any key in it) stripped."""


def _check(resp: httpx.Response, name: str) -> None:
    if resp.status_code >= 400:
        raise SourceError(f"{name} HTTP {resp.status_code}")
USER_AGENT = "futureeval-forecasting-lab (open-source Metaculus bot)"


def _key(name: str) -> str:
    value = str(os.environ.get(name, "") or "").strip()
    return "" if value in {"", "REPLACE_ME"} else value


def available(source: str) -> bool:
    return {
        "tavily": bool(_key("TAVILY_API_KEY")),
        "exa": bool(_key("EXA_API_KEY")),
        "markets": True,
        "data": True,  # CoinGecko needs no key; FRED part is skipped without FRED_API_KEY
    }.get(source, False)


def search_query(question_text: str, limit: int = 380) -> str:
    text = re.sub(r"\s+", " ", question_text or "").strip()
    return text[:limit]


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


# --------------------------------------------------------------------------- search
async def tavily(question_text: str) -> str:
    payload = {
        "query": search_query(question_text),
        "topic": "news",
        "days": 30,
        "max_results": 6,
        "search_depth": "basic",  # 1 credit
        "include_answer": False,
    }
    headers = {"Authorization": f"Bearer {_key('TAVILY_API_KEY')}", "User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post("https://api.tavily.com/search", json=payload, headers=headers)
        _check(resp, "Tavily")
        results = resp.json().get("results", [])
    lines = []
    for r in results:
        date = r.get("published_date") or ""
        lines.append(f"- [{date}] {r.get('title', '')} ({r.get('url', '')}): {_clip(r.get('content', ''), 600)}")
    return "\n".join(lines)


async def exa(question_text: str) -> str:
    since = (datetime.now(timezone.utc) - timedelta(days=45)).strftime("%Y-%m-%dT00:00:00.000Z")
    payload = {
        "query": search_query(question_text),
        "numResults": 5,
        "type": "auto",
        "startPublishedDate": since,
        "contents": {"text": {"maxCharacters": 1200}},
    }
    headers = {"x-api-key": _key("EXA_API_KEY"), "User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post("https://api.exa.ai/search", json=payload, headers=headers)
        _check(resp, "Exa")
        results = resp.json().get("results", [])
    lines = []
    for r in results:
        date = (r.get("publishedDate") or "")[:10]
        lines.append(f"- [{date}] {r.get('title', '')} ({r.get('url', '')}): {_clip(r.get('text', ''), 900)}")
    return "\n".join(lines)


# -------------------------------------------------------------------------- markets
def _polymarket_lines(data: dict) -> list[str]:
    lines = []
    for event in (data.get("events") or [])[:4]:
        for market in (event.get("markets") or [])[:3]:
            if market.get("closed"):
                continue
            try:
                outcomes = json.loads(market.get("outcomes") or "[]")
                prices = [float(p) for p in json.loads(market.get("outcomePrices") or "[]")]
            except (ValueError, TypeError):
                continue
            if not outcomes or len(outcomes) != len(prices):
                continue
            quoted = ", ".join(f"{o} {p:.0%}" for o, p in zip(outcomes, prices))
            volume = float(market.get("volume") or 0)
            lines.append(
                f"- Polymarket: \"{market.get('question', event.get('title', ''))}\" -> {quoted}; "
                f"volume ${volume:,.0f}; ends {str(market.get('endDate', ''))[:10]}"
            )
    return lines


def _manifold_lines(data: list) -> list[str]:
    lines = []
    for market in (data or [])[:5]:
        if market.get("isResolved"):
            continue
        prob = market.get("probability")
        if prob is None:
            continue
        close = market.get("closeTime")
        close_s = datetime.fromtimestamp(close / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if close else "?"
        lines.append(
            f"- Manifold (play money): \"{market.get('question', '')}\" -> {float(prob):.0%}; "
            f"traders {market.get('uniqueBettorCount', '?')}; closes {close_s}"
        )
    return lines


async def markets(question_text: str) -> str:
    query = search_query(question_text, 120)
    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        poly = await client.get(
            "https://gamma-api.polymarket.com/public-search",
            params={"q": query, "limit_per_type": 4, "events_status": "active"},
        )
        mani = await client.get(
            "https://api.manifold.markets/v0/search-markets",
            params={"term": query, "limit": 5, "filter": "open", "sort": "score"},
        )
    lines = []
    if poly.status_code == 200:
        lines += _polymarket_lines(poly.json())
    if mani.status_code == 200:
        lines += _manifold_lines(mani.json())
    if not lines:
        return ""
    return (
        "Possibly related prediction markets (search matches; resolution criteria and dates may differ "
        "from this question, so use only as a loose reference prior):\n" + "\n".join(lines)
    )


# ----------------------------------------------------------------------------- data
EXTRACTION_PROMPT = """You map a forecasting question to official data series that would show its current value and recent history.
Reply with ONLY a JSON object, no prose:
{{"fred_search": <short FRED search text for the single most relevant US/international economic series, or null>,
  "coingecko_ids": <list of up to 2 CoinGecko coin ids such as "bitcoin" or "ethereum" if the question is about crypto prices, else []>}}
Use null / [] when no such series is clearly relevant.

Question: {question}
Resolution criteria: {criteria}"""


def parse_extraction(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return {}
    out: dict[str, Any] = {}
    fred = data.get("fred_search")
    if isinstance(fred, str) and 2 <= len(fred) <= 100:
        out["fred_search"] = fred
    coins = data.get("coingecko_ids") or []
    if isinstance(coins, list):
        out["coingecko_ids"] = [c for c in coins if isinstance(c, str) and re.fullmatch(r"[a-z0-9-]{2,40}", c)][:2]
    return out


def fmt(value: float) -> str:
    """Plain notation (the forecast prompts forbid scientific notation)."""
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def summarise_series(values: list[tuple[str, float]], label: str, unit: str = "") -> str:
    """values: (date, value) oldest -> newest."""
    if not values:
        return ""
    last_date, last = values[-1]
    year_ago = [v for d, v in values if d <= (datetime.fromisoformat(last_date[:10]) - timedelta(days=365)).strftime("%Y-%m-%d")]
    window = [v for d, v in values if d > (datetime.fromisoformat(last_date[:10]) - timedelta(days=365)).strftime("%Y-%m-%d")]
    parts = [f"{label}: latest {fmt(last)}{unit} on {last_date[:10]}"]
    if len(values) >= 2:
        parts.append(f"previous {fmt(values[-2][1])} on {values[-2][0][:10]}")
    if year_ago:
        parts.append(f"1y ago {fmt(year_ago[-1])}")
    if len(window) >= 3:
        parts.append(f"12m range {fmt(min(window))}..{fmt(max(window))}")
        changes = [b - a for a, b in zip(window, window[1:])]
        if len(changes) >= 2:
            parts.append(f"typical step change (stdev) {fmt(statistics.pstdev(changes))}")
    return "; ".join(parts)


async def _fred(client: httpx.AsyncClient, search_text: str) -> list[str]:
    key = _key("FRED_API_KEY")
    if not key:
        return []
    base = "https://api.stlouisfed.org/fred"
    resp = await client.get(f"{base}/series/search", params={
        "search_text": search_text, "api_key": key, "file_type": "json", "limit": 2, "order_by": "popularity",
    })
    _check(resp, "FRED search")
    lines = []
    for series in resp.json().get("seriess", [])[:2]:
        obs = await client.get(f"{base}/series/observations", params={
            "series_id": series["id"], "api_key": key, "file_type": "json", "sort_order": "desc", "limit": 500,
        })
        _check(obs, "FRED observations")
        values = []
        for o in reversed(obs.json().get("observations", [])):
            try:
                values.append((o["date"], float(o["value"])))
            except (KeyError, ValueError):
                continue  # "." = missing
        label = f"FRED {series['id']} ({series.get('title', '')}; {series.get('units_short', '')}, {series.get('frequency_short', '')})"
        line = summarise_series(values, label)
        if line:
            lines.append("- " + line)
    return lines


async def _coingecko(client: httpx.AsyncClient, coin: str) -> list[str]:
    resp = await client.get(
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
        params={"vs_currency": "usd", "days": 365, "interval": "daily"},
    )
    if resp.status_code != 200:
        return []
    prices = resp.json().get("prices", [])
    values = [(datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"), float(p)) for ts, p in prices]
    line = summarise_series(values, f"CoinGecko {coin} price (USD)")
    return ["- " + line] if line else []


async def data(question_text: str, resolution_criteria: str, extract: Callable[[str], Awaitable[str]]) -> str:
    plan = parse_extraction(await extract(EXTRACTION_PROMPT.format(
        question=_clip(question_text, 600), criteria=_clip(resolution_criteria or "", 900))))
    lines: list[str] = []
    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        if plan.get("fred_search"):
            lines += await _fred(client, plan["fred_search"])
        for coin in plan.get("coingecko_ids", []):
            lines += await _coingecko(client, coin)
    if not lines:
        return ""
    return ("Official data (automatically retrieved; check that the series matches the resolution source "
            "before relying on it):\n" + "\n".join(lines))
