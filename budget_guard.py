"""Per-provider monthly budget guard and append-only run ledger.

Why this exists
---------------
Anthropic/OpenAI API keys cannot report the remaining credit balance, so the bot
keeps its own ledger. Every run writes one JSON record (never edits old ones) to
``<ledger_dir>/runs/<YYYY-MM>/``. In GitHub Actions that directory is a checkout
of the ``ops-ledger`` branch, so records survive between runs and never conflict.

Rules (all fail closed). Budgets are PER API KEY (each key has its own monthly credit):
- Monthly spend per provider and key slot = sum of this month's run records.
- Active key >= hard ratio of its budget -> switch to the provider's next key proactively;
  when every configured key of a provider is >= hard ratio -> stop starting new LLM work.
- Provider spend >= soft ratio of (per-key budget x number of keys) -> 1 prediction per question.
- A single run may not exceed RUN_COST_CAP_USD in total.
- A corrupt ledger record aborts the run instead of being treated as $0.
- An LLM call whose cost litellm cannot price is charged a conservative amount.

The provider-side limits (Console spend limits, no auto-recharge) remain the
real backstop; this guard only stops the bot from starting new expensive work.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "openai")
PROVIDER_LABEL = {"anthropic": "Claude", "openai": "GPT"}
# Monthly credit of EACH key (the owner has one budget per key, two keys per provider).
DEFAULT_MONTHLY_BUDGET_USD = {"anthropic": 200.0, "openai": 100.0}
BUDGET_ENV = {"anthropic": "CLAUDE_MONTHLY_BUDGET_PER_KEY_USD", "openai": "GPT_MONTHLY_BUDGET_PER_KEY_USD"}
DEFAULT_SOFT_RATIO = 0.70
DEFAULT_HARD_RATIO = 0.90
# The full ensemble costs ~$1/question; MiniBench can open ~10 questions at once.
DEFAULT_RUN_COST_CAP_USD = 30.0
# AskNews prepaid wallet ($15 granted by the owner). Pay-as-you-go: $0.025/credit (latest-news
# search = 1 credit) plus returned tokens at $5-12/M, i.e. roughly $0.03-0.09 per search.
# 150 searches keeps the worst case near $13.5 of the $15.
DEFAULT_ASKNEWS_TOTAL_CALL_CAP = 150
# Charged when litellm cannot price a call (unknown model, missing usage).
UNPRICED_CALL_CHARGE_USD = 0.50


WEB_SEARCH_FEE_USD = 0.01  # Anthropic and OpenAI both list web search at $10 per 1,000 searches


def web_search_fee(response_obj: Any) -> float:
    """Search fees that token pricing may not include (may double count: conservative by design)."""
    try:
        searches = 0
        for item in getattr(response_obj, "output", None) or []:  # OpenAI Responses API
            kind = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if kind == "web_search_call":
                searches += 1
        usage = getattr(response_obj, "usage", None)
        server = getattr(usage, "server_tool_use", None) if usage is not None else None
        if server is not None:  # Anthropic server-side web search
            count = server.get("web_search_requests") if isinstance(server, dict) else getattr(server, "web_search_requests", 0)
            searches += int(count or 0)
        return searches * WEB_SEARCH_FEE_USD
    except Exception:  # noqa: BLE001 - never let accounting break a forecast
        return 0.0


class BudgetExceeded(RuntimeError):
    """Raised before starting LLM work that would break a budget rule."""


class LedgerError(RuntimeError):
    """The ledger could not be read reliably; the run must not proceed."""


def provider_for_model(model: str | None, explicit_provider: str | None = None) -> str | None:
    if explicit_provider in PROVIDERS:
        return explicit_provider
    name = (model or "").lower()
    if name.startswith("anthropic/") or "claude" in name:
        return "anthropic"
    if name.startswith("openai/") or name.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    return None


def _env_float(env: Mapping[str, str], name: str, default: float, low: float, high: float) -> float:
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be within {low}..{high}")
    return value


def _env_key_budgets(env: Mapping[str, str], name: str, default: float) -> tuple[float, ...]:
    """'400,200' -> key 1 $400, key 2 $200; a single value applies to every key."""
    raw = str(env.get(name, "") or "").strip()
    if not raw:
        return tuple(default for _ in KEY_SLOTS)
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in (1, len(KEY_SLOTS)):
        raise ValueError(f"{name} must be one number or {len(KEY_SLOTS)} comma-separated numbers")
    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"{name} must contain numbers") from exc
        if not (math.isfinite(value) and 0.0 <= value <= 10_000.0):
            raise ValueError(f"{name} values must be within 0..10000")
        values.append(value)
    return tuple(values * len(KEY_SLOTS)) if len(values) == 1 else tuple(values)


def current_month(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def previous_month(month: str) -> str:
    year, mon = (int(x) for x in month.split("-"))
    return f"{year - 1}-12" if mon == 1 else f"{year}-{mon - 1:02d}"


MODES = ("tournament", "test_questions", "metaculus_cup")
QUESTION_STATUSES = ("submitted", "failed", "skipped_budget", "dry_run")
MAX_RECORD_COST_USD = 1_000.0
MAX_RECORD_BYTES = 1_000_000
KEY_SLOTS = (1, 2)


def _reject_constant(name: str) -> float:
    raise ValueError(f"non-finite number {name} is not allowed")


def _is_cost(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= MAX_RECORD_COST_USD
    )


def validate_record(data: Any) -> dict:
    """Single schema check shared by the budget guard and the ledger ingest step.

    Anything that could make a monthly sum wrong (NaN/Infinity, negatives, bools,
    strings, unknown providers) is rejected so that a bad record stops the bot
    instead of silently lowering the measured spend.
    """
    if not isinstance(data, dict):
        raise LedgerError("record is not an object")
    costs = data.get("cost_usd")
    if not isinstance(costs, dict) or not costs.keys() <= set(PROVIDERS):
        raise LedgerError("cost_usd must map only known providers")
    if not all(_is_cost(v) for v in costs.values()):
        raise LedgerError("cost_usd values must be finite numbers in 0..1000")
    for counter in ("asknews_calls", "research_failures", "unpriced_calls"):
        value = data.get(counter, 0)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100_000:
            raise LedgerError(f"{counter} invalid")
    if not _is_cost(data.get("unknown_provider_cost_usd", 0.0)):
        raise LedgerError("unknown_provider_cost_usd invalid")
    slots = data.get("key_slot", {})
    if not isinstance(slots, dict) or not slots.keys() <= set(PROVIDERS):
        raise LedgerError("key_slot invalid")
    for slot in slots.values():
        if not isinstance(slot, int) or isinstance(slot, bool) or slot not in KEY_SLOTS:
            raise LedgerError("key_slot value invalid")
    by_slot = data.get("cost_by_key_slot", {})
    if not isinstance(by_slot, dict) or not by_slot.keys() <= set(PROVIDERS):
        raise LedgerError("cost_by_key_slot invalid")
    for costs_for_provider in by_slot.values():
        if not isinstance(costs_for_provider, dict) or not costs_for_provider.keys() <= {str(s) for s in KEY_SLOTS}:
            raise LedgerError("cost_by_key_slot entry invalid")
        if not all(_is_cost(v) for v in costs_for_provider.values()):
            raise LedgerError("cost_by_key_slot values invalid")
    if data.get("mode") not in MODES:
        raise LedgerError("mode invalid")
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        raise LedgerError("questions must be a list")
    for q in questions:
        if not isinstance(q, dict) or not isinstance(q.get("key"), str) or q.get("status") not in QUESTION_STATUSES:
            raise LedgerError("question entry invalid")
    return data


def read_record_file(path: Path) -> dict:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_RECORD_BYTES:
            raise LedgerError(f"Ledger record {path.name} too large")
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except LedgerError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LedgerError(f"Unreadable ledger record {path.name}: {exc}") from exc
    try:
        return validate_record(data)
    except LedgerError as exc:
        raise LedgerError(f"Malformed ledger record {path.name}: {exc}") from exc


def _iter_records(ledger_dir: Path, month: str) -> Iterable[dict]:
    month_dir = ledger_dir / "runs" / month
    if not month_dir.is_dir():
        return []
    return [read_record_file(path) for path in sorted(month_dir.glob("*.json"))]


def load_month_spend(ledger_dir: Path, month: str) -> dict[str, float]:
    spend = {p: 0.0 for p in PROVIDERS}
    for record in _iter_records(ledger_dir, month):
        for provider in PROVIDERS:
            spend[provider] += float(record["cost_usd"].get(provider, 0.0))
    for provider, total in spend.items():
        if not math.isfinite(total):
            raise LedgerError(f"Non-finite monthly spend for {provider}")
    return spend


def load_month_spend_by_slot(ledger_dir: Path, month: str) -> dict[str, dict[str, float]]:
    """Spend per provider and key slot. Records without a per-slot split count against slot 1."""
    spend: dict[str, dict[str, float]] = {p: {} for p in PROVIDERS}
    for record in _iter_records(ledger_dir, month):
        by_slot = record.get("cost_by_key_slot", {})
        for provider in PROVIDERS:
            parts = by_slot.get(provider) or {"1": record["cost_usd"].get(provider, 0.0)}
            for slot, cost in parts.items():
                spend[provider][slot] = spend[provider].get(slot, 0.0) + float(cost)
    for provider, slots in spend.items():
        if not all(math.isfinite(v) for v in slots.values()):
            raise LedgerError(f"Non-finite monthly spend for {provider}")
    return spend


def load_total_asknews_calls(ledger_dir: Path) -> int:
    """AskNews searches across ALL months: the prepaid wallet does not reset monthly."""
    runs = ledger_dir / "runs"
    if not runs.is_dir():
        return 0
    months = sorted(d.name for d in runs.iterdir() if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}", d.name))
    return sum(int(r.get("asknews_calls", 0)) for m in months for r in _iter_records(ledger_dir, m))


def load_submitted_keys(ledger_dir: Path, months: Iterable[str]) -> set[str]:
    """Question keys already submitted to a scored tournament (duplicate guard)."""
    keys: set[str] = set()
    for month in months:
        for record in _iter_records(ledger_dir, month):
            if record.get("mode") != "tournament":
                continue
            for q in record.get("questions", []):
                if q.get("status") == "submitted" and q.get("key"):
                    keys.add(str(q["key"]))
    return keys


def load_last_key_slots(ledger_dir: Path, month: str) -> dict[str, int]:
    """Key slot per provider the previous run of this month ended on (slot 1 at month start)."""
    records = list(_iter_records(ledger_dir, month))  # sorted by timestamped filename
    last = records[-1].get("key_slot", {}) if records else {}
    return {p: int(last.get(p, 1)) for p in PROVIDERS}


def question_key(question: Any) -> str:
    return f"{getattr(question, 'id_of_post', None)}:{getattr(question, 'id_of_question', None)}"


@dataclass
class BudgetGuard:
    ledger_dir: Path
    mode: str
    month: str
    budgets: dict[str, tuple[float, ...]]  # monthly budget of key 1, key 2
    month_spend: dict[str, float]
    month_slot_spend: dict[str, dict[str, float]] = field(default_factory=lambda: {p: {} for p in PROVIDERS})
    soft_ratio: float = DEFAULT_SOFT_RATIO
    hard_ratio: float = DEFAULT_HARD_RATIO
    run_cost_cap: float = DEFAULT_RUN_COST_CAP_USD
    run_spend: dict[str, float] = field(default_factory=lambda: {p: 0.0 for p in PROVIDERS})
    unpriced_calls: int = 0
    research_failures: int = 0
    asknews_calls: int = 0  # this run
    asknews_calls_before: int = 0  # all earlier runs, all months
    asknews_call_cap: int = DEFAULT_ASKNEWS_TOTAL_CALL_CAP
    unknown_provider_cost: float = 0.0
    cost_by_slot: dict[str, dict[str, float]] = field(default_factory=lambda: {p: {} for p in PROVIDERS})
    key_slot: Any = None  # callable(provider) -> active 1-based key slot
    key_counts: dict[str, int] = field(default_factory=lambda: {p: 1 for p in PROVIDERS})
    key_slots: dict[str, tuple[int, ...]] = field(default_factory=lambda: {p: (1,) for p in PROVIDERS})
    key_pools: dict = field(default_factory=dict)
    questions: dict[str, dict] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    _seen_call_ids: set = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_env(cls, mode: str, env: Mapping[str, str] | None = None, now: datetime | None = None) -> "BudgetGuard":
        e = os.environ if env is None else env
        ledger_dir = Path(str(e.get("LEDGER_DIR") or "ops_ledger"))
        month = current_month(now)
        budgets = {p: _env_key_budgets(e, BUDGET_ENV[p], DEFAULT_MONTHLY_BUDGET_USD[p]) for p in PROVIDERS}
        soft = _env_float(e, "BUDGET_SOFT_RATIO", DEFAULT_SOFT_RATIO, 0.05, 1.0)
        hard = _env_float(e, "BUDGET_HARD_RATIO", DEFAULT_HARD_RATIO, 0.05, 1.0)
        if soft > hard:
            raise ValueError("BUDGET_SOFT_RATIO must be <= BUDGET_HARD_RATIO")
        run_cap = _env_float(e, "RUN_COST_CAP_USD", DEFAULT_RUN_COST_CAP_USD, 0.01, 1_000.0)
        require_ledger = str(e.get("REQUIRE_LEDGER", "")).lower() == "true"
        if require_ledger and not ledger_dir.is_dir():
            raise LedgerError(f"Ledger directory {ledger_dir} is missing (REQUIRE_LEDGER=true)")
        spend = load_month_spend(ledger_dir, month)
        slot_spend = load_month_spend_by_slot(ledger_dir, month)
        guard = cls(ledger_dir, mode, month, budgets, spend, slot_spend, soft, hard, run_cap)
        guard.asknews_call_cap = int(_env_float(e, "ASKNEWS_TOTAL_CALL_CAP", DEFAULT_ASKNEWS_TOTAL_CALL_CAP, 0, 100_000))
        guard.asknews_calls_before = load_total_asknews_calls(ledger_dir)
        return guard

    def attach_key_pools(self, pools: dict) -> None:
        """Use api_keys.KeyPool objects: per-key budgets, slot tracking and proactive switching."""
        self.key_pools = dict(pools)
        self.key_slots = {p: (pools[p].slots or (pools[p].slot,)) if p in pools else (1,) for p in PROVIDERS}
        self.key_counts = {p: len(self.key_slots[p]) for p in PROVIDERS}
        self.key_slot = lambda provider: pools[provider].slot if provider in pools else 1

    def install(self) -> None:
        """Register a litellm success callback that books each call's cost."""
        import litellm
        from litellm.integrations.custom_logger import CustomLogger

        guard = self

        class _LedgerCallback(CustomLogger):
            def log_success_event(self, kwargs, response_obj, start_time, end_time):  # noqa: D401
                guard.record_llm_call(kwargs, response_obj)

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                guard.record_llm_call(kwargs, response_obj)

        litellm.callbacks.append(_LedgerCallback())

    # ------------------------------------------------------------- accounting
    def record_llm_call(self, kwargs: dict, response_obj: Any) -> None:
        call_id = kwargs.get("litellm_call_id")
        params = kwargs.get("litellm_params") or {}
        provider = provider_for_model(kwargs.get("model"), params.get("custom_llm_provider"))
        cost = kwargs.get("response_cost")
        if cost is None:
            try:
                import litellm

                cost = litellm.completion_cost(completion_response=response_obj)
            except Exception:  # noqa: BLE001 - unpriced calls are charged conservatively below
                cost = None
        with self._lock:
            if call_id and call_id in self._seen_call_ids:
                return
            if call_id:
                self._seen_call_ids.add(call_id)
            if cost is not None and not _is_cost(cost):
                cost = None  # NaN/negative/absurd values are treated as unpriced
            if cost is not None:
                cost = float(cost) + web_search_fee(response_obj)
            if cost is None:
                self.unpriced_calls += 1
                cost = UNPRICED_CALL_CHARGE_USD
                logger.warning("Could not price LLM call for model=%s; charging $%.2f", kwargs.get("model"), cost)
            if provider is None:
                self.unknown_provider_cost += float(cost)
            else:
                self.run_spend[provider] += float(cost)
                slot = str(self._slot(provider))
                self.cost_by_slot[provider][slot] = self.cost_by_slot[provider].get(slot, 0.0) + float(cost)

    def _slot(self, provider: str) -> int:
        return int(self.key_slot(provider)) if callable(self.key_slot) else 1

    def add_cost(self, provider: str, cost: float) -> None:
        with self._lock:
            self.run_spend[provider] += cost
            slot = str(self._slot(provider))
            self.cost_by_slot[provider][slot] = self.cost_by_slot[provider].get(slot, 0.0) + cost

    def total_spend(self, provider: str) -> float:
        return self.month_spend[provider] + self.run_spend[provider]

    def capacity(self, provider: str) -> float:
        return sum(self.key_budget(provider, s) for s in self.key_slots[provider])

    def key_budget(self, provider: str, slot: int) -> float:
        return self.budgets[provider][slot - 1]

    def slot_spend(self, provider: str, slot: int) -> float:
        key = str(slot)
        return self.month_slot_spend[provider].get(key, 0.0) + self.cost_by_slot[provider].get(key, 0.0)

    def slot_ratio(self, provider: str, slot: int) -> float:
        budget = self.key_budget(provider, slot)
        spent = self.slot_spend(provider, slot)
        if budget <= 0 or not math.isfinite(spent):
            return float("inf")  # fail closed
        return spent / budget

    def ratio(self, provider: str) -> float:
        """Spend on the keys in use against their combined budget (drives cheap mode)."""
        capacity = self.capacity(provider)
        spent = sum(self.slot_spend(provider, s) for s in self.key_slots[provider])
        if capacity <= 0 or not math.isfinite(spent):
            return float("inf")  # fail closed
        return spent / capacity

    def provider_exhausted(self, provider: str) -> bool:
        slots = self.key_slots[provider]
        return all(self.slot_ratio(provider, s) >= self.hard_ratio for s in slots) or not math.isfinite(
            self.total_spend(provider)
        )

    def level(self) -> str:
        if any(self.provider_exhausted(p) for p in PROVIDERS):
            return "hard"
        if max(self.ratio(p) for p in PROVIDERS) >= self.soft_ratio:
            return "soft"
        return "ok"

    def _rotate_off_spent_key(self, provider: str) -> None:
        """Move to the next key before the active one crosses its own hard ratio."""
        pool = self.key_pools.get(provider)
        while pool is not None and self.slot_ratio(provider, pool.slot) >= self.hard_ratio:
            slot = pool.slot
            if not pool.mark_exhausted(slot) or pool.slot == slot:
                break
            logger.warning("%s key %d reached %.0f%% of its budget; using key %d",
                           PROVIDER_LABEL[provider], slot, self.hard_ratio * 100, pool.slot)

    def run_total(self) -> float:
        return sum(self.run_spend.values()) + self.unknown_provider_cost

    def assert_can_spend(self, what: str = "LLM work") -> None:
        """Call before starting research/prediction work for a question."""
        for provider in PROVIDERS:
            self._rotate_off_spent_key(provider)
            if self.provider_exhausted(provider) or self.slot_ratio(provider, self._slot(provider)) >= self.hard_ratio:
                raise BudgetExceeded(
                    f"{PROVIDER_LABEL[provider]} keys reached {self.hard_ratio:.0%} of their "
                    f"per-key budgets { {s: self.key_budget(provider, s) for s in self.key_slots[provider]} } "
                    f"(month ${self.total_spend(provider):.2f}); refusing {what}"
                )
        if self.run_total() >= self.run_cost_cap:
            raise BudgetExceeded(f"Run cost ${self.run_total():.2f} reached cap ${self.run_cost_cap:.2f}; refusing {what}")

    # --------------------------------------------------------------- reporting
    def try_reserve_asknews_call(self) -> bool:
        """Count one AskNews search if the lifetime cap allows it."""
        with self._lock:
            if self.asknews_calls_before + self.asknews_calls >= self.asknews_call_cap:
                return False
            self.asknews_calls += 1
            return True

    def note_research_failure(self) -> None:
        with self._lock:
            self.research_failures += 1

    def note_question(self, key: str, url: str | None, status: str, error: str | None = None) -> None:
        with self._lock:
            entry = {"key": key, "url": url, "status": status}
            if error:
                # Public ledger: keep only a short identifier such as an exception class name.
                head = (error.split(":", 1)[0].split() or [""])[0]
                is_identifier = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,79}", head) is not None
                entry["error"] = head if is_identifier else "Error"
            self.questions[key] = entry

    def summary_lines(self) -> list[str]:
        lines = [f"Budget month {self.month} (soft {self.soft_ratio:.0%}, hard {self.hard_ratio:.0%}, run cap ${self.run_cost_cap:.2f})"]
        for p in PROVIDERS:
            lines.append(
                f"  {PROVIDER_LABEL[p]}: this run ${self.run_spend[p]:.4f}, month ${self.total_spend(p):.2f} / "
                f"${self.capacity(p):.2f} ({self.key_counts[p]} key(s), {self.ratio(p):.1%})"
            )
            per_key = ", ".join(
                f"key{s} ${self.slot_spend(p, s):.2f}/${self.key_budget(p, s):.0f} ({self.slot_ratio(p, s):.0%})"
                for s in self.key_slots[p]
            )
            lines.append(f"    active key {self._slot(p)}; month per key: {per_key}")
        lines.append(
            f"  AskNews searches: this run {self.asknews_calls}, total "
            f"{self.asknews_calls_before + self.asknews_calls} / cap {self.asknews_call_cap}"
        )
        if self.research_failures:
            lines.append(f"  WARNING: research failed {self.research_failures} time(s); forecasted without news")
        if self.unpriced_calls:
            lines.append(f"  WARNING: {self.unpriced_calls} unpriced call(s) charged conservatively")
        if self.unknown_provider_cost:
            lines.append(f"  WARNING: ${self.unknown_provider_cost:.4f} from unknown provider")
        lines.append(f"  Budget level now: {self.level()}")
        return lines

    def write_record(self, extra: dict | None = None, now: datetime | None = None) -> Path:
        finished = now or datetime.now(timezone.utc)
        run_id = os.environ.get("GITHUB_RUN_ID", "local")
        attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
        record = {
            "schema": 1,
            "run_id": run_id,
            "run_attempt": attempt,
            "mode": self.mode,
            "started_at": self.started_at,
            "finished_at": finished.isoformat(),
            "cost_usd": {p: round(self.run_spend[p], 6) for p in PROVIDERS},
            "unknown_provider_cost_usd": round(self.unknown_provider_cost, 6),
            "unpriced_calls": self.unpriced_calls,
            "research_failures": self.research_failures,
            "asknews_calls": self.asknews_calls,
            "key_slot": {p: self._slot(p) for p in PROVIDERS},
            "cost_by_key_slot": {p: {k: round(v, 6) for k, v in self.cost_by_slot[p].items()} for p in PROVIDERS},
            "questions": list(self.questions.values()),
        }
        if extra:
            record.update(extra)
        month_dir = self.ledger_dir / "runs" / self.month
        month_dir.mkdir(parents=True, exist_ok=True)
        stamp = finished.strftime("%Y%m%dT%H%M%SZ")
        path = month_dir / f"{stamp}_{self.mode}_{run_id}-{attempt}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


def _cli(argv: list[str]) -> int:
    """`python3 budget_guard.py validate RECORD.json YYYY-MM` (stdlib only; used by ledger ingest)."""
    if len(argv) != 3 or argv[0] != "validate":
        print("usage: budget_guard.py validate RECORD.json YYYY-MM")
        return 2
    path, month = Path(argv[1]), argv[2]
    now_month = current_month()
    if month not in (now_month, previous_month(now_month)):
        print(f"REJECTED month {month}: only {now_month} or {previous_month(now_month)} may be appended")
        return 1
    try:
        read_record_file(path)
    except LedgerError as exc:
        print(f"REJECTED content: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))
