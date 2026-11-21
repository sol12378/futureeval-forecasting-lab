import json
import os
import tempfile
from unittest.mock import patch
import unittest
from datetime import datetime, timezone
from pathlib import Path

from budget_guard import (
    BudgetExceeded,
    BudgetGuard,
    LedgerError,
    load_month_spend,
    load_submitted_keys,
    previous_month,
    provider_for_model,
)

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def write(ledger: Path, month: str, name: str, record: dict) -> None:
    d = ledger / "runs" / month
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({"mode": "tournament", **record}))


class BudgetGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.tmp.name)
        self.env = {"LEDGER_DIR": str(self.ledger)}

    def tearDown(self):
        self.tmp.cleanup()

    def guard(self, **env):
        return BudgetGuard.from_env("tournament", {**self.env, **env}, now=NOW)

    def test_provider_detection(self):
        self.assertEqual(provider_for_model("anthropic/claude-opus-5-5"), "anthropic")
        self.assertEqual(provider_for_model("claude-opus-5-5"), "anthropic")
        self.assertEqual(provider_for_model("gpt-5.4-mini"), "openai")
        self.assertEqual(provider_for_model("whatever", "openai"), "openai")
        self.assertIsNone(provider_for_model("asknews/news-summaries"))

    def test_empty_ledger_is_ok(self):
        g = self.guard()
        self.assertEqual(g.level(), "ok")
        g.assert_can_spend()

    def test_month_sum_and_levels(self):
        write(self.ledger, "2026-11", "a.json", {"cost_usd": {"anthropic": 100.0, "openai": 5.0}})
        write(self.ledger, "2026-11", "b.json", {"cost_usd": {"anthropic": 45.0, "openai": 0.0}})
        write(self.ledger, "2026-10", "old.json", {"cost_usd": {"anthropic": 999.0, "openai": 999.0}})
        self.assertEqual(load_month_spend(self.ledger, "2026-11"), {"anthropic": 145.0, "openai": 5.0})
        g = self.guard()
        self.assertEqual(g.level(), "soft")  # 145/200 = 72.5%
        g.add_cost("anthropic", 40.0)  # 185/200 = 92.5%
        self.assertEqual(g.level(), "hard")
        with self.assertRaises(BudgetExceeded):
            g.assert_can_spend()

    def test_openai_hard_stop_blocks_everything(self):
        write(self.ledger, "2026-11", "a.json", {"cost_usd": {"anthropic": 0.0, "openai": 95.0}})
        with self.assertRaises(BudgetExceeded):
            self.guard().assert_can_spend()

    def test_run_cap(self):
        g = self.guard(RUN_COST_CAP_USD="1")
        g.add_cost("openai", 1.0)
        with self.assertRaises(BudgetExceeded):
            g.assert_can_spend()

    def test_corrupt_record_fails_closed(self):
        d = self.ledger / "runs" / "2026-11"
        d.mkdir(parents=True)
        (d / "bad.json").write_text("{not json")
        with self.assertRaises(LedgerError):
            self.guard()
        for bad in (
            '{"mode": "tournament", "cost_usd": {"anthropic": -5}}',
            '{"mode": "tournament", "cost_usd": {"anthropic": NaN}}',
            '{"mode": "tournament", "cost_usd": {"openai": Infinity}}',
            '{"mode": "tournament", "cost_usd": {"anthropic": "1"}}',
            '{"mode": "tournament", "cost_usd": {"anthropic": true}}',
            '{"mode": "tournament", "cost_usd": {"gemini": 1}}',
            '{"mode": "hacked", "cost_usd": {"anthropic": 1}}',
            '{"mode": "tournament", "cost_usd": {}, "questions": [{"key": "1:2", "status": "weird"}]}',
        ):
            (d / "bad.json").write_text(bad)
            with self.assertRaises(LedgerError, msg=bad):
                self.guard()

    def test_require_ledger(self):
        with self.assertRaises(LedgerError):
            BudgetGuard.from_env("tournament", {"LEDGER_DIR": str(self.ledger / "missing"), "REQUIRE_LEDGER": "true"}, now=NOW)

    def test_invalid_env(self):
        with self.assertRaises(ValueError):
            self.guard(BUDGET_SOFT_RATIO="0.95", BUDGET_HARD_RATIO="0.9")
        with self.assertRaises(ValueError):
            self.guard(CLAUDE_MONTHLY_BUDGET_PER_KEY_USD="abc")

    def test_record_llm_call_prices_and_dedupes(self):
        g = self.guard()
        kwargs = {"model": "claude-opus-5-5", "litellm_call_id": "c1", "response_cost": 0.12,
                  "litellm_params": {"custom_llm_provider": "anthropic"}}
        g.record_llm_call(kwargs, None)
        g.record_llm_call(kwargs, None)  # same call reported twice (sync+async) -> counted once
        self.assertAlmostEqual(g.run_spend["anthropic"], 0.12)
        g.record_llm_call({"model": "gpt-5.4-mini", "litellm_call_id": "c2", "response_cost": None}, object())
        self.assertEqual(g.unpriced_calls, 1)
        self.assertAlmostEqual(g.run_spend["openai"], 0.50)  # conservative charge

    def test_write_record_and_duplicate_keys(self):
        g = self.guard()
        g.add_cost("anthropic", 0.3)
        g.note_question("101:201", "https://x/101", "submitted")
        g.note_question("102:202", "https://x/102", "failed", "boom")
        path = g.write_record({"notes": []}, now=NOW)
        data = json.loads(path.read_text())
        self.assertEqual(data["cost_usd"]["anthropic"], 0.3)
        self.assertEqual(load_month_spend(self.ledger, "2026-11")["anthropic"], 0.3)
        self.assertEqual(load_submitted_keys(self.ledger, ["2026-11"]), {"101:201"})
        # test-mode submissions never block tournament questions
        t = BudgetGuard.from_env("test_questions", self.env, now=NOW)
        t.note_question("103:203", None, "submitted")
        t.write_record(now=NOW.replace(minute=5))
        self.assertEqual(load_submitted_keys(self.ledger, ["2026-11"]), {"101:201"})

    def test_public_ledger_error_is_identifier_only(self):
        g = self.guard()
        g.note_question("1:2", None, "failed", "AuthenticationError: invalid x-api-key sk-ant-abc123 for request")
        self.assertEqual(g.questions["1:2"]["error"], "AuthenticationError")
        g.note_question("3:4", None, "failed", "sk-ant-abc123 leaked")
        self.assertEqual(g.questions["3:4"]["error"], "Error")

    def test_nan_spend_fails_closed(self):
        g = self.guard()
        g.run_spend["anthropic"] = float("nan")
        self.assertEqual(g.level(), "hard")
        g.run_spend["anthropic"] = 0.0
        g.record_llm_call({"model": "claude-opus-5-5", "litellm_call_id": "n", "response_cost": float("nan")}, None)
        self.assertEqual(g.unpriced_calls, 1)  # NaN cost charged conservatively, never as NaN
        self.assertAlmostEqual(g.run_spend["anthropic"], 0.50)

    def test_cli_validate(self):
        from budget_guard import _cli, current_month

        good = self.ledger / "good.json"
        good.write_text('{"mode": "test_questions", "cost_usd": {"anthropic": 0.2}}')
        self.assertEqual(_cli(["validate", str(good), current_month()]), 0)
        self.assertEqual(_cli(["validate", str(good), "2001-01"]), 1)
        good.write_text('{"mode": "test_questions", "cost_usd": {"anthropic": NaN}}')
        self.assertEqual(_cli(["validate", str(good), current_month()]), 1)

    def test_key_slot_tracking_and_resume(self):
        from budget_guard import load_last_key_slots

        g = self.guard()
        slots = {"anthropic": 1, "openai": 1}
        g.key_slot = lambda provider: slots[provider]
        g.record_llm_call({"model": "gpt-6.2-sol", "litellm_call_id": "a", "response_cost": 0.02}, None)
        slots["openai"] = 2
        g.record_llm_call({"model": "gpt-6.2-sol", "litellm_call_id": "b", "response_cost": 0.03}, None)
        g.record_llm_call({"model": "claude-opus-5-5", "litellm_call_id": "c", "response_cost": 0.10}, None)
        self.assertEqual(g.cost_by_slot, {"anthropic": {"1": 0.10}, "openai": {"1": 0.02, "2": 0.03}})
        g.write_record(now=NOW)
        self.assertEqual(load_last_key_slots(self.ledger, "2026-11"), {"anthropic": 1, "openai": 2})
        self.assertEqual(load_last_key_slots(self.ledger, "2026-12"), {"anthropic": 1, "openai": 1})
        for bad in ({"key_slot": {"openai": 3}}, {"key_slot": {"gemini": 1}},
                    {"cost_by_key_slot": {"openai": {"9": 1.0}}}, {"cost_by_key_slot": {"anthropic": {"1": float("nan")}}}):
            write(self.ledger, "2026-11", "zz.json", {"cost_usd": {}, **bad})
            with self.assertRaises(LedgerError, msg=str(bad)):
                load_last_key_slots(self.ledger, "2026-11")

    def test_per_key_budget_rotates_then_stops(self):
        from api_keys import KeyPool

        # key 1 of Claude already spent $185 of its $200 this month (92.5% >= 90%)
        write(self.ledger, "2026-11", "a.json",
              {"cost_usd": {"anthropic": 185.0}, "cost_by_key_slot": {"anthropic": {"1": 185.0}}})
        g = self.guard()
        self.assertEqual(g.level(), "hard")  # single key known so far -> exhausted
        pools = {"anthropic": KeyPool("anthropic", {"ANTHROPIC_API_KEY": "a1", "ANTHROPIC_API_KEY_2": "a2"}),
                 "openai": KeyPool("openai", {"OPENAI_API_KEY": "o1"})}
        with patch.dict(os.environ, {}):
            g.attach_key_pools(pools)
            self.assertEqual(g.capacity("anthropic"), 400.0)
            self.assertEqual(g.level(), "ok")  # combined 185/400 = 46%: key 2 still fresh
            g.assert_can_spend()  # rotates Claude to key 2 instead of stopping
            self.assertEqual(pools["anthropic"].slot, 2)
            self.assertEqual(os.environ["ANTHROPIC_API_KEY"], "a2")
            g.add_cost("anthropic", 181.0)  # key 2 now at 90.5%
            self.assertTrue(g.provider_exhausted("anthropic"))
            with self.assertRaises(BudgetExceeded):
                g.assert_can_spend()
            self.assertEqual(g.cost_by_slot["anthropic"], {"2": 181.0})

    def test_legacy_records_count_against_key1(self):
        from budget_guard import load_month_spend_by_slot

        write(self.ledger, "2026-11", "a.json", {"cost_usd": {"openai": 3.0}})
        write(self.ledger, "2026-11", "b.json", {"cost_usd": {"openai": 2.0}, "cost_by_key_slot": {"openai": {"2": 2.0}}})
        self.assertEqual(load_month_spend_by_slot(self.ledger, "2026-11")["openai"], {"1": 3.0, "2": 2.0})

    def test_asknews_lifetime_cap_spans_months(self):
        write(self.ledger, "2026-10", "a.json", {"cost_usd": {}, "asknews_calls": 2})
        write(self.ledger, "2026-11", "b.json", {"cost_usd": {}, "asknews_calls": 1})
        g = self.guard(ASKNEWS_TOTAL_CALL_CAP="4")
        self.assertEqual(g.asknews_calls_before, 3)
        self.assertTrue(g.try_reserve_asknews_call())
        self.assertFalse(g.try_reserve_asknews_call())
        self.assertEqual(g.asknews_calls, 1)
        write(self.ledger, "2026-11", "c.json", {"cost_usd": {}, "asknews_calls": -1})
        with self.assertRaises(LedgerError):
            self.guard()

    def test_unequal_key_budgets(self):
        from api_keys import KeyPool

        g = self.guard(CLAUDE_MONTHLY_BUDGET_PER_KEY_USD="400,200", GPT_MONTHLY_BUDGET_PER_KEY_USD="180, 100")
        self.assertEqual(g.budgets, {"anthropic": (400.0, 200.0), "openai": (180.0, 100.0)})
        pools = {"anthropic": KeyPool("anthropic", {"ANTHROPIC_API_KEY": "a1", "ANTHROPIC_API_KEY_2": "a2"}),
                 "openai": KeyPool("openai", {"OPENAI_API_KEY": "o1", "OPENAI_API_KEY_2": "o2"})}
        with patch.dict(os.environ, {}):
            g.attach_key_pools(pools)
            self.assertEqual((g.capacity("anthropic"), g.capacity("openai")), (600.0, 280.0))
            g.add_cost("anthropic", 361.0)  # key 1 at 90.25% of $400 -> rotate
            g.assert_can_spend()
            self.assertEqual(pools["anthropic"].slot, 2)
        for bad in ("1,2,3", "abc", "-5", "nan"):
            with self.assertRaises(ValueError, msg=bad):
                self.guard(GPT_MONTHLY_BUDGET_PER_KEY_USD=bad)

    def test_previous_month(self):
        self.assertEqual(previous_month("2026-01"), "2025-12")
        self.assertEqual(previous_month("2026-11"), "2026-10")


if __name__ == "__main__":
    unittest.main()
