import json
import tempfile
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
    (d / name).write_text(json.dumps(record))


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
        (d / "bad.json").write_text(json.dumps({"cost_usd": {"anthropic": -5}}))
        with self.assertRaises(LedgerError):
            self.guard()

    def test_require_ledger(self):
        with self.assertRaises(LedgerError):
            BudgetGuard.from_env("tournament", {"LEDGER_DIR": str(self.ledger / "missing"), "REQUIRE_LEDGER": "true"}, now=NOW)

    def test_invalid_env(self):
        with self.assertRaises(ValueError):
            self.guard(BUDGET_SOFT_RATIO="0.95", BUDGET_HARD_RATIO="0.9")
        with self.assertRaises(ValueError):
            self.guard(CLAUDE_MONTHLY_BUDGET_USD="abc")

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

    def test_previous_month(self):
        self.assertEqual(previous_month("2026-01"), "2025-12")
        self.assertEqual(previous_month("2026-11"), "2026-10")


if __name__ == "__main__":
    unittest.main()
