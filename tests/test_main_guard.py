"""Offline tests of main.py control flow (no network, no LLM calls)."""
import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

FAKE_ENV = {"METACULUS_TOKEN": "t", "ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"}


class MainGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {**FAKE_ENV, "LEDGER_DIR": self.tmp.name, "ENABLE_LIVE_FORECASTS": ""})
        self.env.start()
        import main  # noqa: PLC0415  (imports forecasting_tools; needs the uv env)

        self.main = main

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_live_disabled_makes_no_calls(self):
        with patch.object(self.main, "MetaculusClient") as client, patch.object(self.main, "build_llm_config") as llms:
            self.assertEqual(self.main.run("tournament"), 0)
            client.assert_not_called()
            llms.assert_not_called()

    def test_metaculus_cup_out_of_scope(self):
        os.environ["ENABLE_LIVE_FORECASTS"] = "true"
        with patch.object(self.main, "MetaculusClient") as client:
            self.assertEqual(self.main.run("metaculus_cup"), 0)
            client.assert_not_called()

    def test_hard_budget_refuses_before_any_call(self):
        os.environ.update(ENABLE_LIVE_FORECASTS="true", CLAUDE_MONTHLY_BUDGET_USD="1")
        from budget_guard import current_month

        d = Path(self.tmp.name) / "runs" / current_month()
        d.mkdir(parents=True)
        (d / "x.json").write_text('{"mode": "tournament", "cost_usd": {"anthropic": 0.95, "openai": 0}}')
        with patch.object(self.main, "MetaculusClient") as client:
            self.assertEqual(self.main.run("tournament"), self.main.EXIT_BUDGET)
            client.assert_not_called()

    def test_unpriced_model_refused(self):
        os.environ.update(ENABLE_LIVE_FORECASTS="true", CLAUDE_MODEL="anthropic/claude-imaginary-9")
        with patch.object(self.main, "MetaculusClient") as client:
            self.assertEqual(self.main.run("tournament"), self.main.EXIT_CONFIG)
            client.assert_not_called()

    def test_guarded_bot_skips_question_when_budget_exhausted(self):
        from budget_guard import BudgetExceeded, BudgetGuard

        guard = BudgetGuard.from_env("tournament", {"LEDGER_DIR": self.tmp.name, "RUN_COST_CAP_USD": "1"})
        guard.add_cost("anthropic", 1.0)
        bot = self.main.GuardedBot(llms={"default": "x", "parser": "x", "summarizer": "x", "researcher": "no_research"})
        bot.budget_guard = guard

        class Q:
            id_of_post, id_of_question, page_url = 1, 2, "https://example/1"

        with self.assertRaises(BudgetExceeded):
            asyncio.run(bot._run_individual_question(Q()))
        self.assertEqual(guard.questions["1:2"]["status"], "skipped_budget")


if __name__ == "__main__":
    unittest.main()
