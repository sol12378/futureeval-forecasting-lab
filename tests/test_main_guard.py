"""Offline tests of main.py control flow (no network, no LLM calls)."""
import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
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
        os.environ.update(ENABLE_LIVE_FORECASTS="true", CLAUDE_MONTHLY_BUDGET_PER_KEY_USD="1")
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

    def test_research_failure_falls_back_to_no_research(self):
        from budget_guard import BudgetGuard

        guard = BudgetGuard.from_env("tournament", {"LEDGER_DIR": self.tmp.name})
        bot = self.main.GuardedBot(llms={"default": "x", "parser": "x", "summarizer": "x", "researcher": "no_research"})
        bot.budget_guard = guard

        class Q:
            page_url = "https://example/9"

        async def boom(_self, _q):
            raise ConnectionError("asknews down")

        with patch.object(self.main.FallTemplateBot2026, "run_research", boom):
            self.assertEqual(asyncio.run(bot.run_research(Q())), "")
        self.assertEqual(guard.research_failures, 1)

    def test_verify_on_metaculus(self):
        from types import SimpleNamespace

        t0 = datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc)

        def q(post, ts, flag=True):
            hist = [SimpleNamespace(timestamp=ts)] if ts else None
            return SimpleNamespace(id_of_post=post, id_of_question=post, page_url=f"https://m/{post}",
                                   previous_forecasts=hist, already_forecasted=flag)

        client = SimpleNamespace(get_all_open_questions_from_tournament=lambda _t: [
            q(1, t0.replace(minute=5)), q(2, t0.replace(hour=1)), q(3, None, True)])
        lines, ok = self.main.verify_on_metaculus(client, "bot-testing-area", {"1:1", "3:3"}, t0)
        self.assertTrue(ok)
        self.assertTrue(lines[0].startswith("VERIFIED on Metaculus: https://m/1"))
        lines, ok = self.main.verify_on_metaculus(client, "bot-testing-area", {"2:2", "4:4"}, t0)
        self.assertFalse(ok)  # 2 only has an old forecast
        self.assertTrue(any("Could not re-read 4:4" in line for line in lines))

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
