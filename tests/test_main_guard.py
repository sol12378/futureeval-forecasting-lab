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
        bot.research_sources = ("asknews",)

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

    def test_asknews_cap_skips_search_without_calling_it(self):
        from budget_guard import BudgetGuard

        guard = BudgetGuard.from_env("tournament", {"LEDGER_DIR": self.tmp.name, "ASKNEWS_TOTAL_CALL_CAP": "0"})
        bot = self.main.GuardedBot(llms={"default": "x", "parser": "x", "summarizer": "x",
                                         "researcher": "asknews/news-summaries"})
        bot.budget_guard = guard
        bot.research_sources = ("asknews",)

        class Q:
            page_url = "https://example/7"

        async def never(_self, _q):
            raise AssertionError("AskNews must not be called past the cap")

        with patch.object(self.main.FallTemplateBot2026, "run_research", never):
            self.assertEqual(asyncio.run(bot.run_research(Q())), "")
        self.assertEqual(guard.asknews_calls, 0)

    def test_numeric_parsing_pins_six_percentiles(self):
        from types import SimpleNamespace

        bot = self.main.GuardedBot(llms={"default": "x", "parser": "x", "summarizer": "x", "researcher": "no_research"})
        msg = bot._create_single_distribution_parsing_message(SimpleNamespace(group_question_option=None))
        self.assertIn("10, 20, 40, 60, 80 and 90", msg)
        self.assertIn("Ignore any other percentiles", msg)

    def test_ensemble_round_robin_routes_default_llm(self):
        from types import SimpleNamespace

        bot = self.main.GuardedBot(llms={"default": "claude-x", "gpt_forecaster": "gpt-x", "parser": "x",
                                         "summarizer": "x", "researcher": "no_research"})
        bot.forecaster_plan = ("claude", "gpt", "claude")
        seen = []

        async def fake_make(_self, question, research):
            seen.append(_self.get_llm("default"))
            return "ok"

        q = SimpleNamespace(id_of_post=5, id_of_question=6, page_url="https://m/5")
        with patch.object(self.main.FallTemplateBot2026, "_make_prediction", fake_make):
            for _ in range(4):
                asyncio.run(bot._make_prediction(q, ""))
        self.assertEqual(seen, ["claude-x", "gpt-x", "claude-x", "claude-x"])
        self.assertEqual(bot.get_llm("default"), "claude-x")  # outside a slot -> Claude

    def test_research_merges_sources_and_tolerates_failures(self):
        from types import SimpleNamespace

        from budget_guard import BudgetGuard

        guard = BudgetGuard.from_env("tournament", {"LEDGER_DIR": self.tmp.name})
        bot = self.main.GuardedBot(llms={"default": "x", "parser": "x", "summarizer": "x", "researcher": "no_research"})
        bot.budget_guard = guard
        bot.research_sources = ("claude_web", "gpt_web", "asknews")

        async def fake_with(_self, source, question):
            if source == "gpt_web":
                raise TimeoutError("slow")
            return f"facts from {source}"

        q = SimpleNamespace(page_url="https://m/8")
        with patch.object(self.main.GuardedBot, "_research_with", fake_with):
            text = asyncio.run(bot.run_research(q))
        self.assertIn("Web research (Claude)", text)
        self.assertIn("News (AskNews)", text)
        self.assertNotIn("GPT", text)
        self.assertEqual(guard.research_failures, 1)

    def test_prompt_guard_and_mc_floor(self):
        from types import SimpleNamespace

        from forecasting_tools import PredictedOption, PredictedOptionList, ReasonedPrediction

        bot = self.main.GuardedBot(llms={"default": "x", "parser": "x", "summarizer": "x", "researcher": "no_research"})
        q = SimpleNamespace(conditional_type=None, close_time="2026-12-01", scheduled_resolution_time="2026-12-31")
        guard_text = bot._get_conditional_disclaimer_if_necessary(q)
        self.assertIn("assume it has not happened yet", guard_text)
        self.assertIn("units", guard_text)

        async def fake_mc(_self, question, research):
            opts = [PredictedOption(option_name="A", probability=0.999), PredictedOption(option_name="B", probability=0.001),
                    PredictedOption(option_name="C", probability=0.0)]
            return ReasonedPrediction(prediction_value=PredictedOptionList(predicted_options=opts), reasoning="r")

        with patch.object(self.main.FallTemplateBot2026, "_run_forecast_on_multiple_choice", fake_mc):
            result = asyncio.run(bot._run_forecast_on_multiple_choice(q, ""))
        probs = [o.probability for o in result.prediction_value.predicted_options]
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(all(p >= 0.0098 for p in probs))

    def test_find_missed(self):
        from datetime import timedelta
        from types import SimpleNamespace

        now = datetime(2026, 11, 21, 6, 0, tzinfo=timezone.utc)
        qs = [
            SimpleNamespace(open_time=now - timedelta(hours=3), already_forecasted=False, page_url="https://m/1"),  # missed
            SimpleNamespace(open_time=now - timedelta(hours=3), already_forecasted=True, page_url="https://m/2"),
            SimpleNamespace(open_time=now - timedelta(minutes=20), already_forecasted=False, page_url="https://m/3"),  # too fresh
            SimpleNamespace(open_time=now + timedelta(hours=1), already_forecasted=False, page_url="https://m/4"),  # upcoming
            SimpleNamespace(open_time=None, already_forecasted=False, page_url="https://m/5"),
        ]
        self.assertEqual(self.main.find_missed(qs, now), ["https://m/1"])

    def test_wide_percentiles_with_fallback(self):
        from types import SimpleNamespace

        bot = self.main.GuardedBot(llms={"default": "x", "parser": "x", "summarizer": "x", "researcher": "no_research"})
        q = SimpleNamespace(page_url="https://m/n", group_question_option=None)
        prompt = "intro\n" + self.main.NUMERIC_BLOCK + "\nend"
        seen = []

        async def fake(_self, question, p):
            msg = _self._create_single_distribution_parsing_message(question)
            seen.append(("97.5" in p, "13 percentiles" in msg))
            if "97.5" in p and len(seen) == 1:
                raise ValueError("too far from bounds")
            return "ok"

        with patch.object(self.main.FallTemplateBot2026, "_numeric_prompt_to_forecast", fake):
            self.assertEqual(asyncio.run(bot._numeric_prompt_to_forecast(q, prompt)), "ok")
        self.assertEqual(seen, [(True, True), (False, False)])  # wide attempt, then 6-point fallback
        self.assertIsNone(self.main._expand_percentile_block("changed template", self.main.NUMERIC_BLOCK, "XX"))

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
