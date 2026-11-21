import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from forecast_config import build_llm_config, config_counts, has_asknews, required_credentials_present


class LaunchConfigTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(config_counts({}), (1, 1))

    def test_valid_settings(self):
        env = {"RESEARCH_REPORTS_PER_QUESTION": "2", "PREDICTIONS_PER_RESEARCH_REPORT": "3"}
        self.assertEqual(config_counts(env), (2, 3))

    def test_empty_string_uses_default(self):
        self.assertEqual(config_counts({"PREDICTIONS_PER_RESEARCH_REPORT": ""}), (1, 1))

    def test_reject_expensive_settings(self):
        with self.assertRaises(ValueError):
            config_counts({"PREDICTIONS_PER_RESEARCH_REPORT": "99"})
        with self.assertRaises(ValueError):
            config_counts({"RESEARCH_REPORTS_PER_QUESTION": "0"})

    def test_credentials(self):
        result = required_credentials_present({"METACULUS_TOKEN": "x", "ANTHROPIC_API_KEY": "REPLACE_ME"})
        self.assertTrue(result["METACULUS_TOKEN"])
        self.assertFalse(result["ANTHROPIC_API_KEY"])
        self.assertFalse(result["OPENAI_API_KEY"])

    def test_asknews_requires_both(self):
        self.assertFalse(has_asknews({"ASKNEWS_CLIENT_ID": "id"}))
        self.assertTrue(has_asknews({"ASKNEWS_CLIENT_ID": "id", "ASKNEWS_SECRET": "s"}))


class ModelSelectionTests(unittest.TestCase):
    def _fake_module(self):
        m = ModuleType("forecasting_tools")

        class FakeGeneralLlm:
            def __init__(self, **kwargs):
                self.params = kwargs

        m.GeneralLlm = FakeGeneralLlm
        return m

    def test_models_and_asknews_gate(self):
        creds = {"METACULUS_TOKEN": "x", "ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "g", "CLAUDE_MODEL": "", "GPT_MODEL": ""}
        with patch.dict(sys.modules, {"forecasting_tools": self._fake_module()}):
            d = build_llm_config(creds)
            self.assertEqual(d["default"].params["model"], "anthropic/claude-opus-5-5")
            self.assertIsNone(d["default"].params["temperature"])
            self.assertEqual(d["parser"].params["model"], "openai/gpt-5.4-mini")
            self.assertEqual(d["researcher"], "no_research")
            creds.update(ASKNEWS_CLIENT_ID="id", ASKNEWS_SECRET="secret", CLAUDE_MODEL="anthropic/claude-sonnet-5-5")
            d = build_llm_config(creds)
            self.assertEqual(d["researcher"], "asknews/news-summaries")
            self.assertEqual(d["default"].params["model"], "anthropic/claude-sonnet-5-5")

    def test_missing_secret_raises(self):
        with self.assertRaises(RuntimeError):
            build_llm_config({"METACULUS_TOKEN": "x"})


if __name__ == "__main__":
    unittest.main()
