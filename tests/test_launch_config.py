import unittest
from forecast_config import config_counts, required_credentials_present

class LaunchConfigTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(config_counts({}), (1, 1))
    def test_valid_settings(self):
        self.assertEqual(config_counts({"RESEARCH_REPORTS_PER_QUESTION": "2", "PREDICTIONS_PER_RESEARCH_REPORT": "3"}), (2, 3))
    def test_reject_expensive_settings(self):
        with self.assertRaises(ValueError):
            config_counts({"PREDICTIONS_PER_RESEARCH_REPORT": "99"})
    def test_credentials(self):
        result=required_credentials_present({"METACULUS_TOKEN":"x", "ANTHROPIC_API_KEY":"y"})
        self.assertTrue(result["METACULUS_TOKEN"])
        self.assertFalse(result["OPENAI_API_KEY"])

if __name__ == "__main__":
    unittest.main()

# Allows the launch kit to validate model selection offline without installing forecasting-tools.
class ModelSelectionTests(unittest.TestCase):
    def test_models_and_asknews_gate(self):
        import sys
        from types import ModuleType
        from unittest.mock import patch
        from forecast_config import build_llm_config
        m = ModuleType('forecasting_tools')
        class FakeGeneralLlm:
            def __init__(self, **kwargs): self.params=kwargs
        m.GeneralLlm = FakeGeneralLlm
        creds={"METACULUS_TOKEN":"x", "ANTHROPIC_API_KEY":"a", "OPENAI_API_KEY":"g", "CLAUDE_MODEL":"", "GPT_MODEL":""}
        with patch.dict(sys.modules, {'forecasting_tools': m}):
            d=build_llm_config(creds)
            self.assertEqual(d['default'].params['model'], 'anthropic/claude-sonnet-5')
            self.assertEqual(d['parser'], 'openai/gpt-5.4-mini')
            self.assertEqual(d['researcher'], 'no_research')
            creds['ASKNEWS_CLIENT_ID']='id';creds['ASKNEWS_SECRET']='secret'
            self.assertEqual(build_llm_config(creds)['researcher'], 'asknews/news-summaries')
