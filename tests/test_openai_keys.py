import asyncio
import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from openai_keys import OpenAIKeyPool, is_quota_exhausted, make_failover_llm


class QuotaDetectionTests(unittest.TestCase):
    def test_quota_markers(self):
        self.assertTrue(is_quota_exhausted(Exception("Error code: 429 - {'error': {'code': 'insufficient_quota'}}")))
        self.assertTrue(is_quota_exhausted(Exception("You exceeded your current quota, please check your plan")))
        try:
            try:
                raise RuntimeError("insufficient_quota")
            except RuntimeError as inner:
                raise ValueError("wrapped") from inner
        except ValueError as outer:
            self.assertTrue(is_quota_exhausted(outer))

    def test_other_errors_do_not_switch(self):
        self.assertFalse(is_quota_exhausted(Exception("Rate limit reached for requests per min")))
        self.assertFalse(is_quota_exhausted(Exception("Incorrect API key provided")))
        self.assertFalse(is_quota_exhausted(None))


class KeyPoolTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "k-one"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_switch_once_then_stop(self):
        pool = OpenAIKeyPool({"OPENAI_API_KEY": "k-one", "OPENAI_API_KEY_2": "k-two"})
        pool.activate()
        self.assertEqual((pool.size, pool.slot, os.environ["OPENAI_API_KEY"]), (2, 1, "k-one"))
        self.assertTrue(pool.mark_exhausted(1))
        self.assertEqual((pool.slot, os.environ["OPENAI_API_KEY"]), (2, "k-two"))
        self.assertTrue(pool.mark_exhausted(1))  # stale concurrent report: no double switch
        self.assertEqual(pool.slot, 2)
        self.assertFalse(pool.mark_exhausted(2))  # nothing left
        self.assertEqual(pool.switched_from, [1])

    def test_single_key_and_placeholder(self):
        pool = OpenAIKeyPool({"OPENAI_API_KEY": "k-one", "OPENAI_API_KEY_2": "REPLACE_ME"})
        self.assertEqual(pool.size, 1)
        self.assertFalse(pool.mark_exhausted(1))

    def test_start_slot_is_clamped(self):
        self.assertEqual(OpenAIKeyPool({"OPENAI_API_KEY": "a", "OPENAI_API_KEY_2": "b"}, start_slot=2).slot, 2)
        self.assertEqual(OpenAIKeyPool({"OPENAI_API_KEY": "a"}, start_slot=2).slot, 1)


class FailoverLlmTests(unittest.TestCase):
    def test_invoke_retries_on_next_key_only_for_quota(self):
        calls = []

        class FakeGeneralLlm:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def invoke(self, prompt, system_prompt=None):
                key = os.environ["OPENAI_API_KEY"]
                calls.append(key)
                if key == "k-one":
                    raise Exception("insufficient_quota")
                if prompt == "boom":
                    raise Exception("Rate limit reached")
                return f"ok:{key}"

        fake = ModuleType("forecasting_tools")
        fake.GeneralLlm = FakeGeneralLlm
        with patch.dict(sys.modules, {"forecasting_tools": fake}), patch.dict(os.environ, {}, clear=False):
            pool = OpenAIKeyPool({"OPENAI_API_KEY": "k-one", "OPENAI_API_KEY_2": "k-two"})
            pool.activate()
            llm = make_failover_llm(pool, model="openai/gpt-6.2-sol")
            self.assertEqual(asyncio.run(llm.invoke("hi")), "ok:k-two")
            self.assertEqual(calls, ["k-one", "k-two"])
            with self.assertRaises(Exception):
                asyncio.run(llm.invoke("boom"))  # non-quota error is raised, no further switching
            self.assertEqual(pool.slot, 2)


if __name__ == "__main__":
    unittest.main()
