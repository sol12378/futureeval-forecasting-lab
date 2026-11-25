import asyncio
import os
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from api_keys import KeyPool, is_credit_exhausted, make_failover_llm


class CreditDetectionTests(unittest.TestCase):
    def test_openai_markers(self):
        self.assertTrue(is_credit_exhausted(Exception("Error code: 429 - {'error': {'code': 'insufficient_quota'}}"), "openai"))
        self.assertTrue(is_credit_exhausted(Exception("You exceeded your current quota, please check your plan"), "openai"))
        try:
            try:
                raise RuntimeError("insufficient_quota")
            except RuntimeError as inner:
                raise ValueError("wrapped") from inner
        except ValueError as outer:
            self.assertTrue(is_credit_exhausted(outer, "openai"))

    def test_anthropic_markers(self):
        self.assertTrue(is_credit_exhausted(Exception(
            'AnthropicException - {"type":"error","error":{"type":"invalid_request_error","message":'
            '"Your credit balance is too low to access the Anthropic API."}}'), "anthropic"))
        self.assertTrue(is_credit_exhausted(Exception('{"error":{"type":"billing_error"}}'), "anthropic"))
        self.assertTrue(is_credit_exhausted(Exception('429 {"details":{"error_code":"enforced_spend_limit_reached"}}'), "anthropic"))

    def test_other_errors_do_not_switch(self):
        for provider in ("openai", "anthropic"):
            self.assertFalse(is_credit_exhausted(Exception("Rate limit reached for requests per min"), provider))
            self.assertFalse(is_credit_exhausted(Exception("Incorrect API key provided / authentication_error"), provider))
            self.assertFalse(is_credit_exhausted(Exception("overloaded_error"), provider))
            self.assertFalse(is_credit_exhausted(None, provider))
        # markers are provider specific
        self.assertFalse(is_credit_exhausted(Exception("insufficient_quota"), "anthropic"))


class KeyPoolTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "k-one", "ANTHROPIC_API_KEY": "a-one"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_switch_once_then_stop(self):
        for provider, env_name in (("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")):
            pool = KeyPool(provider, {env_name: "k-one", f"{env_name}_2": "k-two"})
            pool.activate()
            self.assertEqual((pool.size, pool.slot, os.environ[env_name]), (2, 1, "k-one"))
            self.assertTrue(pool.mark_exhausted(1))
            self.assertEqual((pool.slot, os.environ[env_name]), (2, "k-two"))
            self.assertTrue(pool.mark_exhausted(1))  # stale concurrent report: no double switch
            self.assertEqual(pool.slot, 2)
            self.assertFalse(pool.mark_exhausted(2))  # nothing left
            self.assertEqual(pool.switched_from, [1])

    def test_pools_are_independent(self):
        o = KeyPool("openai", {"OPENAI_API_KEY": "o1", "OPENAI_API_KEY_2": "o2"})
        a = KeyPool("anthropic", {"ANTHROPIC_API_KEY": "a1", "ANTHROPIC_API_KEY_2": "a2"})
        o.activate()
        a.activate()
        o.mark_exhausted(1)
        self.assertEqual((os.environ["OPENAI_API_KEY"], os.environ["ANTHROPIC_API_KEY"]), ("o2", "a1"))

    def test_single_key_placeholder_and_clamp(self):
        pool = KeyPool("openai", {"OPENAI_API_KEY": "k-one", "OPENAI_API_KEY_2": "REPLACE_ME"})
        self.assertEqual(pool.size, 1)
        self.assertFalse(pool.mark_exhausted(1))
        self.assertEqual(KeyPool("anthropic", {"ANTHROPIC_API_KEY": "a", "ANTHROPIC_API_KEY_2": "b"}, start_slot=2).slot, 2)
        self.assertEqual(KeyPool("anthropic", {"ANTHROPIC_API_KEY": "a"}, start_slot=2).slot, 1)
        with self.assertRaises(ValueError):
            KeyPool("gemini", {})


class FailoverLlmTests(unittest.TestCase):
    def test_invoke_retries_on_next_key_only_for_credit_errors(self):
        calls = []

        class FakeGeneralLlm:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def invoke(self, prompt, system_prompt=None):
                key = os.environ["ANTHROPIC_API_KEY"]
                calls.append(key)
                if key == "a-one":
                    raise Exception("Your credit balance is too low to access the Anthropic API.")
                if prompt == "boom":
                    raise Exception("rate_limit_error")
                return f"ok:{key}"

        fake = ModuleType("forecasting_tools")
        fake.GeneralLlm = FakeGeneralLlm
        with patch.dict(sys.modules, {"forecasting_tools": fake}), patch.dict(os.environ, {}, clear=False):
            pool = KeyPool("anthropic", {"ANTHROPIC_API_KEY": "a-one", "ANTHROPIC_API_KEY_2": "a-two"})
            pool.activate()
            llm = make_failover_llm(pool, model="anthropic/claude-opus-5-5")
            self.assertEqual(asyncio.run(llm.invoke("hi")), "ok:a-two")
            self.assertEqual(calls, ["a-one", "a-two"])
            with self.assertRaises(Exception):
                asyncio.run(llm.invoke("boom"))  # non-credit error is raised, no further switching
            self.assertEqual(pool.slot, 2)


if __name__ == "__main__":
    unittest.main()


class KeyOrderTests(unittest.TestCase):
    ENV = {"ANTHROPIC_API_KEY": "a1", "ANTHROPIC_API_KEY_2": "a2", "OPENAI_API_KEY": "o1", "OPENAI_API_KEY_2": "o2"}

    def test_key2_first_then_key1(self):
        from api_keys import configured_key_order

        env = dict(self.ENV, GPT_KEY_ORDER="2,1")
        self.assertEqual(configured_key_order("openai", env), (2, 1))
        self.assertIsNone(configured_key_order("anthropic", env))
        pool = KeyPool("openai", env)
        with patch.dict(os.environ, {}):
            pool.activate()
            self.assertEqual((pool.slots, pool.slot, os.environ["OPENAI_API_KEY"]), ((2, 1), 2, "o2"))
            self.assertTrue(pool.mark_exhausted(2))
            self.assertEqual((pool.slot, os.environ["OPENAI_API_KEY"]), (1, "o1"))
            self.assertFalse(pool.mark_exhausted(1))
            self.assertEqual(pool.switched_from, [2])

    def test_single_key_order_excludes_the_other(self):
        pool = KeyPool("anthropic", dict(self.ENV, CLAUDE_KEY_ORDER="2"))
        self.assertEqual((pool.size, pool.slots, pool.slot), (1, (2,), 2))
        self.assertFalse(pool.mark_exhausted(2))
        # A missing secret is skipped even if it is named in the order.
        pool = KeyPool("anthropic", {"ANTHROPIC_API_KEY": "a1", "CLAUDE_KEY_ORDER": "2,1"})
        self.assertEqual(pool.slots, (1,))

    def test_start_slot_follows_order(self):
        pool = KeyPool("openai", self.ENV, start_slot=1, order=(2, 1))
        self.assertEqual(pool.slot, 1)
        self.assertFalse(pool.mark_exhausted(1))  # key 1 is last in this order

    def test_bad_orders_rejected(self):
        from api_keys import configured_key_order

        for bad in ("3", "1,1", "a", "0", "2,1,2", ","):
            with self.assertRaises(ValueError, msg=bad):
                configured_key_order("anthropic", {"CLAUDE_KEY_ORDER": bad})
