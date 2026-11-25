import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

import research_sources as rs

REAL_CLIENT = httpx.AsyncClient


def mock_client(handler):
    def factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return REAL_CLIENT(*args, transport=httpx.MockTransport(handler), **kwargs)
    return factory


class PureFunctionTests(unittest.TestCase):
    def test_parse_extraction(self):
        self.assertEqual(rs.parse_extraction('ok {"fred_search": "US unemployment rate", "coingecko_ids": ["bitcoin", "BAD ID"]}'),
                         {"fred_search": "US unemployment rate", "coingecko_ids": ["bitcoin"]})
        self.assertEqual(rs.parse_extraction("no json"), {})
        self.assertEqual(rs.parse_extraction('{"fred_search": null, "coingecko_ids": []}'), {"coingecko_ids": []})

    def test_summarise_series(self):
        values = [(f"2025-{m:02d}-01", float(m)) for m in range(1, 13)] + [("2026-01-01", 20.0), ("2026-02-01", 21.0)]
        line = rs.summarise_series(values, "X")
        self.assertIn("latest 21 on 2026-02-01", line)
        self.assertIn("previous 20", line)
        self.assertIn("1y ago 2", line)  # value on 2025-02-01
        self.assertIn("12m range", line)
        self.assertEqual(rs.summarise_series([], "X"), "")

    def test_fmt_never_scientific(self):
        self.assertEqual(rs.fmt(56270.4), "56,270")
        self.assertEqual(rs.fmt(12345678.9), "12,345,679")
        self.assertEqual(rs.fmt(4.1), "4.1")
        self.assertEqual(rs.fmt(0.000123), "0.000123")
        self.assertNotIn("e", rs.fmt(1.5e9))

    def test_market_lines(self):
        poly = {"events": [{"title": "E", "markets": [
            {"question": "Will A?", "outcomes": '["Yes","No"]', "outcomePrices": '["0.12","0.88"]', "volume": "15000", "endDate": "2026-12-31T00:00:00Z"},
            {"question": "closed", "closed": True, "outcomes": '["Yes","No"]', "outcomePrices": '["1","0"]'},
            {"question": "broken", "outcomes": "nope", "outcomePrices": "[]"}]}]}
        lines = rs._polymarket_lines(poly)
        self.assertEqual(len(lines), 1)
        self.assertIn("Yes 12%", lines[0])
        mani = [{"question": "Will B?", "probability": 0.3, "uniqueBettorCount": 9, "closeTime": 1798675200000},
                {"question": "resolved", "isResolved": True, "probability": 1.0}]
        self.assertEqual(len(rs._manifold_lines(mani)), 1)

    def test_query_is_bounded(self):
        self.assertEqual(len(rs.search_query("word " * 500)), 380)


class HttpTests(unittest.TestCase):
    def test_markets_combines_sources_and_tolerates_errors(self):
        def handler(request: httpx.Request):
            if "polymarket" in request.url.host:
                return httpx.Response(500)
            return httpx.Response(200, json=[{"question": "Will B?", "probability": 0.42, "closeTime": None}])
        with patch.object(rs.httpx, "AsyncClient", mock_client(handler)):
            text = asyncio.run(rs.markets("Will B happen before 2027?"))
        self.assertIn("Manifold", text)
        self.assertIn("42%", text)

    def test_tavily_and_exa_format(self):
        def handler(request: httpx.Request):
            if request.url.host == "api.tavily.com":
                self.assertTrue(request.headers["Authorization"].startswith("Bearer "))
                return httpx.Response(200, json={"results": [{"title": "T", "url": "u", "content": "c", "published_date": "2026-11-20"}]})
            self.assertIn("x-api-key", request.headers)
            return httpx.Response(200, json={"results": [{"title": "E", "url": "v", "text": "body", "publishedDate": "2026-11-19T00:00:00Z"}]})
        with patch.dict("os.environ", {"TAVILY_API_KEY": "t", "EXA_API_KEY": "e"}), \
             patch.object(rs.httpx, "AsyncClient", mock_client(handler)):
            self.assertIn("[2026-11-20] T", asyncio.run(rs.tavily("q")))
            self.assertIn("[2026-11-19] E", asyncio.run(rs.exa("q")))

    def test_data_uses_fred_and_coingecko(self):
        def handler(request: httpx.Request):
            path = request.url.path
            if path.endswith("/series/search"):
                return httpx.Response(200, json={"seriess": [{"id": "UNRATE", "title": "Unemployment Rate", "units_short": "%", "frequency_short": "M"}]})
            if path.endswith("/series/observations"):
                obs = [{"date": "2026-10-01", "value": "4.3"}, {"date": "2026-09-01", "value": "."}, {"date": "2025-09-01", "value": "4.1"}]
                return httpx.Response(200, json={"observations": obs})
            return httpx.Response(200, json={"prices": [[1760000000000, 100.0], [1790000000000, 120.0]]})

        async def extract(prompt):
            return json.dumps({"fred_search": "unemployment rate", "coingecko_ids": ["bitcoin"]})

        with patch.dict("os.environ", {"FRED_API_KEY": "k"}), patch.object(rs.httpx, "AsyncClient", mock_client(handler)):
            text = asyncio.run(rs.data("Will US unemployment exceed 4.5%?", "BLS", extract))
        self.assertIn("FRED UNRATE", text)
        self.assertIn("latest 4.3", text)
        self.assertIn("CoinGecko bitcoin", text)

    def test_fred_errors_never_contain_the_key(self):
        def handler(request: httpx.Request):
            return httpx.Response(403)

        async def extract(prompt):
            return '{"fred_search": "cpi", "coingecko_ids": []}'
        with patch.dict("os.environ", {"FRED_API_KEY": "SECRETKEY123"}), patch.object(rs.httpx, "AsyncClient", mock_client(handler)):
            with self.assertRaises(rs.SourceError) as ctx:
                asyncio.run(rs.data("q", "c", extract))
        self.assertNotIn("SECRETKEY123", str(ctx.exception))
        self.assertNotIn("stlouisfed", str(ctx.exception))
        import logging
        self.assertGreaterEqual(logging.getLogger("httpx").level, logging.WARNING)

    def test_data_without_fred_key_skips_fred(self):
        async def extract(prompt):
            return '{"fred_search": "cpi", "coingecko_ids": []}'
        with patch.dict("os.environ", {"FRED_API_KEY": ""}):
            self.assertEqual(asyncio.run(rs.data("q", "c", extract)), "")


if __name__ == "__main__":
    unittest.main()


class ResolutionPageTests(unittest.TestCase):
    def test_resolution_urls(self):
        criteria = ("Resolves per [CDC](https://www.cdc.gov/measles/data.html), see also "
                    "https://www.metaculus.com/questions/1/ and https://www.cdc.gov/measles/data.html.")
        fine = "Backup: (https://example.org/a?b=1). Extra https://x.test/1 https://x.test/2 https://x.test/3"
        self.assertEqual(rs.resolution_urls(criteria, fine),
                         ["https://www.cdc.gov/measles/data.html", "https://example.org/a?b=1", "https://x.test/1"])
        self.assertEqual(rs.resolution_urls("no links here", None), [])

    def test_html_to_text(self):
        html = "<html><head><title>t</title></head><body><script>var x=1</script><p>Cases: 5,907</p><br>As of &amp; Nov</body></html>"
        text = rs.html_to_text(html)
        self.assertIn("Cases: 5,907", text)
        self.assertIn("As of & Nov", text)
        self.assertNotIn("var x", text)

    def test_reader_then_fallback_and_condense(self):
        prompts = []

        async def extract(prompt):
            prompts.append(prompt)
            return "latest 5,907 cases as of 2026-11-20"

        def handler(request: httpx.Request):
            if request.url.host == "r.jina.ai":
                if "cdc.gov" in str(request.url):
                    return httpx.Response(200, text="Title: Measles\n" + "Total cases 5,907 as of Nov 20. " * 10)
                return httpx.Response(429, text="rate limited")
            if request.url.host == "example.org":
                return httpx.Response(200, headers={"content-type": "text/html"},
                                      text="<p>" + "Backup table row 12 " * 20 + "</p>")
            return httpx.Response(404)

        criteria = "Resolves per https://www.cdc.gov/measles/data.html or https://example.org/t or https://gone.test/x"
        with patch.object(rs.httpx, "AsyncClient", mock_client(handler)):
            text = asyncio.run(rs.resolution_pages("How many measles cases?", criteria, "", "", extract))
        self.assertIn("https://www.cdc.gov/measles/data.html:", text)
        self.assertIn("https://example.org/t:", text)
        self.assertNotIn("gone.test", text)  # 404 on both paths -> skipped, not fatal
        self.assertEqual(len(prompts), 2)
        self.assertTrue(any("Total cases 5,907" in p for p in prompts))
        self.assertTrue(any("Backup table row 12" in p for p in prompts))

    def test_no_urls_no_calls(self):
        async def extract(prompt):
            raise AssertionError("should not be called")

        self.assertEqual(asyncio.run(rs.resolution_pages("q", "plain text", "", "", extract)), "")
