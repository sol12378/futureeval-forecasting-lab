"""Idempotently patch the CURRENT official Metaculus fork to use dual-provider keys.

Run in a clean fork of Metaculus/metac-bot-template. On unexpected upstream changes,
this script aborts rather than risk deploying a half-patched bot.
"""
from __future__ import annotations
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "main.py"
if not path.is_file():
    raise SystemExit("main.py missing. Unzip this launch kit INTO your official Metaculus template fork.")
content = path.read_text(encoding="utf-8")
if "from forecast_config import build_llm_config, config_counts" in content:
    print("Already patched; no changes")
    raise SystemExit(0)

if "from typing import Literal\n" not in content:
    raise SystemExit("Unexpected upstream main.py imports; aborting patch")
content = content.replace(
    "from typing import Literal\n",
    "from typing import Literal\n\nfrom forecast_config import build_llm_config, config_counts\n",
    1,
)
if "    template_bot = FallTemplateBot2026(\n" not in content:
    raise SystemExit("Unexpected upstream bot class; aborting patch")
content = content.replace(
    "    template_bot = FallTemplateBot2026(\n",
    "    research_count, prediction_count = config_counts()\n"
    "    template_bot = FallTemplateBot2026(\n",
    1,
)
content = content.replace(
    "        research_reports_per_question=1,\n",
    "        research_reports_per_question=research_count,\n",
    1,
)
content = content.replace(
    "        predictions_per_research_report=5,\n",
    "        predictions_per_research_report=prediction_count,\n",
    1,
)
# Remove exactly the example commented llms= block in the constructor.
pattern = r"(?m)^        # llms=\{\n(?:^        #.*\n)+?^        # \},\n"
content, n = re.subn(pattern, "        llms=build_llm_config(),\n", content, count=1)
if n != 1:
    raise SystemExit("Upstream example model block changed; aborting patch")
ast.parse(content)
path.write_text(content, encoding="utf-8")
print("PATCH_OK: main.py configured for Claude + GPT. Next run: python scripts/preflight.py")
