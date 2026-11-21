"""Fail before an expensive run if required credentials are absent; never print secret values."""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast_config import config_counts, required_credentials_present


def main() -> int:
    credentials = required_credentials_present()
    missing = [k for k, valid in credentials.items() if not valid]
    if missing:
        print("SETUP_ERROR: missing Secrets:", ", ".join(missing))
        return 2
    try:
        research_count, prediction_count = config_counts()
    except ValueError as exc:
        print("CONFIG_ERROR:", exc)
        return 2
    print(f"PREFLIGHT_OK: research_reports={research_count}, predictions_per_report={prediction_count}")
    if not (os.environ.get("ASKNEWS_CLIENT_ID") and os.environ.get("ASKNEWS_SECRET")):
        print("WARNING: AskNews not configured; production runs have NO fresh external research. Configure it to improve time-sensitive forecasts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
