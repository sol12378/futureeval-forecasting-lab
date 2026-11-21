# ops-ledger

Append-only spend/submission ledger written by the GitHub Actions bot.
One JSON file per run under `runs/<YYYY-MM>/`. Never edit or delete records:
the budget guard sums them to enforce per-provider monthly caps, and a
malformed record intentionally stops the bot (fail closed).
Records contain costs, question URLs and statuses only - no forecasts or reasoning.
