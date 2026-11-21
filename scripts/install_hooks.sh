#!/usr/bin/env bash
# One-time setup per clone: enable the privacy hooks and a noreply commit identity.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
chmod +x .githooks/* scripts/privacy_check.sh
if ! git config user.email | grep -q '@users.noreply.github.com$'; then
  echo "Set a GitHub noreply identity for this repo, e.g.:"
  echo "  git config user.name  <your-github-login>"
  echo "  git config user.email <your-github-login>@users.noreply.github.com"
fi
mkdir -p "$HOME/.config/futureeval-forecasting-lab"
echo "Hooks enabled. Optional: list personal strings (emails, real name, phone) one per line in"
echo "  $HOME/.config/futureeval-forecasting-lab/denylist.txt   (stays on this machine, never committed)"
