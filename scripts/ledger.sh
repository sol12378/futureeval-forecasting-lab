#!/usr/bin/env bash
# Sync the append-only spend ledger kept on the `ops-ledger` branch.
#   pull: check the branch out into $LEDGER_DIR (fails if the branch is missing -> fail closed)
#   push: commit new run records and push, retrying on races with other runs
set -euo pipefail
DIR="${LEDGER_DIR:-ops_ledger}"
BRANCH="${LEDGER_BRANCH:-ops-ledger}"

case "${1:-}" in
  pull)
    git fetch --no-tags origin "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
    git worktree add --detach "$DIR" "origin/${BRANCH}"
    echo "Ledger loaded: $(find "$DIR/runs" -name '*.json' 2>/dev/null | wc -l | tr -d ' ') record(s)"
    ;;
  push)
    if [ ! -d "$DIR" ]; then echo "No ledger checkout; nothing to push"; exit 0; fi
    cd "$DIR"
    git add -A runs 2>/dev/null || true
    if git diff --cached --quiet; then echo "No new ledger records"; exit 0; fi
    git -c user.name="futureeval-ledger-bot" \
        -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
        commit -q -m "ledger: ${GITHUB_WORKFLOW:-local} run ${GITHUB_RUN_ID:-local}"
    for i in 1 2 3 4 5; do
      if git push -q origin "HEAD:refs/heads/${BRANCH}"; then echo "Ledger pushed"; exit 0; fi
      echo "Push rejected (attempt $i); rebasing on latest ledger"
      git fetch -q --no-tags origin "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
      git rebase -q "origin/${BRANCH}"
      sleep $((i * 3))
    done
    echo "LEDGER_PUSH_FAILED"; exit 1
    ;;
  *) echo "usage: $0 pull|push"; exit 2 ;;
esac
