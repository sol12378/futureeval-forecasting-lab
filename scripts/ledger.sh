#!/usr/bin/env bash
# Sync the append-only spend ledger kept on the `ops-ledger` branch.
#   pull           check the branch out into $LEDGER_DIR (fails if missing -> fail closed)
#   collect OUT    copy run records created by this job into OUT/ (for upload as an artifact)
#   ingest IN      validate records from IN/ and copy them into $LEDGER_DIR
#   push           commit new run records and push, retrying on races with other runs
# The forecasting job only needs pull+collect (read-only token). A separate job with
# contents:write runs pull+ingest+push and never installs third-party Python packages.
set -euo pipefail
DIR="${LEDGER_DIR:-ops_ledger}"
BRANCH="${LEDGER_BRANCH:-ops-ledger}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${1:-}" in
  pull)
    git fetch --no-tags origin "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
    git worktree add --detach "$DIR" "origin/${BRANCH}"
    echo "Ledger loaded: $(find "$DIR/runs" -name '*.json' 2>/dev/null | wc -l | tr -d ' ') record(s)"
    ;;
  collect)
    OUT="${2:?usage: collect OUTDIR}"
    mkdir -p "$OUT"
    [ -d "$DIR" ] || { echo "No ledger checkout"; exit 0; }
    (cd "$DIR" && git ls-files --others --exclude-standard -- runs) | while IFS= read -r f; do
      mkdir -p "$OUT/$(dirname "$f")"
      cp "$DIR/$f" "$OUT/$f"
      echo "collected $f"
    done
    ;;
  ingest)
    IN="${2:?usage: ingest INDIR}"
    [ -d "$IN/runs" ] || { echo "No records to ingest"; exit 0; }
    find "$IN/runs" -type f | while IFS= read -r src; do
      rel="${src#"$IN"/}"
      # Only runs/YYYY-MM/<safe-name>.json for the current or previous month is accepted.
      if ! printf '%s' "$rel" | grep -Eq '^runs/[0-9]{4}-[0-9]{2}/[A-Za-z0-9._-]+\.json$'; then
        echo "REJECTED path: $rel"; exit 1
      fi
      month="$(basename "$(dirname "$rel")")"
      # Same schema check the budget guard applies when summing spend (stdlib-only Python).
      python3 "$ROOT/budget_guard.py" validate "$src" "$month" || { echo "REJECTED content: $rel"; exit 1; }
      if [ -e "$DIR/$rel" ]; then echo "REJECTED overwrite: $rel"; exit 1; fi
      mkdir -p "$DIR/$(dirname "$rel")"
      cp "$src" "$DIR/$rel"
      echo "ingested $rel"
    done
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
  *) echo "usage: $0 pull|collect OUT|ingest IN|push"; exit 2 ;;
esac
