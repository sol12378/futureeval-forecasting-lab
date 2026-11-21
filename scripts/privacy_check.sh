#!/usr/bin/env bash
# Privacy/secret gate for this PUBLIC repo. Used by git hooks and CI.
#   privacy_check.sh staged          check the staged diff (pre-commit)
#   privacy_check.sh range <args>    check commits in a rev range (pre-push / CI)
# Checks: gitleaks secret scan, forbidden file names, noreply-only commit identities,
# and an optional LOCAL denylist of personal strings that is never committed:
#   ${FUTUREEVAL_DENYLIST:-$HOME/.config/futureeval-forecasting-lab/denylist.txt}
set -euo pipefail
mode="${1:-staged}"; shift || true
root="$(git rev-parse --show-toplevel)"
denylist="${FUTUREEVAL_DENYLIST:-$HOME/.config/futureeval-forecasting-lab/denylist.txt}"
fail=0
say() { printf '%s\n' "privacy_check: $*" >&2; }

allowed_email() {
  case "$1" in
    *@users.noreply.github.com) return 0 ;;
    *) return 1 ;;
  esac
}

forbidden_path() {
  case "$(basename "$1")" in
    .env.example*|.env.template) return 1 ;;
    .env|.env.*|*.pem|*.key|*.p12|*.pfx|id_rsa*|id_ed25519*|credentials*.json|*secret*.json|*.sqlite|*.db) return 0 ;;
  esac
  return 1
}

if ! command -v gitleaks >/dev/null 2>&1; then
  say "gitleaks is not installed (brew install gitleaks). Refusing to continue."; exit 1
fi

if [ "$mode" = "staged" ]; then
  ident="$(git var GIT_AUTHOR_IDENT | sed -E 's/.*<([^>]*)>.*/\1/')"
  allowed_email "$ident" || { say "author email '$ident' is not a GitHub noreply address. Run: git config user.email <id>+<login>@users.noreply.github.com"; fail=1; }
  while IFS= read -r f; do
    forbidden_path "$f" && { say "refusing to commit sensitive file: $f"; fail=1; }
  done < <(git diff --cached --name-only --diff-filter=ACMR)
  gitleaks git --staged --no-banner --redact --exit-code 1 "$root" >/dev/null 2>&1 \
    || { say "gitleaks found a possible secret in staged changes (run: gitleaks git --staged -v .)"; fail=1; }
  added="$(git diff --cached -U0 --no-color | grep -E '^\+' | grep -vE '^\+\+\+' || true)"
else
  commits="$(git rev-list "$@")"
  for c in $commits; do
    for e in "$(git show -s --format=%ae "$c")" "$(git show -s --format=%ce "$c")"; do
      allowed_email "$e" || [ "$e" = "noreply@github.com" ] || { say "commit $c uses non-noreply email '$e'"; fail=1; }
    done
    while IFS= read -r f; do
      forbidden_path "$f" && { say "commit $c adds sensitive file: $f"; fail=1; }
    done < <(git diff-tree --no-commit-id --name-only -r --diff-filter=ACMR "$c")
  done
  if [ -n "$commits" ]; then
    first="$(printf '%s\n' "$commits" | tail -1)"
    log_opts="$first^..$(printf '%s\n' "$commits" | head -1)"
    git rev-parse -q --verify "$first^" >/dev/null || log_opts="$(printf '%s\n' "$commits" | head -1)"
    gitleaks git --no-banner --redact --exit-code 1 --log-opts="$log_opts" "$root" >/dev/null 2>&1 \
      || { say "gitleaks found a possible secret in commits to push (run: gitleaks git -v --log-opts='$log_opts' .)"; fail=1; }
    added="$(git log -p --no-color --format= $log_opts | grep -E '^\+' | grep -vE '^\+\+\+' || true)"
  else
    added=""
  fi
fi

if [ -f "$denylist" ] && [ -n "$added" ]; then
  while IFS= read -r needle || [ -n "$needle" ]; do
    needle="${needle%%#*}"; needle="$(printf '%s' "$needle" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
    [ -z "$needle" ] && continue
    if printf '%s' "$added" | grep -qiF -- "$needle"; then
      say "added lines contain an entry from your local denylist (value not shown)"; fail=1
    fi
  done < "$denylist"
fi

[ "$fail" -eq 0 ] || { say "BLOCKED. Fix the issues above (this repo is public)."; exit 1; }
