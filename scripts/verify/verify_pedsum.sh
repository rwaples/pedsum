#!/usr/bin/env bash
# verify_pedsum.sh — fresh-computer install verification for pedsum (public, no auth).
#
# Mimics a brand-new user on a clean machine: clones pedsum from GitHub into a
# throwaway workdir, creates its own conda env via the documented
# environment.yml, runs the documented CLI smoke (version + validate +
# summarize), asserts concrete outputs, then tears everything down. Reads no
# sibling repo or project files — only lib.sh beside it. Run from any
# directory; needs only `git` + `conda`.
#
# Usage:
#   bash verify_pedsum.sh [--pedsum-ref REF] [--pedsum-url URL] [--full] [--keep]
#
# Flags (env-var equivalents in parens):
#   --pedsum-ref REF   git ref to check out          (PEDSUM_REF; default: main)
#   --pedsum-url URL   clone URL                      (PEDSUM_URL; default: rwaples/pedsum)
#   --full             also run the pedsum test suite (~80 s; default: skip)
#   --keep             keep the env + workdir on exit (for debugging)
#   -h, --help         show this help

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

PEDSUM_URL="${PEDSUM_URL:-https://github.com/rwaples/pedsum.git}"
PEDSUM_REF="${PEDSUM_REF:-main}"
FULL=0
EXPECT_VERSION="0.10.0"

while [ $# -gt 0 ]; do
  case "$1" in
    --pedsum-ref) PEDSUM_REF="$2"; shift 2 ;;
    --pedsum-url) PEDSUM_URL="$2"; shift 2 ;;
    --full)       FULL=1; shift ;;
    --keep)       KEEP=1; shift ;;
    -h|--help)    show_help "${BASH_SOURCE[0]}" ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

step "Preflight"
preflight_tools

make_workdir

step "Clone pedsum ($PEDSUM_REF)"
clone "$PEDSUM_URL" "$PEDSUM_REF" "$WORK/pedsum"
ok "cloned pedsum"

step "Create conda env from environment.yml"
ENV="$(unique_env pedsum)"
register_env "$ENV"
( cd "$WORK/pedsum" && conda env create -f environment.yml -n "$ENV" )
ok "env $ENV created (deps + pedigree-graph@v0.5.1)"

# `--version` is dual-purpose: it tests that the install works *and* that
# pedsum's hardcoded VERSION was synced to the tag. A mismatch on a fresh clone
# is therefore ambiguous, so the failure messages say which is which.
step "Check version (expect $EXPECT_VERSION)"
if ver="$( cd "$WORK/pedsum" && run_in_env "$ENV" -- python pedigree_summary.py --version )"; then
  if printf '%s' "$ver" | grep -q "$EXPECT_VERSION"; then
    ok "pedsum reports $EXPECT_VERSION ($ver)"
  else
    fail "version mismatch: expected $EXPECT_VERSION, got '$ver' (install OK — hardcoded VERSION/tag out of sync)"
  fi
else
  fail "pedsum --version failed to run (broken install)"
fi

step "Run documented CLI: validate + summarize"
if ( cd "$WORK/pedsum" && run_in_env "$ENV" -- \
      python pedigree_summary.py validate --in example_pedigree.tsv --out "$WORK/out/validate" ); then
  ok "validate ran"
else
  fail "validate failed"
fi
if ( cd "$WORK/pedsum" && run_in_env "$ENV" -- \
      python pedigree_summary.py summarize --in example_pedigree.tsv --out "$WORK/out/summary" ); then
  ok "summarize ran"
else
  fail "summarize failed"
fi

step "Assert CLI outputs"
assert_file "$WORK/out/validate/validate.log" "validate log"
assert_file "$WORK/out/summary/summary.yaml"  "summary yaml"

if [ "$FULL" -eq 1 ]; then
  step "Full test suite (python -m pytest tests/ -q)"
  if ( cd "$WORK/pedsum" && run_in_env "$ENV" -- python -m pytest tests/ -q ); then
    ok "pedsum pytest passed"
  else
    fail "pedsum pytest failed"
  fi
fi

# EXIT trap performs cleanup + prints PASS/FAIL summary and sets the exit code.
