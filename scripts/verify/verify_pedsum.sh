#!/usr/bin/env bash
# verify_pedsum.sh — fresh-computer install verification for pedsum (public, no auth).
#
# Mimics a brand-new user on a clean machine: clones pedsum from GitHub into a
# throwaway workdir, follows the README install (conda env from
# environment.yml, then `pip install -e .`), runs the documented CLI smoke
# (`pedsum --version`, `pedsum validate`, `pedsum summarize`, and
# `python -m pedsum`), asserts concrete outputs, then tears everything down.
# Reads no sibling repo or project files — only lib.sh beside it. Run from any
# directory; needs only `git` + `conda`.
#
# Usage:
#   bash verify_pedsum.sh [--pedsum-ref REF] [--pedsum-url URL] [--git-install] [--full] [--keep]
#
# Flags (env-var equivalents in parens):
#   --pedsum-ref REF   git ref to check out          (PEDSUM_REF; default: main)
#   --pedsum-url URL   clone URL                      (PEDSUM_URL; default: rwaples/pedsum)
#   --git-install      also `pip install git+URL@REF` into a bare Python env
#                      and run the smoke there (GIT_INSTALL=1; default: skip)
#   --full             also run the pedsum test suite (~80 s; default: skip)
#   --keep             keep the env + workdir on exit (for debugging)
#   -h, --help         show this help
#
# The expected version is the tag's when REF is a vX.Y.Z tag, else the
# checkout's pedsum/base.py VERSION. REF must have the `pedsum` command
# (pedsum#18, after v0.14.0); older refs exit 2 before the install.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

PEDSUM_URL="${PEDSUM_URL:-https://github.com/rwaples/pedsum.git}"
PEDSUM_REF="${PEDSUM_REF:-main}"
GIT_INSTALL="${GIT_INSTALL:-0}"
FULL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --pedsum-ref)  PEDSUM_REF="$2"; shift 2 ;;
    --pedsum-url)  PEDSUM_URL="$2"; shift 2 ;;
    --git-install) GIT_INSTALL=1; shift ;;
    --full)        FULL=1; shift ;;
    --keep)        KEEP=1; shift ;;
    -h|--help)     show_help "${BASH_SOURCE[0]}" ;;
    *) err "unknown argument: $1"; exit 2 ;;
  esac
done

# smoke <env> <label> — run the documented CLI from outside the clone, so the
# installed `pedsum` command is what runs, and assert its outputs.
smoke() {
  local env="$1" label="$2" ver out="$WORK/out/$2"
  mkdir -p "$out"

  # `--version` is dual-purpose: it tests that the install works *and* that
  # pedsum's hardcoded VERSION matches the expected one. A mismatch on a fresh
  # clone is therefore ambiguous, so the failure messages say which is which.
  if ver="$( cd "$out" && run_in_env "$env" -- pedsum --version )"; then
    if [ "$ver" = "pedsum $EXPECT_VERSION" ]; then
      ok "[$label] pedsum --version: $ver"
    else
      fail "[$label] version mismatch: expected 'pedsum $EXPECT_VERSION', got '$ver' (install OK — VERSION out of sync with $EXPECT_FROM)"
    fi
  else
    fail "[$label] pedsum --version failed to run (broken install or no pedsum command)"
  fi

  if ( cd "$out" && run_in_env "$env" -- python -m pedsum --help >/dev/null ); then
    ok "[$label] python -m pedsum --help ran"
  else
    fail "[$label] python -m pedsum --help failed"
  fi

  if ( cd "$out" && run_in_env "$env" -- \
        pedsum validate --in "$WORK/pedsum/example_pedigree.tsv" --out "$out/validate" ); then
    ok "[$label] pedsum validate ran"
  else
    fail "[$label] pedsum validate failed"
  fi
  if ( cd "$out" && run_in_env "$env" -- \
        pedsum summarize --in "$WORK/pedsum/example_pedigree.tsv" --out "$out/summary" ); then
    ok "[$label] pedsum summarize ran"
  else
    fail "[$label] pedsum summarize failed"
  fi

  assert_file "$out/validate/validate.log" "[$label] validate log"
  assert_file "$out/summary/summary.yaml"  "[$label] summary yaml"
}

step "Preflight"
preflight_tools

make_workdir

step "Clone pedsum ($PEDSUM_REF)"
clone "$PEDSUM_URL" "$PEDSUM_REF" "$WORK/pedsum"
ok "cloned pedsum"

if [[ "$PEDSUM_REF" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
  EXPECT_VERSION="${BASH_REMATCH[1]}"
  EXPECT_FROM="tag $PEDSUM_REF"
else
  EXPECT_VERSION="$(sed -n 's/^VERSION = "\(.*\)"$/\1/p' "$WORK/pedsum/pedsum/base.py")"
  EXPECT_FROM="pedsum/base.py"
  if [ -z "$EXPECT_VERSION" ]; then
    err "no VERSION line in pedsum/base.py at $PEDSUM_REF"
    exit 1
  fi
fi
log "expect version $EXPECT_VERSION (from $EXPECT_FROM)"

if [ ! -f "$WORK/pedsum/pedsum/__main__.py" ]; then
  err "$PEDSUM_REF has no pedsum command (added in pedsum#18, after v0.14.0); this script cannot verify it"
  exit 2
fi

step "Install as the README does: conda env from environment.yml, then pip install -e ."
ENV="$(unique_env pedsum)"
register_env "$ENV"
( cd "$WORK/pedsum" && conda env create -f environment.yml -n "$ENV" )
( cd "$WORK/pedsum" && run_in_env "$ENV" -- python -m pip install -e . )
pg_ver="$(run_in_env "$ENV" -- python -c "import importlib.metadata as m; print(m.version('pedigree-graph'))")"
ok "env $ENV created, pedsum installed (pedigree-graph $pg_ver)"

step "Run documented CLI (editable install)"
smoke "$ENV" editable

if [ "$GIT_INSTALL" -eq 1 ]; then
  step "Install straight from git into a bare Python env"
  GIT_ENV="$(unique_env pedsum_git)"
  register_env "$GIT_ENV"
  # The python spec from environment.yml, minus any trailing comment.
  py_spec="$(sed -n 's/^ *- \(python[<>=][^ #]*\).*$/\1/p' "$WORK/pedsum/environment.yml")"
  if [ -z "$py_spec" ]; then
    err "no '- python<op>...' line in environment.yml; cannot pin the bare env's Python"
    exit 1
  fi
  # pip wants a URL: scp-style git@host:path becomes ssh://, a bare path file://.
  case "$PEDSUM_URL" in
    *://*)    pip_url="$PEDSUM_URL" ;;
    *@*:*)    pip_url="ssh://${PEDSUM_URL/:/\/}" ;;
    *)        pip_url="file://$(cd "$PEDSUM_URL" && pwd)" ;;
  esac
  conda create --yes --quiet -n "$GIT_ENV" -c conda-forge "$py_spec" pip
  run_in_env "$GIT_ENV" -- python -m pip install "git+${pip_url}@${PEDSUM_REF}"
  ok "env $GIT_ENV: pip install git+${pip_url}@${PEDSUM_REF}"

  step "Run documented CLI (git install)"
  smoke "$GIT_ENV" git
fi

if [ "$FULL" -eq 1 ]; then
  step "Full test suite (python -m pytest tests/ -q)"
  if ( cd "$WORK/pedsum" && run_in_env "$ENV" -- python -m pytest tests/ -q ); then
    ok "pedsum pytest passed"
  else
    fail "pedsum pytest failed"
  fi
fi

# EXIT trap performs cleanup + prints PASS/FAIL summary and sets the exit code.
