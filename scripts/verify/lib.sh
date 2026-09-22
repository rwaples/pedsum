#!/usr/bin/env bash
# shared helpers for the fresh-computer install-verification scripts
# (scripts/verify/verify_*.sh).
#
# Design contract (see scripts/verify/README.md):
#   - One accumulating EXIT trap installed ONCE here; helpers append to
#     CLEANUP_ENVS / CLEANUP_DIRS, they never re-`trap`.
#   - register_env is called BEFORE `conda env create`, so a failed/partial
#     solve is still torn down.
#   - The EXIT handler captures $? BEFORE cleanup and always prints a summary,
#     so a `set -e` abort mid-script still reports — and is labelled
#     "ABORTED (rc=N)" rather than a misleading "0 failures".
#   - Sourced by each script *beside it*; the scripts read no sibling repo or
#     project files at runtime, only this helper.
#
# Vocabulary:
#   - Hard infra (require_cmd / clone / `conda env create`) is run bare: a
#     failure aborts via `set -e` and surfaces as "ABORTED (rc=N)".
#   - Functional checks (versions, pytest, snakemake, output asserts) are
#     wrapped in if/else and routed through ok()/fail(): they are *counted*,
#     not fatal, so the summary reports how many checks passed/failed.

# ---------------------------------------------------------------------------
# State (initialised once at source time).
# ---------------------------------------------------------------------------
CLEANUP_ENVS=()
CLEANUP_DIRS=()
OKS=0
FAILS=0
KEEP="${KEEP:-0}"
WORK=""
SCRIPT_NAME="$(basename "${0:-verify}")"
_ENV_SEQ=0

if [ -t 1 ]; then
  C_GRN=$'\033[32m'; C_RED=$'\033[31m'; C_YEL=$'\033[33m'; C_BLD=$'\033[1m'; C_RST=$'\033[0m'
else
  C_GRN=""; C_RED=""; C_YEL=""; C_BLD=""; C_RST=""
fi

# ---------------------------------------------------------------------------
# Reporting primitives.
# ---------------------------------------------------------------------------
step() { echo; echo "${C_BLD}>>> $*${C_RST}"; }
ok()   { OKS=$((OKS + 1));   echo "  ${C_GRN}[OK]${C_RST}   $*"; }
fail() { FAILS=$((FAILS + 1)); echo "  ${C_RED}[FAIL]${C_RST} $*" >&2; }
warn() { echo "  ${C_YEL}[WARN]${C_RST} $*" >&2; }
err()  { echo "  ${C_RED}[ERROR]${C_RST} $*" >&2; }
log()  { echo "  $*"; }

# ---------------------------------------------------------------------------
# Cleanup registration + the single EXIT handler.
# ---------------------------------------------------------------------------
register_env() { CLEANUP_ENVS+=("$1"); }
register_dir() { CLEANUP_DIRS+=("$1"); }

_print_summary() {
  local rc="${1:-0}"
  echo
  echo "================ SUMMARY: ${SCRIPT_NAME} ================"
  if [ "$rc" -ne 0 ] && [ "$FAILS" -eq 0 ]; then
    echo "${C_RED}ABORTED (rc=${rc})${C_RST} — exited before a check failed (infra/setup error: clone, env solve, missing prereq, or bad --*-ref)."
  elif [ "$FAILS" -gt 0 ]; then
    echo "${C_RED}FAIL${C_RST} — ${FAILS} check(s) failed, ${OKS} passed."
  else
    echo "${C_GRN}PASS${C_RST} — all ${OKS} check(s) passed."
  fi
  echo "========================================================"
}

_on_exit() {
  local rc=$?                          # capture BEFORE cleanup
  if [ "${KEEP:-0}" != "1" ]; then
    if [ "${#CLEANUP_ENVS[@]}" -gt 0 ]; then
      for e in "${CLEANUP_ENVS[@]}"; do
        conda env remove -n "$e" -y >/dev/null 2>&1 || true
      done
    fi
    if [ "${#CLEANUP_DIRS[@]}" -gt 0 ]; then
      for d in "${CLEANUP_DIRS[@]}"; do
        rm -rf "$d" || true
      done
    fi
  else
    if [ "${#CLEANUP_ENVS[@]}" -gt 0 ] || [ "${#CLEANUP_DIRS[@]}" -gt 0 ]; then
      warn "--keep set: retaining envs [${CLEANUP_ENVS[*]:-}] and dirs [${CLEANUP_DIRS[*]:-}]"
    fi
  fi
  _print_summary "$rc"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
  if [ "$FAILS" -gt 0 ]; then exit 1; fi
  exit 0
}
trap _on_exit EXIT

# ---------------------------------------------------------------------------
# Preflight.
# ---------------------------------------------------------------------------
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "required command not found on PATH: $1"; exit 1; }
}

# The base toolchain every script needs.
preflight_tools() {
  require_cmd git
  require_cmd conda
  ok "git + conda present"
}

# Print a script's own header comment block (line 2 .. first blank) as --help,
# then exit cleanly — clears the EXIT trap so no PASS/FAIL summary is appended.
# Pass the caller's "${BASH_SOURCE[0]}" so the right file's header is shown.
show_help() {
  sed -n '2,/^$/p' "$1" | sed 's/^# \{0,1\}//'
  trap - EXIT
  exit 0
}

# SSH preflight: GitHub's `ssh -T` exits non-zero even on success, so gate on
# the message, not the exit code. Capture-then-grep (NOT a pipe) so `pipefail`
# doesn't fold ssh's non-zero exit into the result. Soft: warn and proceed —
# the actual private `git clone` is the real gate.
ssh_preflight() {
  local out
  out="$(ssh -T -o BatchMode=yes -o ConnectTimeout=10 git@github.com 2>&1 || true)"
  if printf '%s' "$out" | grep -q "successfully authenticated"; then
    ok "GitHub SSH authenticated"
  else
    warn "GitHub SSH check inconclusive — private clones (fitACE, fitACE_epimight) will be the real gate."
  fi
}

# ---------------------------------------------------------------------------
# Workdir + envs + clone.
# ---------------------------------------------------------------------------

# Sets global WORK to a fresh throwaway dir and registers it for cleanup.
# (Assigns a global rather than echoing, so registration is not lost in a
# command-substitution subshell.)
make_workdir() {
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/simace_verify.XXXXXX")"
  register_dir "$WORK"
  log "workdir: $WORK"
}

# Unique, _verify_-tagged env name (so leftovers are greppable:
# `conda env list | grep _verify_`).
unique_env() {
  _ENV_SEQ=$((_ENV_SEQ + 1))
  printf '%s_verify_%s_%s' "$1" "$$" "$_ENV_SEQ"
}

# clone <url> <ref> <dest> [auth_hint]
# Full clone (no --depth) so setuptools-scm derives the CalVer version from
# tags; a shallow clone yields a bogus version and trips simace>= floors.
# On failure prints the optional auth_hint (used to name SSH auth as the likely
# cause for the first private repo) and returns non-zero → caller aborts.
clone() {
  local url="$1" ref="$2" dest="$3" hint="${4:-}" rc=0
  log "clone $url ($ref) -> $dest"
  # Capture git's real exit code via `|| rc=$?` (an `if ! git clone` would make
  # $? inside the branch 0, not git's code) while keeping `set -e` happy.
  git clone "$url" "$dest" || rc=$?
  if [ "$rc" -ne 0 ]; then
    [ -n "$hint" ] && err "$hint"
    err "git clone failed (exit $rc): $url"
    return "$rc"
  fi
  git -C "$dest" checkout --quiet "$ref" || {
    err "checkout failed: ref '$ref' not found in $url (bad --*-ref?)"
    return 1
  }
}

# run_in_env <env> -- <cmd...>   (output streams live; not captured)
run_in_env() {
  local env="$1"; shift
  [ "${1:-}" = "--" ] && shift
  conda run --no-capture-output -n "$env" "$@"
}

# ---------------------------------------------------------------------------
# Output assertions (counted via ok/fail, never fatal).
# ---------------------------------------------------------------------------

# assert_file <path> [description]  — exists AND non-empty.
assert_file() {
  local p="$1" desc="${2:-$1}"
  if [ -s "$p" ]; then ok "$desc (non-empty: $p)"; else fail "$desc missing or empty: $p"; fi
}

# assert_path <path> [description]  — exists (file, dir, or resolvable symlink).
assert_path() {
  local p="$1" desc="${2:-$1}"
  if [ -e "$p" ]; then ok "$desc ($p)"; else fail "$desc missing: $p"; fi
}
