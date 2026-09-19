#!/usr/bin/env bash
# Compare the serial suite with candidate pytest-xdist worker/thread splits.
#
# The default matrix runs three sweeps of serial, pinned n6 and n8, and
# unpinned n6. Each pytest process starts in its own process group so the
# sampler can report aggregate RSS across the controller, workers, and CLI
# subprocesses instead of the largest single process.
#
#   bash benchmarks/bench_pytest_workers.sh
#   SWEEPS=1 CELLS="n8t1" bash benchmarks/bench_pytest_workers.sh tests/test_pairs_properties.py
#
# Logs and results.tsv land in a fresh /tmp directory by default. Set OUT to a
# stable directory when comparing runs across revisions.
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1

SWEEPS=${SWEEPS:-3}
CELLS=${CELLS:-"serial n6t1 n8t1 n6_unpinned"}
WARMUP=${WARMUP:-1}
OUT=${OUT:-$(mktemp -d -t pedsum-pytest-bench.XXXXXX)}

THREAD_VARS="NUMBA_NUM_THREADS OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS POLARS_MAX_THREADS"
RESULTS=$OUT/results.tsv
RUN_ROWS=$(mktemp) || exit 1
trap 'rm -f "$RUN_ROWS"' EXIT

cell_row() {
  case $1 in
    serial) printf '%s\n' "- -" ;;
    n6t1) printf '%s\n' "6 1" ;;
    n8t1) printf '%s\n' "8 1" ;;
    n6_unpinned) printf '%s\n' "6 -" ;;
    *) return 1 ;;
  esac
}

build_env_args() {
  local threads=$1 variable
  env_args=()
  for variable in $THREAD_VARS; do
    if [ "$threads" = "-" ]; then
      env_args+=(-u "$variable")
    else
      env_args+=("$variable=$threads")
    fi
  done
}

median() {
  sort -n | awk '{ values[NR] = $1 } END { if (NR) print values[int((NR + 1) / 2)] }'
}

for cell in $CELLS; do
  if ! cell_row "$cell" >/dev/null; then
    printf "unknown cell '%s'; use serial, n6t1, n8t1, or n6_unpinned\n" "$cell" >&2
    exit 2
  fi
done

mkdir -p "$OUT"
printf 'cell\tsweep\tworkers\tthreads\twall_s\tpeak_tree_rss_mib\ttests\tstatus\n' > "$RESULTS"

if [ "$WARMUP" -eq 1 ]; then
  printf '[%s] WARMUP serial, unpinned, discarded\n' "$(date +%T)"
  build_env_args -
  env "${env_args[@]}" pixi run pytest "$@" > "$OUT/warmup.log" 2>&1
  warmup_status=$?
  if [ "$warmup_status" -ne 0 ]; then
    tail -40 "$OUT/warmup.log" >&2
    exit "$warmup_status"
  fi
fi

for sweep in $(seq 1 "$SWEEPS"); do
  for cell in $CELLS; do
    read -r workers threads <<< "$(cell_row "$cell")"
    build_env_args "$threads"
    xdist_args=()
    if [ "$workers" != "-" ]; then
      xdist_args=(-n "$workers" --dist worksteal)
    fi

    log=$OUT/$cell.$sweep.log
    printf '[%s] START %s sweep %s\n' "$(date +%T)" "$cell" "$sweep"
    start_ns=$(date +%s%N)
    setsid env "${env_args[@]}" pixi run pytest "${xdist_args[@]}" "$@" > "$log" 2>&1 &
    run_pid=$!
    peak_rss_kib=0
    while kill -0 "$run_pid" 2>/dev/null; do
      rss_kib=$(ps -eo pgid=,rss= | awk -v pg="$run_pid" '$1 == pg { total += $2 } END { print total + 0 }')
      if [ "$rss_kib" -gt "$peak_rss_kib" ]; then
        peak_rss_kib=$rss_kib
      fi
      sleep 0.1
    done
    wait "$run_pid"
    status=$?
    end_ns=$(date +%s%N)

    wall_s=$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.2f", (end - start) / 1000000000 }')
    peak_rss_mib=$(awk -v rss="$peak_rss_kib" 'BEGIN { printf "%.1f", rss / 1024 }')
    tests=$(grep -Eo '[0-9]+ (passed|failed|skipped|xfailed|xpassed|errors?)' "$log" |
      awk '{ total += $1 } END { print total + 0 }')
    if [ "$status" -eq 0 ]; then
      state=ok
    else
      state=failed:$status
    fi

    row=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "$cell" "$sweep" "$workers" "$threads" "$wall_s" "$peak_rss_mib" "$tests" "$state")
    printf '%s\n' "$row" >> "$RESULTS"
    printf '%s\n' "$row" >> "$RUN_ROWS"
    printf '[%s] DONE  %s sweep %s  wall=%ss tree_rss=%sMiB tests=%s %s\n' \
      "$(date +%T)" "$cell" "$sweep" "$wall_s" "$peak_rss_mib" "$tests" "$state"
  done
done

printf '\n%-14s %4s %27s %20s %7s\n' cell runs 'wall_s median (min-max)' 'tree_rss_mib median' tests
for cell in $CELLS; do
  rows=$(awk -F'\t' -v cell="$cell" '$1 == cell && $8 == "ok"' "$RUN_ROWS")
  runs=$(printf '%s\n' "$rows" | awk 'NF { count++ } END { print count + 0 }')
  if [ "$runs" -eq 0 ]; then
    printf '%-14s %4s %27s %20s %7s\n' "$cell" 0 - - -
    continue
  fi
  walls=$(printf '%s\n' "$rows" | cut -f5)
  rss_values=$(printf '%s\n' "$rows" | cut -f6)
  med_wall=$(printf '%s\n' "$walls" | median)
  min_wall=$(printf '%s\n' "$walls" | sort -n | head -1)
  max_wall=$(printf '%s\n' "$walls" | sort -n | tail -1)
  med_rss=$(printf '%s\n' "$rss_values" | median)
  tests=$(printf '%s\n' "$rows" | cut -f7 | sort -un | paste -sd/ -)
  wall_range=$(printf '%s (%s-%s)' "$med_wall" "$min_wall" "$max_wall")
  printf '%-14s %4s %27s %20s %7s\n' "$cell" "$runs" "$wall_range" "$med_rss" "$tests"
done

printf '\nResults: %s\n' "$RESULTS"
if awk -F'\t' '$8 != "ok" { found = 1 } END { exit !found }' "$RUN_ROWS"; then
  exit 1
fi
if [ "$(awk -F'\t' '$8 == "ok" { print $7 }' "$RUN_ROWS" | sort -u | wc -l)" -gt 1 ]; then
  printf 'Invalid comparison: successful cells ran different test counts.\n' >&2
  exit 1
fi
