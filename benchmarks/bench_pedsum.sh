#!/usr/bin/env bash
# Benchmark `pedsum summarize` on a simACE pedigree.
#
# Runs from a simACE umbrella checkout (this repo at external/pedsum). Simulate
# the pedigree first: copy simace_scenarios.yaml to simACE's config/bench_pedsum.yaml
# and run `pixi run simace run <scenario> --until simulate` at the umbrella root.
#
# usage: benchmarks/bench_pedsum.sh <scenario> <cell> [extra pedsum args...]
#   scenario  name under <umbrella>/results/bench_pedsum/<scenario>/rep1/pedigree.parquet
#   cell      label for the run (e.g. default, deg2)
# Writes benchmarks/results/simace/<scenario>.<cell>.{time,log} and a one-line TSV row
# to benchmarks/results/simace/results.tsv (scenario, cell, rows, export_s, wall_s, user_s,
# sys_s, max_rss_mib, args). The TSV is exported once per scenario, and again when the
# parquet is newer, and reused by later cells; export_s is that one export's time.
# simACE codes sex 0=female, 1=male; the export recodes it to PLINK's 2/1 and writes
# pedigree.pedsum.tsv, so the 0/1-coded pedigree.tsv of earlier versions is never read.
# A results.tsv with another header (earlier versions logged wall as [h:]mm:ss) is
# refused; move it aside.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)   # the simACE umbrella checkout
scenario=$1; cell=$2; shift 2
parquet=$root/results/bench_pedsum/$scenario/rep1/pedigree.parquet
tsv=$root/results/bench_pedsum/$scenario/rep1/pedigree.pedsum.tsv
out=$root/results/bench_pedsum/$scenario/pedsum_$cell
bench=$here/results/simace; mkdir -p "$bench"
header=$'scenario\tcell\trows\texport_s\twall_s\tuser_s\tsys_s\tmax_rss_mib\targs'
if [ -f "$bench/results.tsv" ] && [ "$(head -n 1 "$bench/results.tsv")" != "$header" ]; then
  echo "error: $bench/results.tsv has another header; move it aside" >&2
  exit 1
fi

if [ ! -f "$tsv" ] || [ "$parquet" -nt "$tsv" ]; then
  t0=$(date +%s.%N)
  (cd "$root" && pixi run python -c "
import polars as pl
(
    pl.scan_parquet('$parquet')
    .select('id', pl.col('sex').replace_strict({0: 2, 1: 1}), 'mother', 'father', 'generation')
    .sink_csv('$tsv', separator='\t')
)
")
  echo "$(echo "$(date +%s.%N) - $t0" | bc)" > "$tsv.export_s"
fi
rows=$(( $(wc -l < "$tsv") - 1 ))
export_s=$(cat "$tsv.export_s")

rm -rf "$out"
/usr/bin/time -v -o "$bench/$scenario.$cell.time" \
  pixi run --manifest-path "$here/../pixi.toml" \
  python -m pedsum summarize \
  --in "$tsv" --out "$out" -v "$@" > "$bench/$scenario.$cell.log" 2>&1
# /usr/bin/time prints wall clock as [h:]mm:ss.ss; convert to seconds.
wall_s=$(grep "Elapsed (wall" "$bench/$scenario.$cell.time" | awk '{n = split($NF, t, ":"); s = 0; for (i = 1; i <= n; i++) s = s * 60 + t[i]; print s}')
user=$(grep "User time" "$bench/$scenario.$cell.time" | awk '{print $NF}')
sys=$(grep "System time" "$bench/$scenario.$cell.time" | awk '{print $NF}')
rss_kb=$(grep "Maximum resident" "$bench/$scenario.$cell.time" | awk '{print $NF}')
rss_mib=$(echo "scale=1; $rss_kb/1024" | bc)
[ -f "$bench/results.tsv" ] || printf '%s\n' "$header" > "$bench/results.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$scenario" "$cell" "$rows" "$export_s" "$wall_s" "$user" "$sys" "$rss_mib" "$*" | tee -a "$bench/results.tsv"
