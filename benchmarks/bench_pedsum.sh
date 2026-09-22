#!/usr/bin/env bash
# Benchmark `pedsum summarize` on a simACE pedigree.
#
# Runs from a simACE umbrella checkout (this repo at external/pedsum). Simulate
# the pedigree first: copy simace_scenarios.yaml to simACE's config/bench_pedsum.yaml
# and run `pixi run snakemake --cores 4 results/bench_pedsum/<scenario>/simulate.done`
# at the umbrella root.
#
# usage: benchmarks/bench_pedsum.sh <scenario> <cell> [extra pedsum args...]
#   scenario  name under <umbrella>/results/bench_pedsum/<scenario>/rep1/pedigree.full.parquet
#   cell      label for the run (e.g. default, nofne)
# Writes benchmarks/results/simace/<scenario>.<cell>.{time,log} and a one-line TSV row
# to benchmarks/results/simace/results.tsv (scenario, cell, rows, export_s, wall_s, user_s, sys_s, max_rss_mib).
# The TSV is exported once per scenario and reused by later cells; export_s is that
# one export's time, or NA when the TSV predates its .export_s sidecar.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)   # the simACE umbrella checkout
scenario=$1; cell=$2; shift 2
parquet=$root/results/bench_pedsum/$scenario/rep1/pedigree.full.parquet
tsv=$root/results/bench_pedsum/$scenario/rep1/pedigree.tsv
out=$root/results/bench_pedsum/$scenario/pedsum_$cell
bench=$here/results/simace; mkdir -p "$bench"

if [ ! -f "$tsv" ]; then
  t0=$(date +%s.%N)
  (cd "$root" && pixi run python -c "
import polars as pl
pl.scan_parquet('$parquet').select('id','sex','mother','father','generation').sink_csv('$tsv', separator='\t')
")
  echo "$(echo "$(date +%s.%N) - $t0" | bc)" > "$tsv.export_s"
fi
rows=$(( $(wc -l < "$tsv") - 1 ))
export_s=$(cat "$tsv.export_s" 2>/dev/null || echo NA)  # sidecar only exists if we did the export

rm -rf "$out"
/usr/bin/time -v -o "$bench/$scenario.$cell.time" \
  pixi run --manifest-path "$here/../pixi.toml" \
  python "$here/../pedigree_summary.py" summarize \
  --in "$tsv" --out "$out" -v "$@" > "$bench/$scenario.$cell.log" 2>&1
wall=$(grep "Elapsed (wall" "$bench/$scenario.$cell.time" | awk '{print $NF}')
user=$(grep "User time" "$bench/$scenario.$cell.time" | awk '{print $NF}')
sys=$(grep "System time" "$bench/$scenario.$cell.time" | awk '{print $NF}')
rss_kb=$(grep "Maximum resident" "$bench/$scenario.$cell.time" | awk '{print $NF}')
rss_mib=$(echo "scale=1; $rss_kb/1024" | bc)
[ -f "$bench/results.tsv" ] || printf 'scenario\tcell\trows\texport_s\twall\tuser_s\tsys_s\tmax_rss_mib\targs\n' > "$bench/results.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$scenario" "$cell" "$rows" "$export_s" "$wall" "$user" "$sys" "$rss_mib" "$*" | tee -a "$bench/results.tsv"
