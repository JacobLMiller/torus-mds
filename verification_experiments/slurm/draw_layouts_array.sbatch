#!/bin/bash
#SBATCH --job-name=torus-draw
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/draw_%x_%A_%a.out
#SBATCH --error=slurm_logs/draw_%x_%A_%a.err
#
# --time/--array above are placeholders -- draw_layouts.py itself would
# otherwise walk and render every shard serially in one process (which is
# what got slow interactively); this splits that into one array task per
# shard directory instead. Size --array to the shard count below, and
# smoke-test one task's --time before submitting the rest.
#
# Count shard dirs first, then submit:
#   find "${LAYOUTS_ROOT:-layouts}" -mindepth 1 -name runs.csv | wc -l   # -> N
#   sbatch --array=0-$((N-1)) --export=ALL verification_experiments/slurm/draw_layouts_array.sbatch
#
# Optional --export=ALL,... overrides: LAYOUTS_ROOT (default "layouts"),
# OUTPUT_ROOT (default "layout_drawings"), METHODS (space-separated, default
# all methods present in each runs.csv), MAX_GRAPHS_PER_SHARD, DPI.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

source "${MINIFORGE_ROOT:-$HOME/nobackup/miniforge3}/etc/profile.d/conda.sh"
conda activate torus-mds

LAYOUTS_ROOT="${LAYOUTS_ROOT:-layouts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-layout_drawings}"
METHODS="${METHODS:-}"
MAX_GRAPHS_PER_SHARD="${MAX_GRAPHS_PER_SHARD:-}"
DPI="${DPI:-110}"

mapfile -t SHARD_DIRS < <(find "$LAYOUTS_ROOT" -mindepth 1 -name runs.csv -exec dirname {} \; | sort -V)

if [ "${#SHARD_DIRS[@]}" -eq 0 ]; then
  echo "No runs.csv found anywhere under $LAYOUTS_ROOT (resolved from REPO_ROOT=${REPO_ROOT}). Check LAYOUTS_ROOT and that this job was submitted from the repo root." >&2
  exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SHARD_DIRS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= ${#SHARD_DIRS[@]} shard directories found under $LAYOUTS_ROOT; nothing to do for this task" >&2
  exit 0
fi

SHARD_DIR="${SHARD_DIRS[$SLURM_ARRAY_TASK_ID]}"

EXTRA_ARGS=()
if [ -n "$METHODS" ]; then
  EXTRA_ARGS+=(--methods $METHODS)
fi
if [ -n "$MAX_GRAPHS_PER_SHARD" ]; then
  EXTRA_ARGS+=(--max-graphs-per-shard "$MAX_GRAPHS_PER_SHARD")
fi

python verification_experiments/draw_layouts.py \
  --layouts-root "$LAYOUTS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --only-shard-dir "$SHARD_DIR" \
  --dpi "$DPI" \
  "${EXTRA_ARGS[@]}"
