#!/bin/bash
#SBATCH --job-name=torus-metrics
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --time=00:40:00
#SBATCH --mem=2G
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/metrics_%x_%A_%a.out
#SBATCH --error=slurm_logs/metrics_%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"


source "${MINIFORGE_ROOT:-$HOME/nobackup/miniforge3}/etc/profile.d/conda.sh"
conda activate torus-mds

FAMILY="${FAMILY:?Set FAMILY=sbm|grg|suitesparse via --export}"
METRIC_SUBSAMPLE="${METRIC_SUBSAMPLE:-100}"
NP_RADIUS="${NP_RADIUS:-2.0}"

mkdir -p results

mapfile -t SHARD_DIRS < <(find "layouts/${FAMILY}" -mindepth 1 -name runs.csv -exec dirname {} \; | sort -V)

if [ "${#SHARD_DIRS[@]}" -eq 0 ]; then
  echo "No runs.csv found anywhere under layouts/${FAMILY} (resolved from REPO_ROOT=${REPO_ROOT}). Check: FAMILY is spelled right, this job was submitted from the repo root, and the corresponding run_embed_array.sbatch job(s) actually completed and wrote output there." >&2
  exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SHARD_DIRS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= ${#SHARD_DIRS[@]} shard directories found under layouts/${FAMILY}; nothing to do for this task" >&2
  exit 0
fi

SHARD_DIR="${SHARD_DIRS[$SLURM_ARRAY_TASK_ID]}"
SHARD_NAME="$(echo "${FAMILY}_${SHARD_DIR#layouts/${FAMILY}/}" | tr '/' '_')"

python verification_experiments/compute_metrics.py \
  --layouts-dir "$SHARD_DIR" \
  --output "results/${SHARD_NAME}_comparison.csv" \
  --metric-subsample "$METRIC_SUBSAMPLE" \
  --np-radius "$NP_RADIUS"
