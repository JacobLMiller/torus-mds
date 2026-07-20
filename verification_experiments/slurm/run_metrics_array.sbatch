#!/bin/bash
#SBATCH --job-name=torus-metrics
#SBATCH --partition=<FILL_IN>
#SBATCH --account=<FILL_IN>
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mem=2G
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/metrics_%x_%A_%a.out
#SBATCH --error=slurm_logs/metrics_%x_%A_%a.err
#
# Metrics phase: one array task = one embedding shard's layouts/ directory
# (discovered by scanning for runs.csv files), scored into its own
# results/<shard>_comparison.csv. Run AFTER the corresponding
# run_embed_array.sbatch submission has finished, e.g.:
#   sbatch --dependency=afterok:<embed_job_id> --export=FAMILY=sbm ... run_metrics_array.sbatch
# then concatenate everything with merge_results.py.
#
# Much cheaper than embedding: dominated by sparse APSP (modules/graphio.py
# apsp_distance_matrix), not the O(n^2) metrics loops the old all-in-one
# scripts used, so --time/--mem here can stay modest even for the largest
# graphs. Set --array to match the number of shard directories found (see
# README.md for the exact `find` command to count them).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- environment setup: adjust to your cluster ---
# module load <python module>
# source activate torus-mds

FAMILY="${FAMILY:?Set FAMILY=sbm|grg|suitesparse via --export}"
METRIC_SUBSAMPLE="${METRIC_SUBSAMPLE:-100}"
NP_RADIUS="${NP_RADIUS:-2.0}"

mkdir -p results

mapfile -t SHARD_DIRS < <(find "layouts/${FAMILY}" -mindepth 1 -name runs.csv -exec dirname {} \; | sort)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SHARD_DIRS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= ${#SHARD_DIRS[@]} shard directories found under layouts/${FAMILY}; nothing to do" >&2
  exit 0
fi

SHARD_DIR="${SHARD_DIRS[$SLURM_ARRAY_TASK_ID]}"
SHARD_NAME="$(echo "${FAMILY}_${SHARD_DIR#layouts/${FAMILY}/}" | tr '/' '_')"

python verification_experiments/compute_metrics.py \
  --layouts-dir "$SHARD_DIR" \
  --output "results/${SHARD_NAME}_comparison.csv" \
  --metric-subsample "$METRIC_SUBSAMPLE" \
  --np-radius "$NP_RADIUS"
