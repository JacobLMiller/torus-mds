#!/bin/bash
#SBATCH --job-name=torus-embed
#SBATCH --partition=<FILL_IN>
#SBATCH --account=<FILL_IN>
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/embed_%x_%A_%a.out
#SBATCH --error=slurm_logs/embed_%x_%A_%a.err
#
# Embedding phase: one array task = one shard of one graph family. TorusMDS
# and s_gd2 are near-linear in n; wrap_python (Chen) is O(n^2 * max_iters)
# and is capped via --wrap-python-max-n, so it's safe to run the same
# template across all size tiers. No Node.js/npm setup is needed anywhere --
# wrap_python() is a pure Python/numba port of the Chen algorithm, it does
# not shell out to a Node subprocess.
#
# Submit once per (family, size tier), overriding --time / --mem / --array
# and passing family-specific settings via --export (command-line flags to
# `sbatch` take precedence over the #SBATCH defaults above). See
# verification_experiments/slurm/README.md for concrete example invocations
# and recommended --time/--mem per tier.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- environment setup: adjust to your cluster ---
# module load <python module>
# source activate torus-mds

FAMILY="${FAMILY:?Set FAMILY=sbm|grg|suitesparse via --export}"
SHARD="${SLURM_ARRAY_TASK_ID}"
TORUS_MAX_ITERS="${TORUS_MAX_ITERS:-2000}"
WRAP_PYTHON_MAX_N="${WRAP_PYTHON_MAX_N:-3500}"
SEED_BASE="${SEED_BASE:-0}"

case "$FAMILY" in
  sbm)
    N_MIN="${N_MIN:?Set N_MIN via --export}"
    N_MAX="${N_MAX:?Set N_MAX via --export}"
    GRAPHS_PER_SHARD="${GRAPHS_PER_SHARD:?Set GRAPHS_PER_SHARD via --export}"
    python verification_experiments/sbm_comparison.py \
      --n-graphs "$GRAPHS_PER_SHARD" \
      --n-min "$N_MIN" --n-max "$N_MAX" \
      --torus-max-iters "$TORUS_MAX_ITERS" \
      --wrap-python-max-n "$WRAP_PYTHON_MAX_N" \
      --output-dir "layouts/sbm/${N_MIN}_${N_MAX}/shard_${SHARD}" \
      --seed "$((SEED_BASE + SHARD))"
    ;;
  grg)
    N_MIN="${N_MIN:?Set N_MIN via --export}"
    N_MAX="${N_MAX:?Set N_MAX via --export}"
    GRAPHS_PER_SHARD="${GRAPHS_PER_SHARD:?Set GRAPHS_PER_SHARD via --export}"
    python verification_experiments/grg_comparison.py \
      --n-graphs "$GRAPHS_PER_SHARD" \
      --n-min "$N_MIN" --n-max "$N_MAX" \
      --torus-max-iters "$TORUS_MAX_ITERS" \
      --wrap-python-max-n "$WRAP_PYTHON_MAX_N" \
      --output-dir "layouts/grg/${N_MIN}_${N_MAX}/shard_${SHARD}" \
      --seed "$((SEED_BASE + SHARD))"
    ;;
  suitesparse)
    NUM_SHARDS="${NUM_SHARDS:?Set NUM_SHARDS via --export (must match --array size)}"
    python verification_experiments/suitesparse_comparison.py \
      --cache-dir "${SUITESPARSE_CACHE_DIR:-data/suitesparse_cache}" \
      --shard-index "$SHARD" --num-shards "$NUM_SHARDS" \
      --torus-max-iters "$TORUS_MAX_ITERS" \
      --wrap-python-max-n "$WRAP_PYTHON_MAX_N" \
      --output-dir "layouts/suitesparse/shard_${SHARD}"
    ;;
  *)
    echo "Unknown FAMILY=$FAMILY (expected sbm|grg|suitesparse)" >&2
    exit 1
    ;;
esac
