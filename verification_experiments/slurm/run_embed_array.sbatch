#!/bin/bash
#SBATCH --job-name=torus-embed
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/embed_%x_%A_%a.out
#SBATCH --error=slurm_logs/embed_%x_%A_%a.err
#
# --time=00:30:00 above is a placeholder, NOT a real budget -- you MUST
# override it per submission (see README.md for measured per-tier values,
# ranging ~20min to ~3hr). Embedding phase: one array task = one shard of
# one graph family.
#


set -euo pipefail

source "${MINIFORGE_ROOT:-$HOME/nobackup/miniforge3}/etc/profile.d/conda.sh"
conda activate torus-mds

FAMILY="${FAMILY:?Set FAMILY=sbm|grg|suitesparse via --export}"
SHARD="${SLURM_ARRAY_TASK_ID}"
TORUS_MAX_ITERS="${TORUS_MAX_ITERS:-2000}"
WRAP_PYTHON_MAX_N="${WRAP_PYTHON_MAX_N:-3500}"
SEED_BASE="${SEED_BASE:-0}"
STRESS_MODE="${STRESS_MODE:-raw}"
METHODS="${METHODS:-TorusMDS s_gd2 wrap_python}"

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
    # Runs that deviate from the (raw, 2000-iter) default land in a
    # distinctly-named layouts/ subdir so they don't collide with -- or get
    # skipped as already-done against -- manifests from a different
    # configuration on the same graphs (e.g. an iteration-count ablation).
    OUTPUT_SUBDIR="suitesparse"
    if [ "$STRESS_MODE" != "raw" ]; then
      OUTPUT_SUBDIR="${OUTPUT_SUBDIR}_${STRESS_MODE}"
    fi
    if [ "$TORUS_MAX_ITERS" != "2000" ]; then
      OUTPUT_SUBDIR="${OUTPUT_SUBDIR}_iters${TORUS_MAX_ITERS}"
    fi
    python verification_experiments/suitesparse_comparison.py \
      --cache-dir "${SUITESPARSE_CACHE_DIR:-data/suitesparse_cache}" \
      --shard-index "$SHARD" --num-shards "$NUM_SHARDS" \
      --torus-max-iters "$TORUS_MAX_ITERS" \
      --wrap-python-max-n "$WRAP_PYTHON_MAX_N" \
      --stress-mode "$STRESS_MODE" \
      --methods $METHODS \
      --output-dir "layouts/${OUTPUT_SUBDIR}/shard_${SHARD}"
    ;;
  *)
    echo "Unknown FAMILY=$FAMILY (expected sbm|grg|suitesparse)" >&2
    exit 1
    ;;
esac
