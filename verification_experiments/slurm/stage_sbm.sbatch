#!/bin/bash
#SBATCH --job-name=torus-stage-sbm
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --output=slurm_logs/stage_sbm_%j.out
#SBATCH --error=slurm_logs/stage_sbm_%j.err
#
# One-off staging job (not an array): pregenerates SBM graphs for one
# (N_MIN, N_MAX) size tier and caches them as graph_<idx>.npz + manifest.csv
# under OUTPUT_DIR. No network needed, unlike stage_suitesparse.sbatch.
#
# Submit once per size tier from the repo root, e.g.:
#   sbatch --export=ALL,N_MIN=100,N_MAX=1000,N_GRAPHS=336 \
#       verification_experiments/slurm/stage_sbm.sbatch
#
# Resumable: stage_sbm.py checkpoints its manifest, so if this hits the
# time limit or gets killed partway, just resubmit -- it picks up where it
# left off rather than regenerating already-staged graphs.
#
# Afterwards, point run_embed_array.sbatch at the cache via
# --export=ALL,FAMILY=sbm,N_MIN=...,N_MAX=...,CACHE_DIR=...,NUM_SHARDS=...
# instead of GRAPHS_PER_SHARD (see slurm/README.md).

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

source "${MINIFORGE_ROOT:-$HOME/nobackup/miniforge3}/etc/profile.d/conda.sh"
conda activate torus-mds

N_MIN="${N_MIN:?Set N_MIN via --export}"
N_MAX="${N_MAX:?Set N_MAX via --export}"
N_GRAPHS="${N_GRAPHS:?Set N_GRAPHS via --export}"

python verification_experiments/stage_sbm.py \
    --n-graphs "$N_GRAPHS" \
    --n-min "$N_MIN" --n-max "$N_MAX" \
    --k-min "${K_MIN:-3}" --k-max "${K_MAX:-8}" \
    --seed "${SEED:-0}" \
    --output-dir "${OUTPUT_DIR:-data/sbm_cache/${N_MIN}_${N_MAX}}"
