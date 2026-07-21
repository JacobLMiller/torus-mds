#!/bin/bash
#SBATCH --job-name=torus-stage-suitesparse
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/stage_suitesparse_%j.out
#SBATCH --error=slurm_logs/stage_suitesparse_%j.err
#
# One-off staging job (not an array): downloads/filters/caches the
# SuiteSparse graphs. This is the one step in the whole pipeline that needs
# internet access -- submit it as a real sbatch job rather than trying to
# background it in a login-node shell. The login node kills ALL of a user's
# processes when their last session ends (systemd-logind session cleanup),
# and tmux/nohup do NOT protect against this on this cluster (see the COMA
# servicedesk docs) -- only an actual SLURM job runs under slurmd
# independent of your SSH session and survives you logging out.
#
# If this fails with connection errors, the compute partition may not have
# internet access on this cluster after all -- fall back to running
# `python verification_experiments/stage_suitesparse.py ...` directly on the
# login node instead, keeping your session open for the duration (expect
# roughly an hour for --limit 2000, extrapolated from local testing).
#
# Resumable: stage_suitesparse.py checkpoints its manifest, so if this hits
# the time limit or gets killed partway, just resubmit -- it picks up where
# it left off rather than re-downloading everything.
#
# Submit from the repo root, e.g.:
#   sbatch --export=ALL verification_experiments/slurm/stage_suitesparse.sbatch
# Override defaults via --export=ALL,LIMIT=20 (smoke test), N_MIN=..., etc.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

# --- environment setup ---
source "${MINIFORGE_ROOT:-$HOME/nobackup/miniforge3}/etc/profile.d/conda.sh"
conda activate torus-mds

python verification_experiments/stage_suitesparse.py \
    --limit "${LIMIT:-2000}" \
    --n-min "${N_MIN:-100}" \
    --n-max "${N_MAX:-10000}" \
    --output-dir "${OUTPUT_DIR:-data/suitesparse_cache}"
