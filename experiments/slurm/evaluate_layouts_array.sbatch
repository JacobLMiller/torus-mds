#!/bin/bash
#SBATCH --job-name=evaluate-layouts
#SBATCH --array=0-25%4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# One output per task, safe to resume. May also be submitted directly via sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
LAYOUTS_ROOT="${LAYOUTS_ROOT:-verification_experiments/layouts}"
RESULTS_ROOT="${RESULTS_ROOT:-verification_experiments/results/layout_metrics}"
threads="${SLURM_CPUS_PER_TASK:-4}"

if [[ ! "$task_id" =~ ^([0-9]|1[0-9]|2[0-5])$ ]]; then
    echo "Expected array index 0-25, got $task_id" >&2; exit 2
fi
tiers=(100_1000 1000_3000 3000_10000)
if (( task_id < 6 )); then
    family=sbm
    if (( task_id >= 3 )); then family=grg; fi
    name="${tiers[$((task_id % 3))]}"
    directory="$LAYOUTS_ROOT/${family}_normalized/$name"
    if [[ ! -d "$directory" ]]; then
        echo "Missing layout tier: $directory" >&2; exit 2
    fi
    input_args=(--layouts-root "$directory")
else
    family=suitesparse
    name="shard_$((task_id - 6))"
    directory="$LAYOUTS_ROOT/suitesparse_normalized/$name"
    if [[ ! -f "$directory/runs.csv" ]]; then
        echo "Missing shard manifest: $directory/runs.csv" >&2; exit 2
    fi
    input_args=(--layouts-dir "$directory")
fi

# Match the existing cluster scripts; override VENV or supply PYTHON_BIN.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    source "${VENV:-/storage/home/brand/RLGD/SmartGD/venv2}/bin/activate"
    PYTHON_BIN=python
fi
export NUMBA_NUM_THREADS="$threads"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
output="$RESULTS_ROOT/${family}_normalized/${name}.csv"
mkdir -p "${output%/*}"
echo "Evaluating $family/$name with $threads threads -> $output"
"$PYTHON_BIN" verification_experiments/evaluate_layouts/evaluate_layouts.py \
    "${input_args[@]}" --output "$output" --threads "$threads" "$@"
