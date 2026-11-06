#!/bin/bash
# Queue both metric phases for the *_normalized layouts (sbm, grg, suitesparse):
#
#   old metrics  run_metrics_array.sbatch -> compute_metrics.py, resuming into
#                the existing results/<family>_..._comparison.csv shards, so only
#                (exp_idx, method) pairs not scored yet are computed; existing
#                rows are carried over unchanged.
#   new metrics  evaluate_layouts_array.sbatch -> evaluate_layouts.py (crossings,
#                incidence angle, node resolution, cluster metrics, ...) for every
#                layout, into a separate NEW_RESULTS_ROOT.
#
# Before submitting it checks the code version, the Python env, that no embed
# jobs are running, and runs preflight_metrics.py (layouts complete/consistent,
# no stale scored rows); then it copies results/*.csv to results_backup_<stamp>/.
#
# Run on the login node from the repo root:
#   bash experiments/slurm/submit_metrics.sh            # check + submit
#   DRY_RUN=1 bash experiments/slurm/submit_metrics.sh  # check, print sbatch commands only

set -euo pipefail

LAYOUTS_ROOT="${LAYOUTS_ROOT:-layouts}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
NEW_RESULTS_ROOT="${NEW_RESULTS_ROOT:-results/layout_metrics}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-$HOME/nobackup/miniforge3}"
PYTHON_BIN="${PYTHON_BIN:-$MINIFORGE_ROOT/envs/torus-mds/bin/python}"
SLURM_DIR=experiments/slurm
FAMILIES=(sbm grg suitesparse)

fail() { echo "ABORT: $*" >&2; exit 1; }
run() { if [ -n "${DRY_RUN:-}" ]; then echo "[dry run] $*" >&2; echo DRYRUN; else "$@"; fi; }

# --- 1. code version (the cluster checkout must match this commit) ---
for f in experiments/compute_metrics.py $SLURM_DIR/run_metrics_array.sbatch \
         $SLURM_DIR/evaluate_layouts_array.sbatch $SLURM_DIR/preflight_metrics.py \
         verification_experiments/evaluate_layouts/evaluate_layouts.py; do
  [ -f "$f" ] || fail "$f not found -- run from the repo root of an up-to-date checkout (git pull)"
done
grep -q "theta_fit" experiments/compute_metrics.py \
  || fail "compute_metrics.py predates the theta_fit fix; parallelogram layouts would be scored with the wrong metric"
grep -q "python experiments/compute_metrics.py" $SLURM_DIR/run_metrics_array.sbatch \
  || fail "run_metrics_array.sbatch does not call experiments/compute_metrics.py"
grep -q "def node_resolution" verification_experiments/evaluate_layouts/layout_metrics.py \
  || fail "evaluator has no node_resolution -- pull the latest code"

# --- 2. Python env (used by the evaluator directly; run_metrics_array activates conda itself) ---
[ -x "$PYTHON_BIN" ] || fail "no Python at $PYTHON_BIN (set PYTHON_BIN or MINIFORGE_ROOT)"
"$PYTHON_BIN" -c "import modules, verification_experiments.evaluate_layouts.evaluate_layouts" \
  || fail "the torus-mds env cannot import the metric code"

# --- 3. nothing may be writing layouts while they are scored ---
if squeue -u "$USER" -h -o "%j" | grep -qiE "embed|fill"; then
  fail "embedding jobs are still queued/running (squeue -u $USER)"
fi

# --- 4. layouts complete and consistent, no stale scored rows ---
"$PYTHON_BIN" $SLURM_DIR/preflight_metrics.py --layouts-root "$LAYOUTS_ROOT" --results-root "$RESULTS_ROOT" \
  || fail "preflight failed -- see errors above"

# The evaluator only resumes CSVs with its current schema; never mix with older outputs.
if compgen -G "$NEW_RESULTS_ROOT/*/*.csv" > /dev/null; then
  if grep -L "node_resolution" "$NEW_RESULTS_ROOT"/*/*.csv | grep -q .; then
    fail "$NEW_RESULTS_ROOT has evaluator CSVs without node_resolution; choose another NEW_RESULTS_ROOT"
  fi
  echo "Note: $NEW_RESULTS_ROOT already has evaluator CSVs; the evaluator will resume them."
fi

# --- 5. back up existing results (small CSVs) ---
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="${RESULTS_ROOT}_backup_${STAMP}"
if [ -z "${DRY_RUN:-}" ]; then
  mkdir -p "$BACKUP"
  cp -p "$RESULTS_ROOT"/*.csv "$BACKUP"/
  echo "Backed up $(ls "$BACKUP" | wc -l) CSVs from $RESULTS_ROOT/ to $BACKUP/"
fi

# --- 6. submit ---
mkdir -p slurm_logs
JOBS=()
for fam in "${FAMILIES[@]}"; do
  n=$(find "$LAYOUTS_ROOT/${fam}_normalized" -mindepth 1 -name runs.csv | wc -l)
  id=$(run sbatch --parsable --job-name="metrics-old-$fam" --array=0-$((n - 1))%8 --time=01:00:00 --mem=4G \
        --export=ALL,FAMILY=$fam,FAMILY_SUBDIR=${fam}_normalized,LAYOUTS_ROOT=$LAYOUTS_ROOT,RESULTS_ROOT=$RESULTS_ROOT \
        $SLURM_DIR/run_metrics_array.sbatch)
  JOBS+=("$id")
  echo "old metrics, $fam: $n shards -> job $id"
done
id=$(run sbatch --parsable --job-name=metrics-new --partition=compute \
      --export=ALL,LAYOUTS_ROOT=$LAYOUTS_ROOT,RESULTS_ROOT=$NEW_RESULTS_ROOT,PYTHON_BIN=$PYTHON_BIN \
      $SLURM_DIR/evaluate_layouts_array.sbatch)
JOBS+=("$id")
echo "new metrics (6 sbm/grg tiers + 20 suitesparse shards) -> job $id"

echo
echo "Monitor:  squeue -u $USER;  sacct -j $(IFS=,; echo "${JOBS[*]}") --format=JobID,JobName,State,Elapsed"
echo "Old-metric rows go into $RESULTS_ROOT/*_comparison.csv; new-metric CSVs into $NEW_RESULTS_ROOT/<family>_normalized/."
