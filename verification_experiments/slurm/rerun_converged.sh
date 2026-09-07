#!/bin/bash
# Re-run the TorusMDS-family embeddings under the improved convergence settings
# (window_iters/chunk_epochs/alpha EMA/geometry-freeze fixes in modules/projector.py),
# without recomputing s_gd2/wrap_python and without touching the existing results.
#
# What this does, per family:
#   1. Copies the existing layouts/<family>_normalized tree to a new root
#      (NEW_LAYOUTS_ROOT) -- the ORIGINAL is left untouched.
#   2. Strips every TorusMDS* coords file and runs.csv row from the COPY only,
#      so run_embeddings' resume logic sees s_gd2/wrap_python as already done
#      (skips them) and TorusMDS-family methods as not-done (recomputes them
#      fresh, under the new convergence code).
#   3. Resubmits the embed job against the copy, reading graphs from the
#      already-staged --cache-dir pools (no regeneration).
#   4. Submits the metrics job with --dependency=afterok on the embed job(s),
#      so it starts automatically once they finish -- no need to babysit squeue.
#
# Submit from the repo root:
#   bash verification_experiments/slurm/rerun_converged.sh
#
# Override via environment before running, e.g.:
#   NEW_LAYOUTS_ROOT=layouts_converged2 bash verification_experiments/slurm/rerun_converged.sh
#
# METHODS/STRESS_MODE/TORUS_MAX_ITERS are deliberately NOT overridden here --
# run_embed_array.sbatch's own defaults already include TorusMDS_spectral_init
# and normalized stress (see its METHODS/STRESS_MODE lines), so this only needs
# to set what actually differs for this run: LAYOUTS_ROOT and CACHE_DIR.

set -euo pipefail

source "${MINIFORGE_ROOT:-$HOME/nobackup/miniforge3}/etc/profile.d/conda.sh"
conda activate torus-mds

OLD_LAYOUTS_ROOT="${OLD_LAYOUTS_ROOT:-layouts}"
NEW_LAYOUTS_ROOT="${NEW_LAYOUTS_ROOT:-layouts_converged}"
NEW_DRAWINGS_ROOT="${NEW_DRAWINGS_ROOT:-layout_drawings_converged}"
SBM_CACHE_ROOT="${SBM_CACHE_ROOT:-data/sbm_cache}"
GRG_CACHE_ROOT="${GRG_CACHE_ROOT:-data/grg_cache}"
SUITESPARSE_CACHE_DIR="${SUITESPARSE_CACHE_DIR:-data/suitesparse_cache}"
METRICS_TIME="${METRICS_TIME:-00:40:00}"
METRICS_MEM="${METRICS_MEM:-2G}"

mkdir -p slurm_logs

strip_torusmds() {
    local dir="$1"
    find "$dir" -type f -path "*/coords/*" -name "*TorusMDS*.npy" -delete
    find "$dir" -name runs.csv -print0 | xargs -0 -I{} python3 -c "
import pandas as pd
p = '{}'
df = pd.read_csv(p)
before = len(df)
df = df[~df['method'].str.startswith('TorusMDS')]
if len(df) != before:
    df.to_csv(p, index=False)
    print(f'{p}: removed {before - len(df)} TorusMDS rows')
"
}

copy_and_strip() {
    local src="$1" dst="$2"
    if [ ! -d "$src" ]; then
        echo "  (skip -- $src does not exist)"
        return 1
    fi
    mkdir -p "$(dirname "$dst")"
    cp -r "$src" "$dst"   # full copy, not hardlinks -- runs.csv gets rewritten in place below
    strip_torusmds "$dst"
    return 0
}

count_shards() {
    find "$1" -mindepth 1 -name runs.csv | wc -l
}

# ============================= SBM =============================
echo "=== SBM ==="
last_sbm_job=""
for tier in 100_1000 1000_3000 3000_10000; do
    src="${OLD_LAYOUTS_ROOT}/sbm_normalized/${tier}"
    dst="${NEW_LAYOUTS_ROOT}/sbm_normalized/${tier}"
    echo "-- tier $tier --"
    if ! copy_and_strip "$src" "$dst"; then continue; fi
    n_min="${tier%%_*}"; n_max="${tier##*_}"
    n_shards=$(find "$dst" -mindepth 1 -name runs.csv | wc -l)
    dep_args=(--job-name=torus-embed-sbm --dependency=singleton)
    job_id=$(sbatch --parsable "${dep_args[@]}" --array=0-$((n_shards - 1)) --time=02:00:00 --mem=4G \
        --export=ALL,FAMILY=sbm,N_MIN="$n_min",N_MAX="$n_max",LAYOUTS_ROOT="$NEW_LAYOUTS_ROOT",CACHE_DIR="${SBM_CACHE_ROOT}/${tier}",NUM_SHARDS="$n_shards" \
        verification_experiments/slurm/run_embed_array.sbatch)
    echo "  submitted embed job $job_id ($n_shards shards)"
    last_sbm_job="$job_id"
done
if [ -n "$last_sbm_job" ]; then
    n_total=$(count_shards "${NEW_LAYOUTS_ROOT}/sbm_normalized")
    metrics_job=$(sbatch --parsable --dependency=afterok:"$last_sbm_job" --array=0-$((n_total - 1)) \
        --time="$METRICS_TIME" --mem="$METRICS_MEM" \
        --export=ALL,FAMILY=sbm,FAMILY_SUBDIR=sbm_normalized,LAYOUTS_ROOT="$NEW_LAYOUTS_ROOT" \
        verification_experiments/slurm/run_metrics_array.sbatch)
    echo "  submitted metrics job $metrics_job (after $last_sbm_job, $n_total shards)"
fi

# ============================= GRG =============================
echo "=== GRG ==="
last_grg_job=""
for tier in 100_1000 1000_3000 3000_10000; do
    src="${OLD_LAYOUTS_ROOT}/grg_normalized/${tier}"
    dst="${NEW_LAYOUTS_ROOT}/grg_normalized/${tier}"
    echo "-- tier $tier --"
    if ! copy_and_strip "$src" "$dst"; then continue; fi
    n_min="${tier%%_*}"; n_max="${tier##*_}"
    n_shards=$(find "$dst" -mindepth 1 -name runs.csv | wc -l)
    job_id=$(sbatch --parsable --job-name=torus-embed-grg --dependency=singleton \
        --array=0-$((n_shards - 1)) --time=02:00:00 --mem=4G \
        --export=ALL,FAMILY=grg,N_MIN="$n_min",N_MAX="$n_max",LAYOUTS_ROOT="$NEW_LAYOUTS_ROOT",CACHE_DIR="${GRG_CACHE_ROOT}/${tier}",NUM_SHARDS="$n_shards" \
        verification_experiments/slurm/run_embed_array.sbatch)
    echo "  submitted embed job $job_id ($n_shards shards)"
    last_grg_job="$job_id"
done
if [ -n "$last_grg_job" ]; then
    n_total=$(count_shards "${NEW_LAYOUTS_ROOT}/grg_normalized")
    metrics_job=$(sbatch --parsable --dependency=afterok:"$last_grg_job" --array=0-$((n_total - 1)) \
        --time="$METRICS_TIME" --mem="$METRICS_MEM" \
        --export=ALL,FAMILY=grg,FAMILY_SUBDIR=grg_normalized,LAYOUTS_ROOT="$NEW_LAYOUTS_ROOT" \
        verification_experiments/slurm/run_metrics_array.sbatch)
    echo "  submitted metrics job $metrics_job (after $last_grg_job, $n_total shards)"
fi

# ========================= SuiteSparse =========================
echo "=== SuiteSparse ==="
src="${OLD_LAYOUTS_ROOT}/suitesparse_normalized"
dst="${NEW_LAYOUTS_ROOT}/suitesparse_normalized"
if copy_and_strip "$src" "$dst"; then
    n_shards=$(find "$dst" -mindepth 1 -name runs.csv | wc -l)
    embed_job=$(sbatch --parsable --job-name=torus-embed-ss --array=0-$((n_shards - 1))%6 \
        --time=02:00:00 --mem=4G \
        --export=ALL,FAMILY=suitesparse,LAYOUTS_ROOT="$NEW_LAYOUTS_ROOT",SUITESPARSE_CACHE_DIR="$SUITESPARSE_CACHE_DIR",NUM_SHARDS="$n_shards" \
        verification_experiments/slurm/run_embed_array.sbatch)
    echo "  submitted embed job $embed_job ($n_shards shards)"
    metrics_job=$(sbatch --parsable --dependency=afterok:"$embed_job" --array=0-$((n_shards - 1)) \
        --time="$METRICS_TIME" --mem="$METRICS_MEM" \
        --export=ALL,FAMILY=suitesparse,FAMILY_SUBDIR=suitesparse_normalized,LAYOUTS_ROOT="$NEW_LAYOUTS_ROOT" \
        verification_experiments/slurm/run_metrics_array.sbatch)
    echo "  submitted metrics job $metrics_job (after $embed_job, $n_shards shards)"
fi

# ============================= Drawings =========================
# draw_layouts_array.sbatch globs every runs.csv under the given LAYOUTS_ROOT
# in one combined, sorted list (it isn't scoped per-family), so this is
# submitted once against the whole NEW_LAYOUTS_ROOT tree, gated on ALL embed
# jobs above (sbm + grg + suitesparse, whichever ran) rather than per family.
echo "=== Drawings ==="
embed_jobs=()
[ -n "$last_sbm_job" ] && embed_jobs+=("$last_sbm_job")
[ -n "$last_grg_job" ] && embed_jobs+=("$last_grg_job")
[ -n "${embed_job:-}" ] && embed_jobs+=("$embed_job")

if [ "${#embed_jobs[@]}" -gt 0 ]; then
    dep="afterok:$(IFS=:; echo "${embed_jobs[*]}")"
    n_total=$(count_shards "$NEW_LAYOUTS_ROOT")
    # %2 throttles the array so at most 2 draw tasks run concurrently.
    draw_job=$(sbatch --parsable --dependency="$dep" --array=0-$((n_total - 1))%2 \
        --export=ALL,LAYOUTS_ROOT="$NEW_LAYOUTS_ROOT",OUTPUT_ROOT="$NEW_DRAWINGS_ROOT" \
        verification_experiments/slurm/draw_layouts_array.sbatch)
    echo "  submitted draw job $draw_job (after ${embed_jobs[*]}, $n_total shards, throttled to 2 at a time)"
else
    echo "  (skip -- no embed jobs were submitted)"
fi

echo
echo "Done. Track with: squeue -u \$USER"
echo "Once all metrics jobs finish, merge as usual (see slurm/README.md step 3),"
echo "globbing results/<family>_normalized_*_comparison.csv as before -- these"
echo "still land in results/, unaffected by LAYOUTS_ROOT."
echo "Drawings land under ${NEW_DRAWINGS_ROOT}/ once the draw job finishes."
