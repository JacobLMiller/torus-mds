# SLURM deployment

Two decoupled phases (see the driver scripts in `verification_experiments/`
and `modules/experiment_runner.py`): **embed** (persists layouts, no
metrics) then **score** (computes metrics from the persisted layouts). No
Node.js/npm setup is needed anywhere -- the Chen `wrap_python` method is a
pure Python/numba port, not a subprocess call to Node.

Configured for the **TUM COMA cluster**: `--partition=compute` (CPU-only
workload -- no `--gres` requested, so it doesn't touch any of that
partition's GPUs, just their CPU cores; `develop` is an alternative if you
already have access, gated behind a service request), no `--account` (not
used on this cluster). Both scripts activate a conda environment named
`torus-mds` (see below) -- set `MINIFORGE_ROOT` if Miniforge isn't installed
at `$HOME/miniforge3`.

**Storage**: clone/place this repo under `~/nobackup/` (e.g.
`~/nobackup/torus-mds`), not directly in `$HOME`. `layouts/` and
`data/suitesparse_cache/` will accumulate a lot of small files across
thousands of graphs -- exactly what `nobackup` is for. Copy the final
(small) `results/*.csv` files into your regular (backed-up) home directory
once merged, since `nobackup` isn't included in the off-site backup.
Per-node local `scratch` is NOT usable here -- it isn't shared across
nodes, and array tasks run on different physical nodes, so all of `layouts/`,
`data/`, and `results/` need to live on the shared `/storage` filesystem.
**This applies to the Miniforge install below too** -- installing it under
`~/scratch/miniforge3` puts it on whichever node's local disk you happened
to run the installer on (usually the login node), which is invisible to
every other node. Install it under `$HOME` or `~/nobackup` instead.

## -1. Install Miniforge and the environment (once, login node)

```bash
curl -L -o Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"

cd ~/torus-mds   # repo root
conda env create -f environment.yml
conda activate torus-mds
python -c "import s_gd2, networkx, numba, ssgetpy; print('deps OK')"
```

`s_gd2` and `ssgetpy` are both on conda-forge, so this is a pure-conda
environment -- no separate pip/build-toolchain step needed. The `numpy<2`
pin in `environment.yml` matters: `s_gd2`'s compiled extension is built
against NumPy 1.x and fails to import under NumPy 2.x.

## 0. Stage SuiteSparse (once, needs internet)

This is the one step in the whole pipeline that touches the network, and
takes roughly an hour for `--limit 2000`. Don't try to background it in a
login-node shell -- this cluster kills a user's entire process tree when
their last session ends, and (per the COMA servicedesk docs) `tmux`/`nohup`
do **not** protect against that. Submit it as a real job instead, which
runs independently of your SSH session:

```bash
sbatch --export=ALL verification_experiments/slurm/stage_suitesparse.sbatch
```

Smoke-test with a small `--limit` first (`--export=ALL,LIMIT=20`) before
committing to the full 2000. It's resumable -- if the job hits its time
limit or gets killed partway, just resubmit the same command and it picks
up where it left off rather than re-downloading everything.

If it fails with connection errors, the `compute` partition may not have
internet access on this cluster after all; fall back to running it directly
on the login node instead, keeping your session open for the duration:

```bash
conda activate torus-mds
python verification_experiments/stage_suitesparse.py \
    --limit 2000 --n-min 100 --n-max 10000 \
    --output-dir data/suitesparse_cache
```

## 1. Submit embedding jobs

**Run every `sbatch` command below from the repo root.** The scripts derive
`REPO_ROOT` from `$SLURM_SUBMIT_DIR` (the directory `sbatch` was invoked
from, which Slurm always sets) -- submitting from anywhere else means the
job `cd`s to the wrong place and every relative path inside it breaks.

One `sbatch` submission per (family, size tier), each with `--array=0-2`
(3 shards). Example for SBM across three log-spaced size tiers.
`GRAPHS_PER_SHARD=112`: 3 shards/tier x 3 tiers x 112 ~= 1000 graphs total
for the family -- since each `--array=0-2` submission already multiplies
`GRAPHS_PER_SHARD` by its 3 shards, don't reuse a single-tier per-shard
count across tiers without dividing by the tier count first.

The `--time` values below are calibrated from real (not extrapolated)
per-graph timings: `wrap_python` (capped at `--wrap-python-max-n`, default
3500) costs ~96s at worst, `TorusMDS` stays under ~4s even at n=10,000, but
**`s_gd2` is not uniformly cheap** -- it jumps from ~5s at n=3000 to ~68s at
n=10,000, and dominates the largest tier. These were measured on one
machine; treat them as a starting point, not a guarantee -- after your
first real submission, check the actual `t_embed` column in each shard's
`runs.csv` and adjust `--time` for later submissions if your cluster's
per-core speed differs.

```bash
cd ~/nobackup/torus-mds   # repo root -- see note above
mkdir -p slurm_logs

sbatch --array=0-2 --time=00:20:00 --mem=2G \
    --export=ALL,FAMILY=sbm,N_MIN=100,N_MAX=1000,GRAPHS_PER_SHARD=112 \
    verification_experiments/slurm/run_embed_array.sbatch

sbatch --array=0-2 --time=01:30:00 --mem=4G \
    --export=ALL,FAMILY=sbm,N_MIN=1000,N_MAX=3000,GRAPHS_PER_SHARD=112 \
    verification_experiments/slurm/run_embed_array.sbatch

sbatch --array=0-2 --time=03:00:00 --mem=8G \
    --export=ALL,FAMILY=sbm,N_MIN=3000,N_MAX=10000,GRAPHS_PER_SHARD=112 \
    verification_experiments/slurm/run_embed_array.sbatch
```

Repeat with `FAMILY=grg`. For SuiteSparse (a fixed pre-staged pool, sharded
by interleaving rather than by size tier since it spans the whole range at
once):

```bash
sbatch --array=0-19 --time=01:00:00 --mem=4G \
    --export=ALL,FAMILY=suitesparse,NUM_SHARDS=20 \
    verification_experiments/slurm/run_embed_array.sbatch
```

Each array task writes its own `layouts/<family>/.../shard_<i>/` directory
(graphs + coords + `runs.csv`) and is independently checkpointed/resumable
-- a killed or requeued task picks up where it left off.

## 2. Submit metrics jobs (after the embed jobs finish)

Count how many shard directories exist for a family, then submit an array
of that size (or use `--dependency=afterok:<embed_job_id>` to chain it
automatically after step 1):

```bash
find layouts/sbm -mindepth 1 -name runs.csv | wc -l   # -> N

sbatch --dependency=afterok:<embed_job_id> --array=0-$((N-1)) \
    --time=00:40:00 --mem=2G --export=ALL,FAMILY=sbm \
    verification_experiments/slurm/run_metrics_array.sbatch
```

Repeat for `grg` and `suitesparse`. This phase used to have a much worse
bottleneck than APSP: `estimate_alpha` (the torus scale-factor fit) is an
unvectorized O(k^2) Python loop, and it was being run on the *full*
unsampled layout instead of the same subsample used for the other metrics
-- at n=10,000 that meant ~50M Python-level calls per graph, easily blowing
past any reasonable time limit. That's fixed now (subsampling happens
before alpha estimation, not after), so the phase is genuinely cheap again
-- dominated by sparse APSP, measured at ~21 minutes worst-case for a
112-graph shard at the largest tier, low seconds for the smaller tiers.
`--time=00:40:00` above has margin over that.

## 3. Merge

```bash
python verification_experiments/slurm/merge_results.py \
    --glob "results/sbm_*_comparison.csv" --output results/sbm_comparison.csv

python verification_experiments/slurm/merge_results.py \
    --glob "results/grg_*_comparison.csv" --output results/grg_comparison.csv

python verification_experiments/slurm/merge_results.py \
    --glob "results/suitesparse_*_comparison.csv" --output results/suitesparse_comparison.csv
```

`results/sbm_comparison.csv` and `results/grg_comparison.csv` land in the
same schema the original all-in-one scripts produced, so
`sbm_results.ipynb` / `grg_results.ipynb` work unmodified.
