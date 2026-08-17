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

Repeat with `FAMILY=grg`.

By default each family runs all three methods (`TorusMDS s_gd2 wrap_python`).
Restrict via `--export=ALL,...,METHODS="TorusMDS s_gd2"` (space-separated,
matches `run_embeddings`'s `methods=` choices) to skip `wrap_python` --
useful since it's the slowest method by far and the one most often excluded
from a quick comparison. For `FAMILY=grg`, `GRAPH_TYPE_WEIGHTS` (default
`1,1,1`, i.e. euclidean/toroidal/spherical in equal thirds) selects the mix
of GRG variants generated; `GRAPH_TYPE_WEIGHTS="0,1,0"` generates toroidal
GRGs only. **Because its value itself contains commas, it can't go inside a
`--export=ALL,KEY=val,KEY=val` list** -- `sbatch --export` splits on every
comma in the whole argument, including ones inside a value, since the
shell's quotes are already gone by the time SLURM parses it (a value like
`"0,1,0"` silently becomes `GRAPH_TYPE_WEIGHTS=0` plus two bogus `1`/`0`
entries, and `grg_comparison.py` then rejects it: `--graph-type-weights
must have exactly 3 comma-separated values`). Export it as a shell variable
instead and let plain `--export=ALL` forward it:

```bash
export GRAPH_TYPE_WEIGHTS="0,1,0"
sbatch --array=0-3 --time=05:00:00 --mem=4G \
    --export=ALL,FAMILY=grg,N_MIN=100,N_MAX=5000,GRAPHS_PER_SHARD=625,METHODS="TorusMDS s_gd2" \
    verification_experiments/slurm/run_embed_array.sbatch
```

(4 shards x 625 graphs = 2500. Fewer, longer-running shards for a small
cluster with limited concurrent job slots -- scale `GRAPHS_PER_SHARD` and
`--time` together if you change the shard count, keeping their product at
the total graphs you want for that family.)

For SuiteSparse (a fixed pre-staged pool, sharded
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

## 1b. Alpha-aspect init comparison (separate experiment)

Compares `learn_mode='alpha_aspect'` TorusMDS (jointly learns scale *and*
aspect ratio, instead of the fixed unit-square torus of the default
`learn_mode='alpha'` used above) against `s_gd2`/`wrap_python`, in four
configurations: {smart spectral init, random init} x {default (4096) batch
size, 100x (409600) batch size}. These are just four more `method` values
(`TorusMDS_smart_1x`, `TorusMDS_smart_100x`, `TorusMDS_random_1x`,
`TorusMDS_random_100x`, defined in `modules/experiment_runner.py`'s
`ASPECT_INIT_VARIANTS`) -- no new sbatch script needed, `run_embed_array.sbatch`
already forwards whatever `METHODS` you export.

**This experiment trains with `STRESS_MODE=normalized`** (minimizes
`sum((alpha*r-d)^2 / d^2)`, each pair weighted by its own `1/d^2`), not
section 1's `raw` default -- set the env var below, don't leave it unset.
Because it deviates from the `(raw, 2000-iter)` default, `run_embed_array.sbatch`
routes it to a distinctly-suffixed `layouts/<family>_normalized/...` (not
plain `layouts/<family>/...`) so it can't collide with, or get silently
skipped as already-done against, any `raw`-mode manifest from section 1 at
the same family/tier. Point the metrics phase (step 2) at that same
suffixed name via `FAMILY_SUBDIR=<family>_normalized`, and glob
`results/<family>_normalized_*_comparison.csv` in the merge step (step 3) --
both shown below. Normalized stress also flips `coordinate_update` from
`bounded` to `gradient` internally (`modules/projector.py:1015`), i.e. this
isn't just a reweighting -- it changes the per-pair step mechanism too.

Six methods per graph (vs. three above) means roughly double the per-shard
cost, so this uses smaller, decreasing `GRAPHS_PER_SHARD` per size tier --
fewer graphs as n grows, since both the baselines (`s_gd2`, `wrap_python`)
and the new `*_100x` variants get slower at scale. **The `--time` values
below are rough, padded, unmeasured estimates for the new methods** (unlike
section 1's, which were calibrated from a real run) -- in particular
`*_100x` uses a batch 100x the default, and at n=10,000 that's roughly a
100x larger per-iteration pair-batch than plain `TorusMDS` (which runs in
under 4s total there), so back-of-envelope that's up to several minutes per
graph at the top tier. Smoke-test the top tier first
(`--array=0-0`, `GRAPHS_PER_SHARD=2`), check the real `t_embed` values in
that shard's `runs.csv`, and size the full submission's `--time` from that --
exactly as section 1 already recommends for its own tiers.

GRG here is restricted to Euclidean + toroidal only (no spherical):
`GRAPH_TYPE_WEIGHTS="1,1,0"`. As with section 1's GRG example, export it as a
shell variable rather than putting it inside `--export=ALL,...` (its commas
would otherwise be split by `sbatch --export` itself).

**Core budget.** Each array task requests `--cpus-per-task=2` (the repo's
existing convention -- one core for the numba-jitted SGD/BFS loop, one of
headroom for `s_gd2`/scipy's BLAS threads). Submitting every tier/family
below unthrottled and simultaneously would ask for up to 76 cores at once
(38 array tasks x 2 cores) on a shared cluster partition -- rude, and likely
to get deprioritized by fair-share scheduling anyway. To cap this at a
polite ~24 cores concurrently:

- **GRG's 3 tiers run one at a time** (`--dependency=singleton`, all sharing
  `--job-name=torus-embed-grg`): singleton makes a submission wait for any
  earlier job with the same name *and user* to finish first, so submitting
  tier 1 then tier 2 then tier 3 in order serializes them automatically --
  at most one GRG tier (3 shards x 2 cores = 6 cores) runs at a time.
- **SBM's 3 tiers do the same** under their own `--job-name=torus-embed-sbm`
  chain (6 cores), running *in parallel with* the GRG chain since the two
  job names don't share a singleton group.
- **SuiteSparse's 20-shard array is throttled in place** via the `%6` array
  suffix (`--array=0-19%6`) instead of serialized -- at most 6 of its 20
  shards run concurrently (12 cores), rather than one at a time, since it's
  a single job array, not several tiers.

Worst case: 6 (GRG) + 6 (SBM) + 12 (SuiteSparse) = 24 cores at once. Raise
or lower the `%6` (and/or add `%K` to the GRG/SBM tiers' own `--array=0-2`
if you want more than one tier per family running at once) to trade cores
for wall-clock time -- lower concurrency finishes later since e.g. GRG's
3 serialized tiers now take roughly the *sum* of their `--time` budgets
back-to-back (~7.5h worst-case here) rather than running side by side.

```bash
cd ~/nobackup/torus-mds   # repo root
mkdir -p slurm_logs

export METHODS="TorusMDS_smart_1x TorusMDS_smart_100x TorusMDS_random_1x TorusMDS_random_100x s_gd2 wrap_python"
export STRESS_MODE=normalized

# --- GRG (Euclidean + toroidal only), 3 tiers serialized via singleton ---
export GRAPH_TYPE_WEIGHTS="1,1,0"
sbatch --job-name=torus-embed-grg --dependency=singleton --array=0-2 --time=00:30:00 --mem=2G \
    --export=ALL,FAMILY=grg,N_MIN=100,N_MAX=1000,GRAPHS_PER_SHARD=150 \
    verification_experiments/slurm/run_embed_array.sbatch
sbatch --job-name=torus-embed-grg --dependency=singleton --array=0-2 --time=02:00:00 --mem=4G \
    --export=ALL,FAMILY=grg,N_MIN=1000,N_MAX=3000,GRAPHS_PER_SHARD=60 \
    verification_experiments/slurm/run_embed_array.sbatch
sbatch --job-name=torus-embed-grg --dependency=singleton --array=0-2 --time=05:00:00 --mem=8G \
    --export=ALL,FAMILY=grg,N_MIN=3000,N_MAX=10000,GRAPHS_PER_SHARD=20 \
    verification_experiments/slurm/run_embed_array.sbatch
unset GRAPH_TYPE_WEIGHTS

# --- SBM, 3 tiers serialized via singleton (own job-name -> runs
#     alongside the GRG chain above, not queued behind it) ---
sbatch --job-name=torus-embed-sbm --dependency=singleton --array=0-2 --time=00:30:00 --mem=2G \
    --export=ALL,FAMILY=sbm,N_MIN=100,N_MAX=1000,GRAPHS_PER_SHARD=150 \
    verification_experiments/slurm/run_embed_array.sbatch
sbatch --job-name=torus-embed-sbm --dependency=singleton --array=0-2 --time=02:00:00 --mem=4G \
    --export=ALL,FAMILY=sbm,N_MIN=1000,N_MAX=3000,GRAPHS_PER_SHARD=60 \
    verification_experiments/slurm/run_embed_array.sbatch
sbatch --job-name=torus-embed-sbm --dependency=singleton --array=0-2 --time=05:00:00 --mem=8G \
    --export=ALL,FAMILY=sbm,N_MIN=3000,N_MAX=10000,GRAPHS_PER_SHARD=20 \
    verification_experiments/slurm/run_embed_array.sbatch

# --- SuiteSparse (fixed staged pool, already spans 100-10,000 with
#     naturally fewer large matrices; sharded by interleaving as usual,
#     throttled to 6 concurrent shards instead of serialized) ---
sbatch --job-name=torus-embed-ss --array=0-19%6 --time=02:00:00 --mem=4G \
    --export=ALL,FAMILY=suitesparse,NUM_SHARDS=20 \
    verification_experiments/slurm/run_embed_array.sbatch
```

Note the `--job-name=...` on the `sbatch` command line overrides the
`#SBATCH --job-name=torus-embed` default baked into `run_embed_array.sbatch`
-- that's required here since `singleton` groups purely by name, and section
1's plain `learn_mode='alpha'` runs (if submitted separately) still use the
script's unmodified default name, so they won't accidentally join either of
these singleton chains.

Track progress with `squeue -u $USER` and `sacct -j <jobid> --format=JobID,Elapsed,State`.
If you want the GRG/SBM chains to run one after another instead of side by
side (12 cores lower peak, longer total time), give them the *same*
`--job-name` instead of two different ones.

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

For section 1b's normalized-stress runs, output lives under the suffixed
`layouts/<family>_normalized/` dir (see section 1b) instead of plain
`layouts/<family>/`, so pass that as `FAMILY_SUBDIR` (defaults to `$FAMILY`
if unset, which is what the plain example above relies on):

```bash
find layouts/sbm_normalized -mindepth 1 -name runs.csv | wc -l   # -> N

sbatch --dependency=afterok:<embed_job_id> --array=0-$((N-1)) \
    --time=00:40:00 --mem=2G --export=ALL,FAMILY=sbm,FAMILY_SUBDIR=sbm_normalized \
    verification_experiments/slurm/run_metrics_array.sbatch
```

Repeat both forms (plain and `_normalized`) for `grg` and `suitesparse`. If
a family's embed phase was submitted as a singleton chain (section 1b),
`<embed_job_id>` should be the *last* tier's job id (e.g. GRG tier 3) --
singleton guarantees tiers 1 and 2 already
finished by the time tier 3 starts, so `afterok` on tier 3 alone is enough
to know the whole family's shards are ready to score. Give the metrics
array its own modest `%K` too (e.g. `--array=0-$((N-1))%8`) if you want to
keep it inside the same core budget as the embed phase, rather than letting
it burst to `N` concurrent tasks.

This phase used to have a much worse
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

Shard CSV names are `<FAMILY_SUBDIR>_<...>_comparison.csv`, so a plain
`sbm_*_comparison.csv` glob would also sweep up `sbm_normalized_*` shards
now that section 1b can produce those too -- anchor the plain-family globs
so they only match a digit (GRG/SBM: `n_min` right after the family name) or
the literal `shard_` (SuiteSparse has no tier prefix), which `_normalized`
shard names never do:

```bash
python verification_experiments/slurm/merge_results.py \
    --glob "results/sbm_[0-9]*_comparison.csv" --output results/sbm_comparison.csv

python verification_experiments/slurm/merge_results.py \
    --glob "results/grg_[0-9]*_comparison.csv" --output results/grg_comparison.csv

python verification_experiments/slurm/merge_results.py \
    --glob "results/suitesparse_shard_*_comparison.csv" --output results/suitesparse_comparison.csv
```

`results/sbm_comparison.csv` and `results/grg_comparison.csv` land in the
same schema the original all-in-one scripts produced, so
`sbm_results.ipynb` / `grg_results.ipynb` work unmodified.

For section 1b's normalized-stress runs, merge separately into their own
`*_normalized_comparison.csv` files rather than mixing them into the plain
merges above (different `stress_mode`, and the new `TorusMDS_*` methods make
these rows a different comparison than section 1's):

```bash
python verification_experiments/slurm/merge_results.py \
    --glob "results/sbm_normalized_*_comparison.csv" --output results/sbm_normalized_comparison.csv

python verification_experiments/slurm/merge_results.py \
    --glob "results/grg_normalized_*_comparison.csv" --output results/grg_normalized_comparison.csv

python verification_experiments/slurm/merge_results.py \
    --glob "results/suitesparse_normalized_shard_*_comparison.csv" --output results/suitesparse_normalized_comparison.csv
```
