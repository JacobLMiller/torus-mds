# SLURM deployment

Two decoupled phases (see the driver scripts in `verification_experiments/`
and `modules/experiment_runner.py`): **embed** (persists layouts, no
metrics) then **score** (computes metrics from the persisted layouts). No
Node.js/npm setup is needed anywhere -- the Chen `wrap_python` method is a
pure Python/numba port, not a subprocess call to Node.

Fill in `<FILL_IN>` in both `.sbatch` files (partition/account) and any
cluster-specific module loads before submitting.

## 0. Stage SuiteSparse (once, locally or on a login node with internet)

Compute nodes never touch the network -- this must run somewhere with
internet access first, writing into a directory the compute nodes can read
(shared filesystem):

```bash
python verification_experiments/stage_suitesparse.py \
    --limit 2000 --n-min 100 --n-max 10000 \
    --output-dir data/suitesparse_cache
```

## 1. Submit embedding jobs

One `sbatch` submission per (family, size tier). Example for SBM across
three log-spaced size tiers (adjust `--time`/`--mem` up for the larger
tiers -- `wrap_python` is capped at `--wrap-python-max-n`, so TorusMDS/s_gd2
dominate runtime above that):

```bash
mkdir -p slurm_logs

sbatch --array=0-2 --time=00:15:00 --mem=2G \
    --export=FAMILY=sbm,N_MIN=100,N_MAX=1000,GRAPHS_PER_SHARD=334 \
    verification_experiments/slurm/run_embed_array.sbatch

sbatch --array=0-2 --time=00:45:00 --mem=4G \
    --export=FAMILY=sbm,N_MIN=1000,N_MAX=3000,GRAPHS_PER_SHARD=334 \
    verification_experiments/slurm/run_embed_array.sbatch

sbatch --array=0-2 --time=03:00:00 --mem=8G \
    --export=FAMILY=sbm,N_MIN=3000,N_MAX=10000,GRAPHS_PER_SHARD=334 \
    verification_experiments/slurm/run_embed_array.sbatch
```

Repeat with `FAMILY=grg`. For SuiteSparse (a fixed pre-staged pool, sharded
by interleaving rather than by size tier since it spans the whole range at
once):

```bash
sbatch --array=0-19 --time=01:00:00 --mem=4G \
    --export=FAMILY=suitesparse,NUM_SHARDS=20 \
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
    --time=00:20:00 --mem=2G --export=FAMILY=sbm \
    verification_experiments/slurm/run_metrics_array.sbatch
```

Repeat for `grg` and `suitesparse`. This phase is much cheaper than
embedding (dominated by sparse APSP), so modest `--time`/`--mem` suffice
even for the largest graphs.

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
