# torus-mds

Embedding graphs on a flat torus via minibatch SGD multidimensional scaling
(MDS). Given a graph's shortest-path distances, `MDSTorusProjector` learns a
2D layout on the surface of a torus (points wrap around at the boundary)
that preserves those distances as closely as possible, jointly fitting the
torus's own scale and, depending on the configuration, its aspect ratio and
shear as well -- from a fixed unit-square torus up to a fully general
parallelogram fundamental domain.

## Setup

```bash
conda env create -f environment.yml
conda activate torus-mds
python -c "import s_gd2, networkx, numba, ssgetpy; print('deps OK')"
```

The `numpy<2` pin in `environment.yml` is so that `s_gd2`'s compiled extension
is built against NumPy 1.x as it fails to import under NumPy 2.x (September 2026).

## Repository map

- **`modules/`** -- the core library. See `modules/__init__.py` for an
  overview of each submodule (the projector/optimizer, torus geometry,
  metrics, visualization, graph I/O, and the experiment-runner driver).
- **`standalone_toruslayout/`** -- a Python/numba port of Chen et al.'s
  "it's-a-wrap" torus layout algorithm [1], used as a baseline comparator
  throughout the experiments (method name `wrap_python`).
- **`verification_experiments/`** -- the paper's experiment pipeline and
  results:
  - `stage_{sbm,grg,suitesparse}.py` -- pregenerate/cache a fixed graph pool
    for a family.
  - `{sbm,grg,suitesparse}_comparison.py` -- embedding phase: run a set of
    methods (TorusMDS variants, `s_gd2`, `wrap_python`) over a graph family
    and persist layouts (no metrics).
  - `compute_metrics.py` -- scoring phase: computes evaluation metrics from
    persisted layouts.
  - `{sbm,grg,suitesparse}_results_normalized.ipynb` and
    `combined_results_normalized.ipynb` -- the paper's results: figures and
    tables built from the merged metrics CSVs.
  - `convergence_test.ipynb` -- a diagnostic notebook for sanity-checking
    SGD convergence/early-stopping behavior, not a results notebook.
  - `draw_layouts.py` / `draw_layout_grids.py` / `draw_layout_singles.py` --
    render persisted layouts to PNG (one-per-method, one grid-per-graph, and
    one-per-method-in-its-own-folder, respectively).
  - `slurm/` -- SLURM deployment scripts and a detailed
    [README](verification_experiments/slurm/README.md) for reproducing the
    full experiment sweep on a cluster.

Generated data (`layouts/`, `layout_drawings*/`, `results/`, `drawings/`,
`figures/`) is not tracked in git (see `.gitignore`) -- reproduce it by
running the pipeline above, or see `verification_experiments/slurm/README.md`
for the cluster-scale version.

## Reproducing the paper's results

1. Stage a graph family (or generate on the fly -- see each script's
   `--help`), then run its embedding phase, e.g.:
   ```bash
   python verification_experiments/stage_suitesparse.py --limit 2000 --n-min 100 --n-max 10000
   python verification_experiments/suitesparse_comparison.py --output-dir layouts/suitesparse
   ```
2. Score the persisted layouts:
   ```bash
   python verification_experiments/compute_metrics.py \
       --layouts-dir layouts/suitesparse --output results/suitesparse_comparison.csv
   ```
3. Open `verification_experiments/suitesparse_results_normalized.ipynb` (or
   `combined_results_normalized.ipynb` for the cross-family comparison) to
   regenerate the paper's figures from that CSV.

This will take several hours on a typical PC. For the full sweep across all three graph families at paper scale, use the
SLURM pipeline documented in
[`verification_experiments/slurm/README.md`](verification_experiments/slurm/README.md).

## References

[1] Chen, Kun-Ting, et al. "It's a Wrap: Toroidal Wrapping of Network
Visualisations Supports Cluster Understanding Tasks." *Proceedings of the
2021 CHI Conference on Human Factors in Computing Systems.* 2021.
https://dl.acm.org/doi/abs/10.1145/3411764.3445439
