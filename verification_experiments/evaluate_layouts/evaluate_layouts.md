# Evaluate saved layouts

Scores persisted layouts without running embedding algorithms or changing inputs.
Run from the repository root:

```sh
python -m pip install numpy scipy pandas numba networkx matplotlib
python verification_experiments/evaluate_layouts/evaluate_layouts.py --layouts-root "/path/to/layouts" --output results/metrics.csv
```

NetworkX and Matplotlib are required by the current `modules/__init__.py` imports;
layout backends such as `s_gd2` are not needed.

## Inputs and options

Each shard must contain: 

```text
shard_0/
  runs.csv
  graphs/graph_<exp_idx>.npz
  coords/exp_<exp_idx>_<method>.npy
```

- `runs.csv`: required columns `exp_idx`, `method`, `status`; only `status=ok`
  rows are evaluated. Preserve fitted geometry and SBM's `k` (see below).
  If present, `n` must match the graph size.
- Graph: SciPy sparse adjacency, square, symmetric, connected, loop-free, with
  finite positive weights and at least two nodes.
- Coordinates: finite `(n, 2)` NumPy array in adjacency row order. Nodes are
  never reordered.

| Option                 | Meaning / default                                                                                 |
|------------------------|---------------------------------------------------------------------------------------------------|
| `--layouts-root PATH`  | Recursively find shards containing `runs.csv`                                                     |
| `--layouts-dir PATH`   | Evaluate one shard; use exactly one input option                                                  |
| `--output PATH`        | Required CSV path; parent directories are created                                                 |
| `--methods NAME ...`   | Exact method names; default: all successful methods                                               |
| `--metric-subsample N` | Sample size and exact-crossing vertex limit; default 1000; 0 means uncapped; otherwise at least 2 |
| `--np-radius R`        | Graph-distance neighbourhood radius; default 2                                                    |
| `--threads N`          | Threads for unweighted BFS and crossings; default 1, within Numba's configured limit              |
| `--checkpoint-every N` | Atomic checkpoint every N new rows; default 20, also saved at completion                          |
| `--overwrite`          | Recompute instead of resuming                                                                     |

## Geometry and scale

`s_gd2` uses Euclidean geometry; every other method uses periodic geometry.
Toroidal coordinates are **fractional lattice coordinates**, wrapped modulo 1,
never min-max normalized. Saved `r0_fit`, `r1_fit`, and `theta_fit` give side
lengths and their angle (radians); missing values default to 1, 1, and pi/2.
Lengths and scales must be finite and positive, and `0 < theta_fit < pi`.
Preserve learned parameters: defaults cannot reconstruct a learned shape.
Skew lattices use a reduced basis with consistently transformed coordinates.

`alpha` is `alpha_fit`, defaulting to 1 when absent. Optimized `TorusMDS*`
methods require it; `TorusMDS_spectral_init` is exempt. Distinguish saved physical
scale from the scale used for distance-error evaluation:

| Method                                                      | `alpha_eval`                                                      |
|-------------------------------------------------------------|-------------------------------------------------------------------|
| `TorusMDS_smart_*`                                          | `alpha * D.max()`; restores units after training on `D / D.max()` |
| Other optimized TorusMDS                                    | `alpha`, without refitting                                        |
| TLE (`TorusMDS_spectral_init`), SGD2, wrap, other baselines | `alpha * beta`, fitted on evaluated pairs                         |

Here `D` is the full graph shortest-path matrix. With saved-scale layout
distances `r_saved`, `beta = sum(D*r_saved) / sum(r_saved^2)` over sampled
unordered pairs. `alpha_eval` affects **only stress, pointwise stress, and
distortion**; it does not change coordinates, shape, hull distances, or edge
lengths. Those use saved geometry and `alpha`. A collapsed baseline has undefined
`alpha_eval` (blank), and all three distance-error scores equal 1.

Edges follow shortest geodesics. Tied shortest paths are selected deterministically
using canonical node-index orientation; opposite-end tangents are reversed.

## Sampling

Uniform sampling without replacement uses seed 42 and the same vertices for all
methods of a graph. Shortest paths are computed on the **full graph before
sampling**, once per graph (BFS for unit weights, Dijkstra otherwise).

| Metrics                                   | Evaluated data                                                        |
|-------------------------------------------|-----------------------------------------------------------------------|
| Stress, pointwise stress, distortion, SGS | All unordered pairs among sampled vertices                            |
| Neighbourhood preservation                | Sampled query nodes; all graph nodes remain candidates                |
| Silhouette                                | Sampled query nodes; all cluster members remain in distance means     |
| Cluster distances                         | Full cluster hulls                                                    |
| Incidence angle, mean edge length         | All nodes and edges                                                   |
| Node resolution                           | All node pairs                                                        |
| Crossings                                 | Exact if original vertex count is within the limit; otherwise skipped |

Sampling can change fitted baseline scales and method rankings. Setting
`--metric-subsample 0` evaluates all nodes and exact crossings on every graph.

## Metric definitions

For distance errors below, sums are over evaluated unordered pairs, `D` denotes
graph distance, `r` the layout distance at `alpha_eval`, and `P` the pair count.

| CSV column                    | Definition                                                                 | Better                  |
|-------------------------------|----------------------------------------------------------------------------|-------------------------|
| `stress`                      | `sqrt(sum((r-D)^2) / sum(D^2))`                                            | Lower                   |
| `pointwise_stress`            | `sqrt(sum(((r-D)/D)^2) / P)`                                               | Lower                   |
| `distortion`                  | `sum(abs(r-D)) / sum(D)`                                                   | Lower                   |
| `sgs`                         | Spearman correlation of graph and layout distances; average ranks for ties | Higher                  |
| `np_score`                    | Mean neighbourhood Jaccard similarity (below)                              | Higher                  |
| `edge_crossings`              | Proper interior intersection events between nonadjacent shortest edges     | Lower                   |
| `incidence_angle`             | Mean relative deviation from ideal angular spacing (below)                 | Lower                   |
| `cluster_distance`            | Mean signed cluster-hull separation (below)                                | Higher                  |
| `mean_edge_length`            | Mean shortest edge length at saved scale                                   | No universal preference |
| `cluster_distance_normalized` | `cluster_distance / mean_edge_length`                                      | Higher                  |
| `silhouette_score`            | Mean SBM-label silhouette in layout distances                              | Higher                  |
| `node_resolution`             | Smallest / largest pairwise node distance in the layout (below)            | Higher                  |

**Neighbourhood preservation.** For each query, compare nodes at graph distance
`0 < D <= np_radius` with equally many closest embedded nodes. Average their
Jaccard similarities, omitting queries without graph-radius neighbours (0 if
none qualify). Retains `geodesic_NP`'s NumPy default sorting and removal of the
first sorted entry, assumed to be self, including its tie/coincident-node behavior.

**Crossings.** Lattice images include seam intersections once. Exclude shared
endpoints, other endpoint contacts, collinear overlaps, and zero-length edges.
Multiple distinct intersections of one edge pair count separately. Tests use a
relative parallelism tolerance and a `1e-12` endpoint-parameter tolerance.
Skipped crossings are blank/NaN, **not zero**, with
`crossings_status=skipped_vertex_limit`; otherwise `exact`. Skipping is not an
error.

**Incidence angle.** Average over all nodes:
`abs(2*pi/degree - minimum_cyclic_angle_gap) / (2*pi/degree)`.
Use physical tangent directions under the saved lattice shape. Degree 0 or 1
contributes zero. Coincident linked endpoints make this undefined and the row
`partial`.

**Node resolution.** `min d(u,v) / max d(u,v)` over all unordered node pairs,
with `d` the Euclidean distance for `s_gd2` and the torus distance under the
saved lattice shape otherwise (same metric as stress). Uniform scale cancels,
so `alpha` is irrelevant. Coincident nodes give 0; blank if all nodes coincide.
Computed in O(n^2) time without materializing a distance matrix.

**SBM labels.** A nonmissing `k` identifies planted consecutive blocks: the first
`k-1` have `n//k` nodes; the last receives the remainder. Other graph families
have blank cluster metrics. Arbitrary labels cannot be supplied through the CLI.

**Cluster distance.** Following the signed hull-distance definition in
[Chen et al., It's a Wrap, section 4.3.2],
average over unordered cluster pairs: positive minimum boundary distance when
separated, negative minimum penetration depth when overlapping, zero on contact.
Euclidean layouts use ordinary hulls and native saved scale.

For torus hulls, first reduce skew lattices, then cut each cluster at the largest
empty gap in each fractional coordinate and center its bounding box in the
central tile. Equal gaps select the first sorted gap. Test the surrounding 3x3
lattice images and take the minimum signed distance (deepest overlap if several
images overlap). This lifting convention is an implementation choice: the paper
does not specify one, and diffuse clusters can depend on it. Distances are
computed via 2D Minkowski differences; point/collinear hulls are supported and
zero-area overlaps have zero penetration depth.

`cluster_distance_normalized` expresses signed separation in mean edge lengths;
uniform scaling cancels. This normalization is additional to the paper's metric.
It is blank when labels are unavailable or mean edge length is zero/nonfinite;
no epsilon is added. Prefer it to raw hull distances for method comparisons on
the same graph.

**Silhouette.** Using layout distances and SBM labels, `a` is mean distance to
other members of the same block; `b` is the minimum mean distance to another
block. Average `(b-a)/max(a,b)`, in [-1, 1]. Singletons and `a=b=0` contribute zero.
Blank without labels or when cluster count is outside `2..n-1`. Uniform scale
cancels. SGS is likewise undefined (blank) for constant distance ranks.

The older `compute_metrics.py` pipeline refits every method's scale, defaults to
100 sampled nodes, and restricts neighbourhood candidates to that sample.
This evaluator's saved-scale rules, 1000-node default, full-candidate queries,
and skew geometry can therefore produce different results. Circular alignment
is a separate experiment, not computed here.

## Output and resume

Each selected instance produces one row retaining manifest metadata. The key is
`(source_shard, exp_idx, method)`; duplicate successful manifest instances are
rejected. `source_shard` is relative to the nearest `layouts/` ancestor, or the
absolute shard path if none exists. Retain it when merging shard outputs.

`metric_status` is `ok`, `partial` (undefined incidence), or `error`, with details
in `metric_error`. Any error row causes a nonzero exit code. Sampling provenance
is recorded as `metric_subsample`, `metric_sample_n`, `metric_sample_seed`, and
`np_radius`.

Reruns reuse `ok`/`partial` rows, retry errors, and append newly selected instances;
existing rows outside a method filter remain. Resume rejects incompatible CSV
schemas, duplicate identities, changed radius/sampling, and ambiguous legacy
shard identities. Keep inputs unchanged; use a new output or `--overwrite` after
changing data or metric code. Use separate output files for concurrent jobs.
An interruption loses work since the last checkpoint; layouts resume only at
row boundaries.

## Cost and checks

Full graph shortest paths still require quadratic memory. Sampled layout matrices
use sample-size-squared memory; query distances are processed in 64-row batches.
Exact crossings use spatial indexing but can still require quadratic work in
edge count. Numba compilation adds first-run overhead.

```sh
python -m pip install pytest
python -m pytest verification_experiments/evaluate_layouts -q
python verification_experiments/evaluate_layouts/verify_cluster_distance.py
```

There is `verify_cluster_distance.py` for an optional hull verification. 
It defaults to saved SBM layouts under `verification_experiments/layouts/sbm_normalized/100_1000/shard_0`; override
`--layouts-dir` / `--output-dir` as needed. It checks analytic cases, an independent
polygon oracle, invariances, and 5x5 versus production 3x3 image searches. Outputs
go to the evaluator's `validation/` directory by default; inputs are unchanged.
