"""
Core library for embedding graphs on a flat torus via minibatch SGD MDS.

- `projector`: `MDSTorusProjector`, the main optimizer (learns point
  positions and, depending on `learn_mode`, the torus's scale/aspect
  ratio/shear -- see `LearnMode`).
- `geometry`: closed-form flat-torus geodesic distances and gradients
  (rectangular, rhombic, and general parallelogram fundamental domains).
- `exact_torus_stress`: exact (non-SGD) stress evaluation/L-BFGS polishing,
  used as the `finalize="lbfgs"` refinement step.
- `metrics`: evaluation metrics (stress, distortion, neighborhood
  preservation) used to score a layout against the input graph distances.
- `visualization`: plotting a layout with torus-wrapped edges.
- `graphio`: graph I/O helpers (shortest-path distances, periodic lattices,
  the Chen et al. reference-graph format).
- `experiment_runner`: the shared driver behind
  `verification_experiments/{sbm,grg,suitesparse}_comparison.py` -- runs a
  set of embedding methods over a stream of graphs and persists layouts.
"""

from .geometry import (
    torus_distance,
    rect_distance,
    rhombic_distance,
    parallelogram_distance,
    gauss_reduce_basis,
    lengths_angle_to_xy,
    xy_to_lengths_angle,
    height_from_raw,
    raw_from_height,
    make_torus_geod,
    torus_grad,
    euclidean_grad,
    rect_grad,
    rect_stress_and_grad,
    parallelogram_grad,
    torus_delta,
    torus_edge_segments,
)
from .projector import (
    TorusProjector,
    MDSTorusProjector,
    TSNETorusProjector,
    UMAPTorusProjector,
)
from .exact_torus_stress import (
    prepare_pair_data,
    evaluate_profiled_objective_and_gradient,
    polish_torus_layout,
)
from .metrics import (
    subsample,
    geodesic_matrix,
    geodesic_stress,
    scaled_geodesic_stress,
    pointwise_geodesic_stress,
    scaled_pointwise_geodesic_stress,
    geodesic_distortion,
    SGS,
    geodesic_NP,
    estimate_alpha,
    layout_edge_geometry,
    edge_crossings,
    incidence_angle,
)
from .visualization import (
    plot_embedding_with_torus_edges,
    plot_embedding,
    shepard_diagram,
)
from .graphio import (
    apsp_distance_matrix,
    periodic_lattice_graph,
    get_periodic_lattice,
    parse_chen_json,
    load_chen_graphs,
)
