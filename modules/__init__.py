from .geometry import (
    torus_distance,
    parallelogram_distance,
    gauss_reduce_basis,
    lengths_angle_to_xy,
    xy_to_lengths_angle,
    height_from_raw,
    raw_from_height,
    make_torus_geod,
    torus_grad,
    euclidean_grad,
    stress_and_grad_rect_torus,
    grad_parallelogram_torus,
    torus_delta,
    torus_edge_segments,
)
from .projector import (
    TorusProjector,
    MDSTorusProjector,
    TSNETorusProjector,
    UMAPTorusProjector,
)
from .metrics import (
    subsample,
    geodesic_matrix,
    geodesic_stress,
    geodesic_distortion,
    SGS,
    geodesic_NP,
    estimate_alpha,
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
from .comparison import (
    GraphComparisonSpec,
    LayoutComparisonResult,
    compare_standalone_vs_mdstorus_graphs,
    run_standalone_vs_mdstorus,
    summarize_standalone_vs_mdstorus,
)
