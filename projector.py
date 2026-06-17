# Backwards-compatibility shim — import from modules.projector instead.
from modules.projector import (
    TorusProjector,
    MDSTorusProjector,
    TSNETorusProjector,
    UMAPTorusProjector,
    sgd_minibatch_njit,
    _check_distance_matrix,
    _robust_affine_to_unit_square,
    _wrap_unit_square,
)
from modules.geometry import (
    torus_distance,
    rect_distance,
    rhombic_distance,
    parallelogram_distance,
    make_torus_geod,
    torus_grad,
    euclidean_grad,
    rect_grad,
    rect_stress_and_grad,
    parallelogram_grad,
)
