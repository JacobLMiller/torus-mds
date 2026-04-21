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
    torus_distance_njit as torus,
    torus_distance,
    torus_grad,
    euclidean_grad,
    stress_and_grad_rect_torus,
)
