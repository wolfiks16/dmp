from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import numpy as np

from magcore.bem.element_integrals import (
    single_layer_face_face_regular,
    triangle_area,
    triangle_centroid,
    triangle_diameter,
    triangle_vertices,
)
from magcore.bem.pair_classification import FacePairRelation, face_pair_relation
from magcore.bem.quadrature import get_triangle_quadrature
from magcore.bem.triangle_subdivision import (
    subdivide_triangle_4,
    triangle_pair_is_regular,
)
from magcore.mesh.surface_mesh import SurfaceMesh


@dataclass(frozen=True, slots=True)
class AdaptiveIntegrationConfig:
    quadrature_order: int = 2
    near_factor: float = 2.0
    max_depth: int = 6
    self_max_depth: int = 8
    min_triangle_area: float = 1.0e-16
    terminal_regularization_factor: float = 0.25


def terminal_pair_approximation(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    regularization_factor: float = 0.25,
) -> float:
    """
    Symmetric finite fallback for unresolved near/singular pair at terminal depth.

    Approximation:
        area(Ta) * area(Tb) / (4*pi*effective_distance)

    where effective_distance = max(centroid_distance, regularization_factor * max(diameters)).
    """
    area_a = triangle_area(tri_a)
    area_b = triangle_area(tri_b)

    ca = triangle_centroid(tri_a)
    cb = triangle_centroid(tri_b)
    dist = float(np.linalg.norm(ca - cb))

    da = triangle_diameter(tri_a)
    db = triangle_diameter(tri_b)
    h = max(da, db)

    eff_dist = max(dist, regularization_factor * h, 1.0e-30)
    return (area_a * area_b) / (4.0 * np.pi * eff_dist)


def single_layer_triangle_pair_adaptive(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    config: AdaptiveIntegrationConfig,
    depth: int = 0,
) -> float:
    """
    Adaptive integration for a general pair of physical triangles.

    Strategy:
    - if pair is regular at current scale -> use regular quadrature
    - else, recursively subdivide the larger triangle
    - if terminal depth or very small triangles are reached -> use terminal finite approximation
    """
    if triangle_pair_is_regular(tri_a, tri_b, near_factor=config.near_factor):
        q = get_triangle_quadrature(config.quadrature_order)
        return single_layer_face_face_regular(
            target_tri=tri_a,
            source_tri=tri_b,
            target_quadrature=q,
            source_quadrature=q,
        )

    area_a = triangle_area(tri_a)
    area_b = triangle_area(tri_b)

    if depth >= config.max_depth or area_a <= config.min_triangle_area or area_b <= config.min_triangle_area:
        return terminal_pair_approximation(
            tri_a,
            tri_b,
            regularization_factor=config.terminal_regularization_factor,
        )

    da = triangle_diameter(tri_a)
    db = triangle_diameter(tri_b)

    acc = 0.0
    if da >= db:
        children = subdivide_triangle_4(tri_a)
        for child in children:
            acc += single_layer_triangle_pair_adaptive(
                child,
                tri_b,
                config=config,
                depth=depth + 1,
            )
    else:
        children = subdivide_triangle_4(tri_b)
        for child in children:
            acc += single_layer_triangle_pair_adaptive(
                tri_a,
                child,
                config=config,
                depth=depth + 1,
            )

    return acc


def single_layer_triangle_self_adaptive(
    tri: np.ndarray,
    config: AdaptiveIntegrationConfig,
    depth: int = 0,
) -> float:
    """
    Adaptive self-interaction integral for one physical triangle:

        I(T,T) = ∫_T ∫_T G(x,y) dS_y dS_x

    Uses self-similarity under midpoint subdivision.

    If T is split into 4 similar subtriangles T_k, then:
        I(T,T) = sum_{k,l} I(T_k, T_l)

    Since each diagonal term satisfies:
        I(T_k, T_k) = (1/8) I(T,T)

    summing k=0..3 gives:
        sum_k I(T_k, T_k) = (1/2) I(T,T)

    Therefore:
        I(T,T) = 2 * sum_{k != l} I(T_k, T_l)

    The off-diagonal child-child pairs are handled by the general adaptive pair integrator.
    """
    area = triangle_area(tri)
    if depth >= config.self_max_depth or area <= config.min_triangle_area:
        return terminal_pair_approximation(
            tri,
            tri,
            regularization_factor=config.terminal_regularization_factor,
        )

    children = subdivide_triangle_4(tri)

    offdiag_sum = 0.0
    for i, tri_i in enumerate(children):
        for j, tri_j in enumerate(children):
            if i == j:
                continue
            offdiag_sum += single_layer_triangle_pair_adaptive(
                tri_i,
                tri_j,
                config=config,
                depth=depth + 1,
            )

    return 2.0 * offdiag_sum


def single_layer_face_face_full(
    mesh: SurfaceMesh,
    face_i: int,
    face_j: int,
    config: AdaptiveIntegrationConfig,
) -> float:
    """
    Complete P0-P0 single-layer interaction between two mesh faces,
    including self / shared-edge / shared-vertex / near / regular cases.
    """
    tri_i = triangle_vertices(mesh, face_i)
    tri_j = triangle_vertices(mesh, face_j)

    relation = face_pair_relation(
        mesh,
        face_i,
        face_j,
        near_factor=config.near_factor,
    )

    if relation == FacePairRelation.SELF:
        return single_layer_triangle_self_adaptive(tri_i, config=config, depth=0)

    if relation == FacePairRelation.REGULAR:
        q = get_triangle_quadrature(config.quadrature_order)
        return single_layer_face_face_regular(
            target_tri=tri_i,
            source_tri=tri_j,
            target_quadrature=q,
            source_quadrature=q,
        )

    # shared-edge / shared-vertex / near
    return single_layer_triangle_pair_adaptive(
        tri_i,
        tri_j,
        config=config,
        depth=0,
    )


def assemble_single_layer_p0p0_full(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    config: AdaptiveIntegrationConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Assemble a complete dense P0-P0 single-layer matrix over the selected faces.

    Returns
    -------
    matrix:
        Dense symmetric matrix of shape (n, n)
    metadata:
        Dict with basic assembly statistics
    """
    if config is None:
        config = AdaptiveIntegrationConfig()

    face_indices = tuple(sorted(face_indices))
    n = len(face_indices)
    local_index = {f: i for i, f in enumerate(face_indices)}

    mat = np.zeros((n, n), dtype=float)
    relation_counter: Counter[str] = Counter()

    for a, face_i in enumerate(face_indices):
        for b, face_j in enumerate(face_indices[a:], start=a):
            relation = face_pair_relation(
                mesh,
                face_i,
                face_j,
                near_factor=config.near_factor,
            )
            relation_counter[relation.value] += 1

            val = single_layer_face_face_full(
                mesh=mesh,
                face_i=face_i,
                face_j=face_j,
                config=config,
            )

            mat[a, b] = val
            mat[b, a] = val

    metadata = {
        "backend": "adaptive_complete_p0p0_single_layer",
        "n_faces": n,
        "quadrature_order": config.quadrature_order,
        "near_factor": config.near_factor,
        "max_depth": config.max_depth,
        "self_max_depth": config.self_max_depth,
        "min_triangle_area": config.min_triangle_area,
        "terminal_regularization_factor": config.terminal_regularization_factor,
        "relation_counts_upper_triangle": dict(relation_counter),
    }

    return mat, metadata