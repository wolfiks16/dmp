from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.bem.element_integrals import (
    FacePairClass,
    classify_face_pair,
    single_layer_face_face_regular,
    triangle_vertices,
)
from magcore.bem.quadrature import get_triangle_quadrature
from magcore.mesh.surface_mesh import SurfaceMesh


@dataclass(frozen=True, slots=True)
class RegularAssemblyMask:
    regular_pairs: tuple[tuple[int, int], ...]
    near_pairs: tuple[tuple[int, int], ...]
    singular_pairs: tuple[tuple[int, int], ...]

    @property
    def n_regular(self) -> int:
        return len(self.regular_pairs)

    @property
    def n_near(self) -> int:
        return len(self.near_pairs)

    @property
    def n_singular(self) -> int:
        return len(self.singular_pairs)


def build_regular_pair_mask(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    near_factor: float = 2.0,
) -> RegularAssemblyMask:
    """
    Classify all ordered face pairs over the given face subset into:
    - regular
    - near
    - singular
    """
    face_indices = tuple(sorted(face_indices))

    regular: list[tuple[int, int]] = []
    near: list[tuple[int, int]] = []
    singular: list[tuple[int, int]] = []

    for i in face_indices:
        for j in face_indices:
            pair_class = classify_face_pair(mesh, i, j, near_factor=near_factor)
            pair = (i, j)

            if pair_class == FacePairClass.REGULAR:
                regular.append(pair)
            elif pair_class == FacePairClass.NEAR:
                near.append(pair)
            else:
                singular.append(pair)

    return RegularAssemblyMask(
        regular_pairs=tuple(regular),
        near_pairs=tuple(near),
        singular_pairs=tuple(singular),
    )


def assemble_single_layer_p0p0_regular(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    quadrature_order_target: int = 2,
    quadrature_order_source: int = 2,
    near_factor: float = 2.0,
    strict: bool = True,
) -> tuple[np.ndarray, RegularAssemblyMask]:
    """
    Assemble a dense P0-P0 regular-only single-layer matrix over the selected faces.

    Matrix entry:
        V_ij = ∫_{T_i} ∫_{T_j} G(x, y) dS_y dS_x

    Policy:
    - REGULAR pairs are assembled numerically
    - NEAR / SINGULAR pairs are unsupported at R1.4
      * if strict=True: raise NotImplementedError
      * if strict=False: store np.nan in those matrix entries
    """
    face_indices = tuple(sorted(face_indices))
    n = len(face_indices)

    q_t = get_triangle_quadrature(quadrature_order_target)
    q_s = get_triangle_quadrature(quadrature_order_source)

    local_index = {f: k for k, f in enumerate(face_indices)}
    mat = np.zeros((n, n), dtype=float)

    mask = build_regular_pair_mask(
        mesh=mesh,
        face_indices=face_indices,
        near_factor=near_factor,
    )

    if strict and (mask.near_pairs or mask.singular_pairs):
        raise NotImplementedError(
            "Regular-only assembly encountered near or singular face pairs. "
            "Use a disjoint face set or set strict=False."
        )

    for i, j in mask.regular_pairs:
        tri_i = triangle_vertices(mesh, i)
        tri_j = triangle_vertices(mesh, j)

        val = single_layer_face_face_regular(
            target_tri=tri_i,
            source_tri=tri_j,
            target_quadrature=q_t,
            source_quadrature=q_s,
        )
        mat[local_index[i], local_index[j]] = val

    if not strict:
        for i, j in mask.near_pairs:
            mat[local_index[i], local_index[j]] = np.nan

        for i, j in mask.singular_pairs:
            mat[local_index[i], local_index[j]] = np.nan

    return mat, mask