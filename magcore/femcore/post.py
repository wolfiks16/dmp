from __future__ import annotations

import numpy as np

from magcore.femcore.basis_nedelec import (
    physical_nedelec_basis,
    physical_nedelec_curl,
)
from magcore.femcore.reference_tetra import AffineTetraMap
from magcore.femcore.spaces import NedelecP1Space


def evaluate_A_on_cell(
    space: NedelecP1Space,
    coeffs: np.ndarray,
    cell_idx: int,
    physical_point: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the discrete vector potential A_h at a point inside one cell.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.shape != (space.ndofs,):
        raise ValueError("coeffs shape must be (space.ndofs,).")

    x = np.asarray(physical_point, dtype=float)
    if x.shape != (3,):
        raise ValueError("physical_point must have shape (3,).")

    amap = AffineTetraMap(space.mesh.cell_vertices(cell_idx))
    gdofs = space.cell_dof_indices(cell_idx)
    sgn = space.cell_dof_signs(cell_idx)

    val = np.zeros(3, dtype=float)
    for i in range(6):
        gi = gdofs[i]
        si = sgn[i]
        wi = physical_nedelec_basis(amap, i, x)
        val += coeffs[gi] * si * wi

    return val


def evaluate_curl_on_cell(
    space: NedelecP1Space,
    coeffs: np.ndarray,
    cell_idx: int,
) -> np.ndarray:
    """
    Evaluate the discrete curl(A_h) on one affine tetrahedral cell.

    For first-order Nedelec elements on an affine tetrahedron, the curl is
    constant on the cell.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.shape != (space.ndofs,):
        raise ValueError("coeffs shape must be (space.ndofs,).")

    amap = AffineTetraMap(space.mesh.cell_vertices(cell_idx))
    gdofs = space.cell_dof_indices(cell_idx)
    sgn = space.cell_dof_signs(cell_idx)

    val = np.zeros(3, dtype=float)
    for i in range(6):
        gi = gdofs[i]
        si = sgn[i]
        ci = physical_nedelec_curl(amap, i)
        val += coeffs[gi] * si * ci

    return val


def l2_error_at_cell_centroids(
    space: NedelecP1Space,
    coeffs: np.ndarray,
    ref_A_fn,
) -> float:
    """
    Compute a cell-volume-weighted L2-like error at cell centroids.

    This is a practical postprocessing norm for regression tests.
    """
    err2 = 0.0
    vol_sum = 0.0

    for cell_idx in range(space.mesh.n_cells):
        x = space.mesh.cell_centroid(cell_idx)
        Ah = evaluate_A_on_cell(space, coeffs, cell_idx, x)
        Aref = np.asarray(ref_A_fn(x), dtype=float)

        vol = space.mesh.cell_volume(cell_idx)
        err2 += vol * float(np.dot(Ah - Aref, Ah - Aref))
        vol_sum += vol

    return float(np.sqrt(err2 / max(vol_sum, 1.0e-30)))


def l2_curl_error_at_cell_centroids(
    space: NedelecP1Space,
    coeffs: np.ndarray,
    ref_curl_fn,
) -> float:
    """
    Compute a cell-volume-weighted L2-like error for curl(A_h) at cell centroids.
    """
    err2 = 0.0
    vol_sum = 0.0

    for cell_idx in range(space.mesh.n_cells):
        curl_h = evaluate_curl_on_cell(space, coeffs, cell_idx)
        x = space.mesh.cell_centroid(cell_idx)
        curl_ref = np.asarray(ref_curl_fn(x), dtype=float)

        vol = space.mesh.cell_volume(cell_idx)
        err2 += vol * float(np.dot(curl_h - curl_ref, curl_h - curl_ref))
        vol_sum += vol

    return float(np.sqrt(err2 / max(vol_sum, 1.0e-30)))