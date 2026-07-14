from __future__ import annotations

import numpy as np

from magcore.fem2d.mesh import p1_gradients, triangle_area
from magcore.fem2d.quadrature import triangle_quadrature
from magcore.fem2d.spaces import LagrangeP1Space2D


def _nu_per_cell(nu, n_cells: int) -> np.ndarray:
    """Привести ν к массиву (n_cells,): скаляр → const, массив → как есть."""
    arr = np.asarray(nu, dtype=float)
    if arr.ndim == 0:
        return np.full(n_cells, float(arr))
    if arr.shape != (n_cells,):
        raise ValueError("nu must be a scalar or an array of shape (n_cells,).")
    return arr


def assemble_stiffness(space: LagrangeP1Space2D, nu) -> np.ndarray:
    """
    Матрица жёсткости планарной магнитостатики: K_ij = ∫ ν ∇φ_i·∇φ_j dx (2D-аналог
    curl-curl). Поячеечная ν (скаляр|(n_cells,)). Плотная (масштаб верификации).
    """
    mesh = space.mesh
    n = space.ndofs
    nu_cells = _nu_per_cell(nu, mesh.n_cells)
    K = np.zeros((n, n), dtype=float)
    for c in range(mesh.n_cells):
        verts = mesh.cell_vertices(c)
        area = triangle_area(verts)
        grads = p1_gradients(verts)                 # (3,2), const по ячейке
        local = nu_cells[c] * area * (grads @ grads.T)  # (3,3)
        idx = mesh.cell_vertex_indices(c)
        for a in range(3):
            for b in range(3):
                K[idx[a], idx[b]] += local[a, b]
    return K


def assemble_mass(space: LagrangeP1Space2D) -> np.ndarray:
    """Матрица масс P1: M_ij = ∫ φ_i φ_j dx. Локально (A/12)·[[2,1,1],[1,2,1],[1,1,2]]."""
    mesh = space.mesh
    n = space.ndofs
    M = np.zeros((n, n), dtype=float)
    local = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=float)
    for c in range(mesh.n_cells):
        area = triangle_area(mesh.cell_vertices(c))
        idx = mesh.cell_vertex_indices(c)
        Me = (area / 12.0) * local
        for a in range(3):
            for b in range(3):
                M[idx[a], idx[b]] += Me[a, b]
    return M


def assemble_current_rhs(
    space: LagrangeP1Space2D, J_fn, *, quadrature_order: int = 5
) -> np.ndarray:
    """
    Вектор нагрузки от внеплоскостного тока: f_i = ∫ J_z(x) φ_i(x) dx.
    J_fn(x:(2,)) -> float  (плотность тока по нормали к плоскости, А/м²).
    """
    mesh = space.mesh
    f = np.zeros(space.ndofs, dtype=float)
    bary, w = triangle_quadrature(quadrature_order)
    for c in range(mesh.n_cells):
        verts = mesh.cell_vertices(c)            # (3,2)
        area = triangle_area(verts)
        idx = mesh.cell_vertex_indices(c)
        for q in range(bary.shape[0]):
            lam = bary[q]                        # (3,) барицентрич. = значения φ в точке
            x = lam @ verts                      # физическая точка
            jz = float(J_fn(x))
            f[list(idx)] += w[q] * area * jz * lam
    return f


def assemble_current_rhs_piecewise(
    space: LagrangeP1Space2D, jz_cells: np.ndarray
) -> np.ndarray:
    """
    Вектор нагрузки от КУСОЧНО-ПОСТОЯННОГО внеплоскостного тока (J_z константа в ячейке —
    как ток в пазу обмотки): f_i = ∫ J_z φ_i dx = Σ_c J_z^c · ∫_c φ_i = Σ_c J_z^c·area_c/3
    (точно для P1: ∫_c φ_i = area/3). `jz_cells`:(n_cells,) — плотность тока А/м² по ячейкам
    (0 вне источника). Быстрее и естественнее квадратурного `assemble_current_rhs`, когда
    источник задан поячеечно (P3: обмотка → J_z).
    """
    mesh = space.mesh
    jz = np.asarray(jz_cells, dtype=float)
    if jz.shape != (mesh.n_cells,):
        raise ValueError("jz_cells must have shape (n_cells,).")
    f = np.zeros(space.ndofs, dtype=float)
    for c in range(mesh.n_cells):
        if jz[c] == 0.0:
            continue
        area = triangle_area(mesh.cell_vertices(c))
        idx = mesh.cell_vertex_indices(c)
        f[list(idx)] += jz[c] * area / 3.0
    return f


def assemble_magnetization_rhs(
    space: LagrangeP1Space2D, nu_br_cells: np.ndarray
) -> np.ndarray:
    """
    Вектор источника постоянного магнита: f_i = ∫ ν B_r · (∂_y φ_i, −∂_x φ_i) dx
    (2D-аналог `∫ ν B_r·curl v`). `nu_br_cells`:(n_cells,2) — поячеечный ν·B_r в плоскости
    (ненулевой только в магните). ∇φ_i и ν·B_r постоянны по ячейке ⇒ точно без квадратуры.
    """
    mesh = space.mesh
    nu_br = np.asarray(nu_br_cells, dtype=float)
    if nu_br.shape != (mesh.n_cells, 2):
        raise ValueError("nu_br_cells must have shape (n_cells, 2).")
    f = np.zeros(space.ndofs, dtype=float)
    for c in range(mesh.n_cells):
        verts = mesh.cell_vertices(c)
        area = triangle_area(verts)
        grads = p1_gradients(verts)              # (3,2)
        idx = mesh.cell_vertex_indices(c)
        bx, by = nu_br[c, 0], nu_br[c, 1]
        # ν B_r · (∂_yφ_i, −∂_xφ_i) = bx·g_i[1] − by·g_i[0]
        contrib = area * (bx * grads[:, 1] - by * grads[:, 0])
        f[list(idx)] += contrib
    return f
