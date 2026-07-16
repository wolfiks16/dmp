from __future__ import annotations

import numpy as np
import scipy.sparse as sp

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


def p1_cell_geometry(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ВЕКТОРИЗОВАННАЯ геометрия P1 для всей сетки за раз (для разрежённой сборки):
      cells (n_cells,3) индексы вершин, grad (n_cells,3,2) ∇φ_i (const по ячейке),
      area (n_cells,).
    Та же формула, что `p1_gradients`: ∇φ_i=(1/2A)·(y_{i+1}−y_{i+2}, x_{i+2}−x_{i+1}),
    где 2A — ЗНАКОВАЯ площадь (сетка CCW ⇒ >0). Проверяется тестом на совпадение с
    поэлементным `p1_gradients`/`triangle_area`.
    """
    cells = np.asarray(mesh.cells, dtype=int)
    v = np.asarray(mesh.vertices, dtype=float)
    tri = v[cells]                                   # (nc,3,2)
    x, y = tri[:, :, 0], tri[:, :, 1]                # (nc,3)
    area2 = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
             - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))    # знаковая 2·площадь
    if np.any(area2 == 0.0):
        raise ValueError("Degenerate triangle: zero area.")
    j, k = [1, 2, 0], [2, 0, 1]                      # циклические соседи
    gx = (y[:, j] - y[:, k]) / area2[:, None]        # (nc,3)
    gy = (x[:, k] - x[:, j]) / area2[:, None]
    grad = np.stack([gx, gy], axis=2)                # (nc,3,2)
    return cells, grad, np.abs(area2) / 2.0


def _scatter_local(cells: np.ndarray, local: np.ndarray, n: int) -> sp.csr_matrix:
    """Собрать (n_cells,3,3) локальные блоки в разрежённую (n,n) через COO (дубли суммируются)."""
    rows = np.broadcast_to(cells[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(cells[:, None, :], local.shape).ravel()
    return sp.coo_matrix((local.ravel(), (rows, cols)), shape=(n, n)).tocsr()


def assemble_stiffness_sparse(space: LagrangeP1Space2D, nu) -> sp.csr_matrix:
    """
    РАЗРЕЖЁННАЯ матрица жёсткости K_ij=∫ ν ∇φ_i·∇φ_j dx (для реальных сеток — плотная
    (n,n) не помещается в память: P1 даёт ~7 ненулей в строке). Численно идентична
    `assemble_stiffness` (тот же локальный блок ν·A·(∇φ·∇φ)), но CSR + векторизовано.
    """
    mesh = space.mesh
    nu_cells = _nu_per_cell(nu, mesh.n_cells)
    cells, grad, area = p1_cell_geometry(mesh)
    gg = np.einsum("cad,cbd->cab", grad, grad)                   # (nc,3,3) ∇φ_a·∇φ_b
    local = (nu_cells * area)[:, None, None] * gg
    return _scatter_local(cells, local, space.ndofs)


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
