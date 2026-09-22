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


# ---------------------------------------------------------------- анизотропная ν (тензор)
# ЗАЧЕМ. Спечённый магнит анизотропен: вдоль лёгкой оси ν_∥, поперёк ν_⊥, и это РАЗНЫЕ
# величины (поперечный отклик — поворот моментов против поля анизотропии,
# χ_⊥ = J_s²/(2μ₀K₁), он от продольного перемагничивания не зависит). Скалярная ν
# применяет одно число во все стороны; вдоль оси это безвредно (источник строится под
# замороженную ν, неподвижная точка лежит на кривой ветви при любой ν — см.
# `IrreversibleMagnetState.__call__`), а ПОПЕРЁК защиты нет: там ν работает как настоящее
# свойство материала. Замер: подаём ν=1/μ_rec — решатель применяет поперёк μ_r=1.11 при
# любом заданном μ_perp (tests/unit/test_magnet_anisotropy.py).
#
# ПОВОРОТ. В планарной A_z-постановке B = rot(A_z ẑ) = R∇A_z, где R — поворот на −90°:
#     R = [[0, 1], [−1, 0]].
# Энергия ∫ Bᵀ ν B = ∫ (R∇A)ᵀ ν (R∇A) = ∫ ∇Aᵀ (Rᵀ ν R) ∇A, поэтому в матрицу жёсткости
# идёт ПОВЁРНУТЫЙ тензор ν̃ = Rᵀ ν R. Покомпонентно для ν=[[a,b],[c,d]]:
#     ν̃ = [[d, −c], [−b, a]].
# Для одноосного ν = ν_⊥(I − eeᵀ) + ν_∥ eeᵀ это даёт ν̃ = ν_⊥ eeᵀ + ν_∥ e^⊥(e^⊥)ᵀ —
# роли МЕНЯЮТСЯ МЕСТАМИ, и это правильно: ∇A_z вдоль e отвечает B вдоль e^⊥.
# Скалярная ν — частный случай: Rᵀ(νI)R = νI, поэтому старый путь не меняется.

_ROT_MINUS90 = np.array([[0.0, 1.0], [-1.0, 0.0]])


def rotate_nu_to_gradient(nu_tensor: np.ndarray) -> np.ndarray:
    """ν̃ = Rᵀ ν R — перевод ν, действующей на B, в тензор при ∇A_z. (n_cells,2,2)→(n_cells,2,2)."""
    arr = np.asarray(nu_tensor, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (2, 2):
        raise ValueError("nu tensor must have shape (n_cells, 2, 2).")
    return np.einsum("da,cde,eb->cab", _ROT_MINUS90, arr, _ROT_MINUS90)


def uniaxial_nu_tensor(nu_par, nu_perp, axis) -> np.ndarray:
    """
    Одноосная ν = ν_⊥(I − eeᵀ) + ν_∥ eeᵀ по ячейкам → (n_cells, 2, 2), действующая на B.

    nu_par, nu_perp — скаляр или (n_cells,); axis — (2,) или (n_cells, 2) (нормируется).
    Возвращает НЕПОВЁРНУТЫЙ тензор: поворот делает сборка (одно место на весь код).
    """
    e = np.asarray(axis, dtype=float)
    if e.ndim == 1:
        e = e[None, :]
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("axis must be (2,) or (n_cells, 2).")
    norm = np.linalg.norm(e, axis=1)
    if np.any(norm <= 0.0):
        raise ValueError("axis must be non-zero.")
    e = e / norm[:, None]
    par = np.broadcast_to(np.asarray(nu_par, dtype=float), (e.shape[0],))
    perp = np.broadcast_to(np.asarray(nu_perp, dtype=float), (e.shape[0],))
    ee = np.einsum("ci,cj->cij", e, e)
    eye = np.broadcast_to(np.eye(2), (e.shape[0], 2, 2))
    return perp[:, None, None] * (eye - ee) + par[:, None, None] * ee


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


def _stiffness_local_blocks(nu, grad: np.ndarray, area: np.ndarray) -> np.ndarray:
    """
    Локальные блоки (n_cells,3,3) матрицы жёсткости. ν задаётся как действующая на B:
    скаляр | (n_cells,) — изотропная; (n_cells,2,2) — тензор (анизотропный магнит).
    Тензор поворачивается здесь (ν̃ = Rᵀ ν R) — единственное место в коде, где живёт
    планарная конвенция B = R∇A_z.
    """
    arr = np.asarray(nu, dtype=float)
    if arr.ndim == 3:
        nu_t = rotate_nu_to_gradient(_nu_tensor_per_cell(arr, grad.shape[0]))
        return area[:, None, None] * np.einsum("cad,cde,cbe->cab", grad, nu_t, grad)
    nu_cells = _nu_per_cell(arr, grad.shape[0])
    gg = np.einsum("cad,cbd->cab", grad, grad)                   # (nc,3,3) ∇φ_a·∇φ_b
    return (nu_cells * area)[:, None, None] * gg


def _nu_tensor_per_cell(nu, n_cells: int) -> np.ndarray:
    arr = np.asarray(nu, dtype=float)
    if arr.shape != (n_cells, 2, 2):
        raise ValueError("nu tensor must have shape (n_cells, 2, 2).")
    return arr


def assemble_stiffness_sparse(space: LagrangeP1Space2D, nu) -> sp.csr_matrix:
    """
    РАЗРЕЖЁННАЯ матрица жёсткости K_ij=∫ ∇φ_i·ν̃·∇φ_j dx (для реальных сеток — плотная
    (n,n) не помещается в память: P1 даёт ~7 ненулей в строке). Численно идентична
    `assemble_stiffness`, но CSR + векторизовано.

    ν — скаляр | (n_cells,) | (n_cells,2,2), задаётся как действующая на B (см.
    `rotate_nu_to_gradient`). Скалярный путь численно НЕ ИЗМЕНИЛСЯ.
    """
    mesh = space.mesh
    cells, grad, area = p1_cell_geometry(mesh)
    return _scatter_local(cells, _stiffness_local_blocks(nu, grad, area), space.ndofs)


def assemble_stiffness(space: LagrangeP1Space2D, nu) -> np.ndarray:
    """
    Матрица жёсткости планарной магнитостатики: K_ij = ∫ ∇φ_i·ν̃·∇φ_j dx (2D-аналог
    curl-curl). Плотная (масштаб верификации). ν — как в `assemble_stiffness_sparse`.
    """
    mesh = space.mesh
    n = space.ndofs
    K = np.zeros((n, n), dtype=float)
    arr = np.asarray(nu, dtype=float)
    if arr.ndim == 3:
        nu_t = rotate_nu_to_gradient(_nu_tensor_per_cell(arr, mesh.n_cells))
    else:
        nu_cells = _nu_per_cell(arr, mesh.n_cells)
        nu_t = None
    for c in range(mesh.n_cells):
        verts = mesh.cell_vertices(c)
        area = triangle_area(verts)
        grads = p1_gradients(verts)                 # (3,2), const по ячейке
        local = (area * (grads @ nu_t[c] @ grads.T) if nu_t is not None
                 else nu_cells[c] * area * (grads @ grads.T))     # (3,3)
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
