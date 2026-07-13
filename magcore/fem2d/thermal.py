from __future__ import annotations

import numpy as np

from magcore.fem2d.assembly import assemble_current_rhs, assemble_stiffness
from magcore.fem2d.mesh import triangle_area
from magcore.fem2d.solver import apply_dirichlet, solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D

# Стационарная теплопроводность на той же треугольной сетке: −div(k∇T)=q + конвекция
# (Robin) −k ∂T/∂n = h(T−T_amb) на границе. Структурно = магнитостатика (k↔ν): жёсткость
# ∫k∇φ·∇φ переиспользует `assemble_stiffness`. Новое здесь — объёмный источник по ячейке
# и граничный член Robin (краевая масса + нагрузка от T_amb).


def assemble_source_rhs(space: LagrangeP1Space2D, q_cells: np.ndarray) -> np.ndarray:
    """Вектор объёмного источника: f_i = ∫ q φ_i, q — кусочно-постоянна (n_cells,). ∫_T φ_i=A/3."""
    mesh = space.mesh
    q = np.asarray(q_cells, dtype=float)
    if q.shape != (mesh.n_cells,):
        raise ValueError("q_cells must have shape (n_cells,).")
    f = np.zeros(space.ndofs, dtype=float)
    for c in range(mesh.n_cells):
        area = triangle_area(mesh.cell_vertices(c))
        f[list(mesh.cell_vertex_indices(c))] += q[c] * area / 3.0
    return f


def assemble_robin_boundary(
    space: LagrangeP1Space2D, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Граничный член конвекции −k∂T/∂n=h(T−T_amb): возвращает (R, amb_load), где
    R_ij=∫_∂Ω h φ_i φ_j ds (добавить к жёсткости), amb_load_i=∫_∂Ω h φ_i ds (RHS=T_amb·amb_load).
    Краевая масса линейного элемента: (L/6)[[2,1],[1,2]]; ∫_ребро φ_i ds = L/2.
    """
    mesh = space.mesh
    n = space.ndofs
    R = np.zeros((n, n), dtype=float)
    amb = np.zeros(n, dtype=float)
    for i, j in mesh.boundary_edges():
        L = float(np.linalg.norm(mesh.vertices[i] - mesh.vertices[j]))
        R[i, i] += h * L / 3.0
        R[j, j] += h * L / 3.0
        R[i, j] += h * L / 6.0
        R[j, i] += h * L / 6.0
        amb[i] += h * L / 2.0
        amb[j] += h * L / 2.0
    return R, amb


def solve_thermal(
    space: LagrangeP1Space2D,
    k,
    *,
    source,
    h: float | None = None,
    T_amb: float = 0.0,
    dirichlet_dofs=None,
    dirichlet_values=0.0,
    quadrature_order: int = 5,
) -> np.ndarray:
    """
    Стационарное тепловое поле T (P1). `k` — тепловодность (скаляр|(n_cells,)).
    `source` — объёмный тепловыдел q: массив (n_cells,) [потери] ИЛИ callable(x)->q [MMS].
    Конвекция: h (коэфф.) + T_amb на границе (Robin). Опц. Dirichlet на части узлов.
    """
    K = assemble_stiffness(space, k)
    if callable(source):
        f = assemble_current_rhs(space, source, quadrature_order=quadrature_order)
    else:
        f = assemble_source_rhs(space, np.asarray(source, dtype=float))
    if h is not None:
        R, amb_load = assemble_robin_boundary(space, float(h))
        K = K + R
        f = f + float(T_amb) * amb_load
    if dirichlet_dofs is not None:
        K, f = apply_dirichlet(K, f, dirichlet_dofs, dirichlet_values)
    return solve_scalar(K, f)
