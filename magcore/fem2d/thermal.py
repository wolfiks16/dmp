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


def assemble_capacity(space: LagrangeP1Space2D, c_cells) -> np.ndarray:
    """
    Матрица теплоёмкости C_ij = ∫ c φ_i φ_j, где c = ρ·c_p [Дж/(м³·K)] — объёмная
    теплоёмкость (скаляр|(n_cells,), разная по регионам: медь/сталь/магнит/воздух).
    Локально (c·A/12)·[[2,1,1],[1,2,1],[1,1,2]]. Структура = матрица масс, взвешенная c.
    """
    mesh = space.mesh
    c = np.asarray(c_cells, dtype=float)
    if c.ndim == 0:
        c = np.full(mesh.n_cells, float(c))
    elif c.shape != (mesh.n_cells,):
        raise ValueError("c_cells must be a scalar or shape (n_cells,).")
    n = space.ndofs
    C = np.zeros((n, n), dtype=float)
    base = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]], dtype=float) / 12.0
    for cell in range(mesh.n_cells):
        area = triangle_area(mesh.cell_vertices(cell))
        idx = mesh.cell_vertex_indices(cell)
        Ce = c[cell] * area * base
        for a in range(3):
            for b in range(3):
                C[idx[a], idx[b]] += Ce[a, b]
    return C


class ImplicitEulerThermalStepper:
    """
    Один шаг неявного Эйлера для C·∂T/∂t − div(k∇T) = q + конвекция (Robin):
        (C/dt + K + R)·T^{n+1} = (C/dt)·T^n + f(q^{n+1}) + T_amb·amb_load.
    Матрицы собираются ОДИН раз (k, c, h, dt постоянны) — шаг переиспользуется как
    нестационарным решателем, так и связкой магнит↔тепло (источник q зависит от T).

    Схема безусловно устойчива (A = C/dt + K + R симметрична положительно определена
    при h>0), поэтому шаг ограничен только точностью, а не устойчивостью.
    """

    def __init__(self, space: LagrangeP1Space2D, k, capacity, *, dt: float, h: float, T_amb: float):
        if float(dt) <= 0.0:
            raise ValueError("dt must be positive.")
        self.space = space
        self.dt = float(dt)
        self.T_amb = float(T_amb)
        self.K = assemble_stiffness(space, k)
        self.C = assemble_capacity(space, capacity)
        self.R, self.amb_load = assemble_robin_boundary(space, float(h))
        self.Cdt = self.C / self.dt
        self.A = self.Cdt + self.K + self.R

    def step(self, T: np.ndarray, q_cells: np.ndarray) -> np.ndarray:
        """Поле на следующем шаге по текущему T и источнику потерь q [Вт/м³] (по ячейкам)."""
        f = assemble_source_rhs(self.space, np.asarray(q_cells, dtype=float))
        return solve_scalar(self.A, self.Cdt @ np.asarray(T, dtype=float)
                            + f + self.T_amb * self.amb_load)

    # --- диагностика энергобаланса (оракул связки) ---
    def stored_energy(self, T) -> float:
        """Запасённая тепловая энергия ∫ c·T dV = 1ᵀ·C·T [Дж/м] (на единицу длины)."""
        return float(np.asarray(T, dtype=float) @ self.C.sum(axis=0))

    def convective_outflow(self, T) -> float:
        """Отвод конвекцией ∫ h(T−T_amb) ds = 1ᵀR·T − T_amb·1ᵀ·amb_load [Вт/м]."""
        Tv = np.asarray(T, dtype=float)
        return float(Tv @ self.R.sum(axis=0)) - self.T_amb * float(self.amb_load.sum())


def solve_thermal_transient(
    space: LagrangeP1Space2D,
    k,
    capacity,
    *,
    source,
    dt: float,
    n_steps: int,
    h: float,
    T_amb: float,
    T0=None,
):
    """
    НЕСТАЦИОНАРНАЯ теплопроводность C·∂T/∂t − div(k∇T) = q + конвекция (Robin), неявный
    Эйлер (безусловно устойчив): (C/dt + K + R)·T^{n+1} = (C/dt)·T^n + f^{n+1}.

    capacity — c=ρc_p [Дж/(м³·K)] (скаляр|(n_cells,)); k — теплопроводность; h,T_amb —
    конвекция на границе; T0 — начальное поле (по умолч. T_amb всюду).
    source — источник потерь q [Вт/м³]: массив (n_cells,) ПОСТОЯННЫЙ, ИЛИ callable(step:int,
    t:float)->(n_cells,) (для связки с магнитными потерями, зависящими от T).

    Возвращает (times:(n_steps+1,), T_hist:(n_steps+1, ndofs)) — поле на каждом шаге.
    """
    stepper = ImplicitEulerThermalStepper(space, k, capacity, dt=dt, h=h, T_amb=T_amb)
    T = (np.full(space.ndofs, float(T_amb), dtype=float)
         if T0 is None else np.asarray(T0, dtype=float).copy())
    times = [0.0]
    hist = [T.copy()]
    for n in range(1, int(n_steps) + 1):
        t = n * float(dt)
        q = source(n, t) if callable(source) else source
        T = stepper.step(T, q)
        times.append(t)
        hist.append(T.copy())
    return np.asarray(times), np.asarray(hist)
