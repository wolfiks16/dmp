from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from magcore.fem2d.assembly import (
    assemble_current_rhs,
    assemble_mass_sparse,
    assemble_stiffness_sparse,
    p1_cell_geometry,
)
from magcore.fem2d.solver import apply_dirichlet, solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D

# Стационарная теплопроводность на той же треугольной сетке: −div(k∇T)=q + конвекция
# (Robin) −k ∂T/∂n = h(T−T_amb) на границе. Структурно = магнитостатика (k↔ν): жёсткость
# ∫k∇φ·∇φ переиспользует `assemble_stiffness_sparse`. Новое здесь — объёмный источник по ячейке
# и граничный член Robin (краевая масса + нагрузка от T_amb).
#
# Все матрицы (n,n) — РАЗРЕЖЁННЫЕ (CSR): у P1 ~7 ненулей в строке, а плотная (n,n) на сетке
# двигателя (32 522 узла) — 7,9 ГиБ на КАЖДУЮ матрицу, и «Тепловая динамика» падала, не начавшись.


def assemble_source_rhs(space: LagrangeP1Space2D, q_cells: np.ndarray) -> np.ndarray:
    """Вектор объёмного источника: f_i = ∫ q φ_i, q — кусочно-постоянна (n_cells,). ∫_T φ_i=A/3."""
    mesh = space.mesh
    q = np.asarray(q_cells, dtype=float)
    if q.shape != (mesh.n_cells,):
        raise ValueError("q_cells must have shape (n_cells,).")
    cells, _, area = p1_cell_geometry(mesh)
    return np.bincount(cells.ravel(), weights=np.repeat(q * area / 3.0, 3),
                       minlength=space.ndofs)


def assemble_robin_boundary(
    space: LagrangeP1Space2D, h: float
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Граничный член конвекции −k∂T/∂n=h(T−T_amb): возвращает (R, amb_load), где
    R_ij=∫_∂Ω h φ_i φ_j ds (добавить к жёсткости), amb_load_i=∫_∂Ω h φ_i ds (RHS=T_amb·amb_load).
    Краевая масса линейного элемента: (L/6)[[2,1],[1,2]]; ∫_ребро φ_i ds = L/2.
    """
    return assemble_robin_edges(space, space.mesh.boundary_edges(), h)


# --- P-B1: дифференцированные ГУ (h_in≠h_out, разные T_ref) ---

def classify_boundary_edges(space, r_inner: float, r_outer: float):
    """
    Разделить граничные рёбра на (внутренние, внешние) по радиусу середины ребра.
    Внутреннее — ближе к r_inner (расточка статора → рама), внешнее — к r_outer (ротор → среда).
    Радиус от центра сетки (машина центрирована в начале координат).
    """
    mesh = space.mesh
    inner: list = []
    outer: list = []
    for i, j in mesh.boundary_edges():
        mid = 0.5 * (mesh.vertices[i] + mesh.vertices[j])
        r = float(np.hypot(mid[0], mid[1]))
        (inner if abs(r - r_inner) <= abs(r - r_outer) else outer).append((i, j))
    return inner, outer


def assemble_robin_edges(space: LagrangeP1Space2D, edges, h: float):
    """
    Robin-вклад по ЗАДАННОМУ списку рёбер: (R, load), где R_ij=∫h φ_i φ_j (CSR),
    load_i=∫h φ_i (домножить на T_ref для RHS). Краевая масса (L/6)[[2,1],[1,2]].
    """
    mesh = space.mesh
    n = space.ndofs
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    i, j = e[:, 0], e[:, 1]
    L = np.linalg.norm(mesh.vertices[i] - mesh.vertices[j], axis=1)
    diag, off = h * L / 3.0, h * L / 6.0
    R = sp.coo_matrix((np.concatenate([diag, diag, off, off]),
                       (np.concatenate([i, j, i, j]), np.concatenate([i, j, j, i]))),
                      shape=(n, n)).tocsr()
    half = h * L / 2.0
    load = np.bincount(np.concatenate([i, j]), weights=np.concatenate([half, half]), minlength=n)
    return R, load


def assemble_robin_multi(space: LagrangeP1Space2D, specs):
    """
    Дифференцированное ГУ конвекции: specs = список (edges, h, T_ref).
    Возвращает (R, rhs_conv), где R — CSR, а rhs_conv_i = Σ (h·T_ref)∫φ_i УЖЕ включает T_ref
    (для шага неявного Эйлера: A = Cdt+K+R, правая часть += rhs_conv).
    """
    n = space.ndofs
    R = sp.csr_matrix((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)
    for edges, h, T_ref in specs:
        Re, load = assemble_robin_edges(space, edges, float(h))
        R = R + Re
        rhs += float(T_ref) * load
    return R, rhs


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
    K = assemble_stiffness_sparse(space, k)
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


def assemble_capacity(space: LagrangeP1Space2D, c_cells) -> sp.csr_matrix:
    """
    Матрица теплоёмкости C_ij = ∫ c φ_i φ_j (CSR), где c = ρ·c_p [Дж/(м³·K)] — объёмная
    теплоёмкость (скаляр|(n_cells,), разная по регионам: медь/сталь/магнит/воздух).
    Локально (c·A/12)·[[2,1,1],[1,2,1],[1,1,2]]. Структура = матрица масс, взвешенная c.
    """
    c = np.asarray(c_cells, dtype=float)
    if c.ndim != 0 and c.shape != (space.mesh.n_cells,):
        raise ValueError("c_cells must be a scalar or shape (n_cells,).")
    return assemble_mass_sparse(space, c)


class ImplicitEulerThermalStepper:
    """
    Один шаг неявного Эйлера для C·∂T/∂t − div(k∇T) = q + конвекция (Robin):
        (C/dt + K + R)·T^{n+1} = (C/dt)·T^n + f(q) + T_amb·amb_load.
    Источник q подаёт вызывающий: связка магнит↔тепло — ЯВНО, по состоянию в начале шага q^n = q(T^n)
    (docs/math/coupled_problem.md §5, (HE)); такая схема устойчива ровно ниже порога разгона при любом dt.
    Матрицы собираются ОДИН раз (k, c, h, dt постоянны) — шаг переиспользуется как
    нестационарным решателем, так и связкой магнит↔тепло (источник q зависит от T).

    Схема безусловно устойчива (A = C/dt + K + R симметрична положительно определена
    при h>0), поэтому шаг ограничен только точностью, а не устойчивостью.

    Матрицы разрежённые (CSR); A постоянна, поэтому LU-разложение (SuperLU) делается тоже
    ОДИН раз, а шаг — только прямая и обратная подстановка.
    """

    def __init__(self, space: LagrangeP1Space2D, k, capacity, *, dt: float,
                 h: float = 0.0, T_amb: float = 0.0, robin=None):
        if float(dt) <= 0.0:
            raise ValueError("dt must be positive.")
        self.space = space
        self.dt = float(dt)
        self.T_amb = float(T_amb)
        self.K = assemble_stiffness_sparse(space, k)
        self.C = assemble_capacity(space, capacity)
        if robin is not None:
            # Дифференцированные ГУ (P-B1): готовые (R, rhs_conv), rhs_conv уже с T_ref.
            R, robin_rhs = robin
            self.R = sp.csr_matrix(R, dtype=float)
            self.robin_rhs = np.asarray(robin_rhs, dtype=float)
        else:
            # Скалярное однородное ГУ (обратная совместимость): rhs = T_amb·∫h φ.
            self.R, amb_load = assemble_robin_boundary(space, float(h))
            self.robin_rhs = float(T_amb) * amb_load
        self.Cdt = (self.C / self.dt).tocsr()
        self.A = (self.Cdt + self.K + self.R).tocsr()
        self._lu = splu(self.A.tocsc())
        # Столбцовые суммы 1ᵀC и 1ᵀR для энергобаланса — тоже один раз.
        self._c_sum = np.asarray(self.C.sum(axis=0)).ravel()
        self._r_sum = np.asarray(self.R.sum(axis=0)).ravel()

    def step(self, T: np.ndarray, q_cells: np.ndarray) -> np.ndarray:
        """Поле на следующем шаге по текущему T и источнику потерь q [Вт/м³] (по ячейкам)."""
        f = assemble_source_rhs(self.space, np.asarray(q_cells, dtype=float))
        return self._lu.solve(self.Cdt @ np.asarray(T, dtype=float) + f + self.robin_rhs)

    # --- диагностика энергобаланса (оракул связки) ---
    def stored_energy(self, T) -> float:
        """Запасённая тепловая энергия ∫ c·T dV = 1ᵀ·C·T [Дж/м] (на единицу длины)."""
        return float(np.asarray(T, dtype=float) @ self._c_sum)

    def convective_outflow(self, T) -> float:
        """Отвод конвекцией ∫ h(T−T_ref) ds = 1ᵀR·T − 1ᵀ·rhs_conv [Вт/м] (обобщённо на дифф. ГУ)."""
        Tv = np.asarray(T, dtype=float)
        return float(Tv @ self._r_sum) - float(self.robin_rhs.sum())


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
