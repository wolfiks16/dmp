import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import eigh

from magcore.fem2d.assembly import (
    assemble_mass,
    assemble_mass_sparse,
    assemble_stiffness,
    assemble_stiffness_sparse,
)
from magcore.fem2d.kelvin import _glue, _image_global_map
from magcore.fem2d.mesh import triangle_area
from magcore.fem2d.mesh_generators import build_disk_tri_mesh, build_structured_rectangle_tri_mesh
from magcore.fem2d.runaway import runaway_threshold
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import (
    ImplicitEulerThermalStepper,
    assemble_capacity,
    assemble_robin_boundary,
    assemble_robin_edges,
    assemble_robin_multi,
    assemble_source_rhs,
)

# Тепловые матрицы — РАЗРЕЖЁННЫЕ: плотная (n,n) на сетке двигателя (32 522 узла) — 7,9 ГиБ на
# каждую, и «Тепловая динамика» падала, не начавшись. Оракул — не «работает без ошибки», а
# совпадение с прежними ПЛОТНЫМИ формулами (перенесены сюда как эталон) и с решением той же
# системы плотным LAPACK. Допуски выведены из округления, а не подобраны:
#   * сборка: значение каждого вклада то же, меняется только порядок сложения; в элемент
#     матрицы P1 сходится ≤ 6 вкладов ⇒ |Δ| ≤ 5·eps·|элемент|; берём 16·eps·max|M|;
#   * сумма N слагаемых (тождества вида Σ_ij C_ij = ∫c): |Δ| ≤ N·eps·Σ|слагаемых|;
#   * решение: оба решателя обратно устойчивы ⇒ |Δx| ≲ n·κ(A)·eps·|x| (n — запас на константы);
#   * разрежённость: у плоской триангуляции рёбер E ≤ 3V − 6 (формула Эйлера) ⇒ ненулей
#     V + 2E < 7V — строгая граница, а не подобранная.

EPS = np.finfo(float).eps


def _dense_capacity(space, c):
    """Прежняя плотная `assemble_capacity` — эталон."""
    mesh = space.mesh
    C = np.zeros((space.ndofs, space.ndofs))
    base = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0
    for cell in range(mesh.n_cells):
        idx = mesh.cell_vertex_indices(cell)
        Ce = c[cell] * triangle_area(mesh.cell_vertices(cell)) * base
        for a in range(3):
            for b in range(3):
                C[idx[a], idx[b]] += Ce[a, b]
    return C


def _dense_robin(space, edges, h):
    """Прежняя плотная `assemble_robin_edges` — эталон."""
    mesh = space.mesh
    R = np.zeros((space.ndofs, space.ndofs))
    load = np.zeros(space.ndofs)
    for i, j in edges:
        L = float(np.linalg.norm(mesh.vertices[i] - mesh.vertices[j]))
        R[i, i] += h * L / 3.0
        R[j, j] += h * L / 3.0
        R[i, j] += h * L / 6.0
        R[j, i] += h * L / 6.0
        load[i] += h * L / 2.0
        load[j] += h * L / 2.0
    return R, load


def _dense_source(space, q):
    """Прежняя поячеечная `assemble_source_rhs` — эталон."""
    mesh = space.mesh
    f = np.zeros(space.ndofs)
    for c in range(mesh.n_cells):
        f[list(mesh.cell_vertex_indices(c))] += q[c] * triangle_area(mesh.cell_vertices(c)) / 3.0
    return f


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _space(nx=7, ny=5):
    mesh = build_structured_rectangle_tri_mesh(nx, ny, x0=-0.03, x1=0.04, y0=0.0, y1=0.025)
    return mesh, LagrangeP1Space2D(mesh)


def test_capacity_and_mass_are_sparse_and_equal_dense_formula():
    mesh, space = _space()
    n = space.ndofs
    c = np.random.default_rng(3).uniform(1.0e4, 3.5e6, mesh.n_cells)   # регионы: воздух…сталь

    C = assemble_capacity(space, c)
    assert sp.issparse(C) and C.nnz < 7 * n
    assert _rel(C.toarray(), _dense_capacity(space, c)) <= 16 * EPS

    M = assemble_mass_sparse(space)
    assert sp.issparse(M) and M.nnz < 7 * n
    assert _rel(M.toarray(), assemble_mass(space)) <= 16 * EPS
    # Скалярная c ≡ c·M; сумма всех элементов — ∫c (Σφ_i ≡ 1): независимое тождество.
    assert _rel(assemble_capacity(space, 2.5e6).toarray(), 2.5e6 * M.toarray()) <= 16 * EPS
    areas = np.array([mesh.cell_area(i) for i in range(mesh.n_cells)])
    n_terms = C.nnz + mesh.n_cells
    assert abs(float(C.sum()) - float(c @ areas)) <= n_terms * EPS * float(c @ areas)
    assert abs(float(M.sum()) - float(areas.sum())) <= n_terms * EPS * float(areas.sum())


def test_capacity_rejects_wrong_shape():
    mesh, space = _space()
    for bad in (np.ones(mesh.n_cells + 1), np.ones((mesh.n_cells, 2))):
        with pytest.raises(ValueError, match="c_cells"):
            assemble_capacity(space, bad)


def test_robin_is_sparse_and_equals_dense_formula():
    mesh, space = _space()
    n = space.ndofs
    h = 37.0
    edges = list(mesh.boundary_edges())
    R, amb = assemble_robin_boundary(space, h)
    Rd, ambd = _dense_robin(space, edges, h)
    assert sp.issparse(R) and R.nnz <= n + 2 * len(edges)
    assert _rel(R.toarray(), Rd) <= 16 * EPS
    assert _rel(amb, ambd) <= 16 * EPS
    # Независимое тождество: Σ_ij R_ij = ∫ h·(Σφ)² ds = h·P и Σ amb = h·P (P — периметр).
    P = sum(float(np.linalg.norm(mesh.vertices[i] - mesh.vertices[j])) for i, j in edges)
    n_terms = R.nnz + len(edges)
    assert abs(float(R.sum()) - h * P) <= n_terms * EPS * h * P
    assert abs(float(amb.sum()) - h * P) <= n_terms * EPS * h * P

    # Дифференцированные ГУ = сумма вкладов по частям границы (P-B1).
    left = [e for e in edges if mesh.vertices[list(e)][:, 0].mean() < 0.0]
    right = [e for e in edges if mesh.vertices[list(e)][:, 0].mean() >= 0.0]
    Rm, rhs = assemble_robin_multi(space, [(left, 5.0, 15.0), (right, 20.0, 45.0)])
    Rl, ll = _dense_robin(space, left, 5.0)
    Rr, lr = _dense_robin(space, right, 20.0)
    assert sp.issparse(Rm)
    assert _rel(Rm.toarray(), Rl + Rr) <= 16 * EPS
    assert _rel(rhs, 15.0 * ll + 45.0 * lr) <= 16 * EPS


def test_robin_on_empty_edge_list_is_zero():
    # Негатив: часть границы без рёбер (напр., машина без внутренней расточки) — ноль, а не сбой.
    _, space = _space()
    R, load = assemble_robin_edges(space, [], 50.0)
    assert R.shape == (space.ndofs, space.ndofs) and R.nnz == 0
    assert load.shape == (space.ndofs,) and not load.any()


def test_source_rhs_equals_cellwise_formula():
    mesh, space = _space()
    q = np.random.default_rng(5).uniform(0.0, 4.0e6, mesh.n_cells)
    assert _rel(assemble_source_rhs(space, q), _dense_source(space, q)) <= 16 * EPS


def test_stepper_matches_dense_lapack_solve():
    # Шаг неявного Эйлера на CSR + однократное LU = тот же шаг на плотных прежних формулах.
    mesh, space = _space(8, 6)
    n = space.ndofs
    rng = np.random.default_rng(11)
    k = rng.uniform(0.5, 25.0, mesh.n_cells)
    c = rng.uniform(1.0e4, 3.5e6, mesh.n_cells)
    q = rng.uniform(0.0, 5.0e6, mesh.n_cells)
    h, T_amb, dt = 400.0, 150.0, 2.0
    st = ImplicitEulerThermalStepper(space, k, c, dt=dt, h=h, T_amb=T_amb)
    assert all(sp.issparse(M) for M in (st.K, st.C, st.R, st.A))

    Cd = _dense_capacity(space, c)
    Rd, ambd = _dense_robin(space, list(mesh.boundary_edges()), h)
    A = Cd / dt + assemble_stiffness(space, k) + Rd
    T0 = rng.uniform(140.0, 160.0, n)
    T_ref = np.linalg.solve(A, Cd / dt @ T0 + _dense_source(space, q) + T_amb * ambd)

    T = st.step(T0, q)
    bound = n * np.linalg.cond(A, np.inf) * EPS * np.max(np.abs(T_ref))
    assert np.max(np.abs(T - T_ref)) <= bound
    # Диагностика энергобаланса — те же 1ᵀC·T и 1ᵀR·T − 1ᵀrhs, что и с плотными матрицами
    # (столбцовые суммы ≤ 7 слагаемых + скалярное произведение n слагаемых, дважды).
    n_terms = 2 * n + 7
    E_ref = float(Cd.sum(axis=0) @ T)
    assert abs(st.stored_energy(T) - E_ref) <= n_terms * EPS * E_ref
    flux = float(Rd.sum(axis=0) @ T)
    out_ref = flux - T_amb * float(ambd.sum())
    assert abs(st.convective_outflow(T) - out_ref) <= n_terms * EPS * flux


def test_stepper_accepts_dense_robin_from_old_callers():
    # Обратная совместимость: готовое (R, rhs) могло прийти плотным — результат тот же.
    mesh, space = _space()
    k, c = np.full(mesh.n_cells, 2.0), np.full(mesh.n_cells, 2.0e5)
    R, rhs = assemble_robin_multi(space, [(list(mesh.boundary_edges()), 12.0, 20.0)])
    s_sparse = ImplicitEulerThermalStepper(space, k, c, dt=0.5, robin=(R, rhs))
    s_dense = ImplicitEulerThermalStepper(space, k, c, dt=0.5, robin=(R.toarray(), rhs))
    T = np.full(space.ndofs, 20.0)
    q = np.full(mesh.n_cells, 300.0)
    assert np.array_equal(s_sparse.step(T, q), s_dense.step(T, q))


def test_runaway_threshold_equals_dense_generalized_eigenvalue():
    # Порог разгона: одно значение сдвигом-обращением (eigsh) = минимум полного плотного спектра
    # той же задачи (K+R)v = s·M·v. Погрешность плотного метода — eps·λ_max·κ(M) (обратная
    # устойчивость обобщённой задачи); запас 100 на константы.
    disk = build_disk_tri_mesh(1.0, 12, 48)
    space = LagrangeP1Space2D(disk.mesh)
    k = np.where(np.arange(disk.mesh.n_cells) % 3 == 0, 3.0, 1.0)     # неоднородная k
    h = 3.0
    Rd, _ = _dense_robin(space, list(disk.mesh.boundary_edges()), h)
    A = assemble_stiffness(space, k) + Rd
    M = assemble_mass(space)
    w = eigh(A, M, eigvals_only=True)
    s_crit = runaway_threshold(space, k, h)
    assert w[0] > 0.0
    assert abs(s_crit - w[0]) <= 100 * EPS * w[-1] * np.linalg.cond(M)


def test_kelvin_glue_equals_dense_composition():
    # Склейка реального диска с образом: CSR через COO = прежняя плотная сборка
    # K[:N,:N] += Kr; K[img,img] += Ki (граничные DOF общие и накапливают оба вклада).
    disk = build_disk_tri_mesh(2.0, 5, 20)
    space = LagrangeP1Space2D(disk.mesh)
    N = disk.mesh.n_vertices
    img, total = _image_global_map(disk)
    nu = np.random.default_rng(2).uniform(0.5, 2.0, disk.mesh.n_cells)
    Kr = assemble_stiffness_sparse(space, nu)
    Ki = assemble_stiffness_sparse(space, np.ones(disk.mesh.n_cells))
    K = _glue(Kr, Ki, img, total)
    Kd = np.zeros((total, total))
    Kd[:N, :N] += Kr.toarray()
    Kd[np.ix_(img, img)] += Ki.toarray()
    assert sp.issparse(K) and K.shape == (total, total)
    assert _rel(K.toarray(), Kd) <= 16 * EPS
