from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh import TetraMesh


def match_periodic_vertices(
    mesh: TetraMesh,
    on_slave,
    on_master,
    transform,
    tol: float = 1.0e-9,
) -> dict[int, int]:
    """
    Сопоставить вершины «ведомой» (slave) границы их образам на «ведущей» (master)
    под отображением периодичности `transform` (slave-координата → ожидаемая master-
    координата). Возвращает {slave_vertex_index: master_vertex_index}.

    `on_slave(coord)->bool`, `on_master(coord)->bool` — предикаты выбора граничных
    вершин по координате (напр. x≈0 / x≈1, или θ≈0 / θ≈θ_seg). `transform` для
    трансляции = `x→x+L·ê`, для поворота = `x→R·x`. Соответствие — по ближайшей
    master-вершине к `transform(slave)`; падает, если ближе `tol` нет (несовпадающие
    сетки на разрезах).
    """
    V = np.asarray(mesh.vertices, dtype=float)
    slave = [i for i in range(V.shape[0]) if bool(on_slave(V[i]))]
    master = np.array([i for i in range(V.shape[0]) if bool(on_master(V[i]))], dtype=int)
    if master.size == 0:
        raise ValueError("No master-boundary vertices selected.")
    master_xyz = V[master]

    pairing: dict[int, int] = {}
    used: set[int] = set()
    for s in slave:
        target = np.asarray(transform(V[s]), dtype=float)
        d = np.linalg.norm(master_xyz - target[None, :], axis=1)
        j = int(np.argmin(d))
        if d[j] > tol:
            raise ValueError(
                f"Slave vertex {s} at {V[s]} has no master match within tol "
                f"(nearest dist {d[j]:.3e}); meshes not periodic-conforming."
            )
        m = int(master[j])
        if m in used:
            raise ValueError(f"Master vertex {m} matched by two slaves (non-bijective).")
        used.add(m)
        pairing[s] = m
    if len(pairing) != len(slave):
        raise ValueError("Slave/master vertex counts differ.")
    return pairing


@dataclass(frozen=True)
class PeriodicReduction:
    """
    Редукция седловой смешанной системы по (анти)периодическим связям через матрицу
    пролонгации `T` (n_full × n_red): `x_full = T x_red`, редуцированная система
    `TᵀMT x_red = Tᵀb` (симметрия и блочная a/p-структура сохраняются, т.к. T блочно-
    диагональна по a/p). `x[slave] = sign · x[master]` (sign = ±1·orient для рёбер,
    ±1 для p-вершин; знак периодичности pf = +1 период / −1 анти-период).
    """

    T: np.ndarray
    n_a: int
    n_p: int
    slave_dofs: np.ndarray
    n_full: int
    n_red: int
    reduced_of: np.ndarray  # (n_full,) глоб. DOF → индекс в редуц. системе, −1 для slave


def build_periodic_reduction(
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    vertex_map: dict[int, int],
    *,
    antiperiodic: bool = False,
    excluded: set[int] | None = None,
) -> PeriodicReduction:
    """
    Построить `PeriodicReduction` для смешанного вектора DOF `[a (рёбра Неделек) ;
    p (вершины P1)]`. Связи:
      • ребро (a,b)⊂slave, a<b: `dof_slave = pf·orient · dof_master`, master-ребро =
        canonical(φ(a),φ(b)), orient=+1 если φ(a)<φ(b) иначе −1 (сохраняет ли φ порядок);
      • вершина s⊂slave (p-блок): `p_slave = pf · p_master`.
    Поддерживается ОДНА пара граней (master-DOF не может быть сам slave — цепочки/углы
    двух периодик не поддержаны; падает при нарушении).
    """
    pf = -1.0 if antiperiodic else 1.0
    excl = set() if excluded is None else set(int(e) for e in excluded)
    n_a = int(vector_space.ndofs)
    n_p = int(scalar_space.ndofs)
    n = n_a + n_p
    e2d = vector_space.edge_to_dof_map()

    slave_g: list[int] = []
    master_g: list[int] = []
    coeff: list[float] = []

    # рёбра slave-грани (обе вершины в vertex_map); пропускаем конфликтные (excluded —
    # напр. рёбра на пересечении периодической и Dirichlet-граней → отдаём Dirichlet).
    for (a, b), dof in e2d.items():
        if a in vertex_map and b in vertex_map:
            fa, fb = vertex_map[a], vertex_map[b]
            m_edge = (fa, fb) if fa < fb else (fb, fa)
            if m_edge not in e2d:
                raise ValueError(f"Master edge {m_edge} for slave edge {(a, b)} not in mesh.")
            m_dof = int(e2d[m_edge])
            if int(dof) in excl or m_dof in excl:
                continue
            orient = 1.0 if fa < fb else -1.0
            slave_g.append(int(dof))
            master_g.append(m_dof)
            coeff.append(pf * orient)

    # вершины slave-грани (p-блок), глобальный индекс = n_a + vertex
    for s, m in vertex_map.items():
        gs, gm = n_a + int(s), n_a + int(m)
        if gs in excl or gm in excl:
            continue
        slave_g.append(gs)
        master_g.append(gm)
        coeff.append(pf)

    slave_set = set(slave_g)
    if len(slave_set) != len(slave_g):
        raise ValueError("Duplicate slave DOF in periodic constraints.")
    for m in master_g:
        if m in slave_set:
            raise ValueError("Master DOF is itself a slave (chained/corner periodicity unsupported).")

    retained = [i for i in range(n) if i not in slave_set]
    red_index = {g: k for k, g in enumerate(retained)}
    n_red = len(retained)

    reduced_of = np.full(n, -1, dtype=int)
    T = np.zeros((n, n_red), dtype=float)
    for g in retained:
        T[g, red_index[g]] = 1.0
        reduced_of[g] = red_index[g]
    for s, m, c in zip(slave_g, master_g, coeff):
        T[s, red_index[m]] = c

    return PeriodicReduction(
        T=T,
        n_a=n_a,
        n_p=n_p,
        slave_dofs=np.array(sorted(slave_set), dtype=int),
        n_full=n,
        n_red=n_red,
        reduced_of=reduced_of,
    )


def build_periodic_with_dirichlet(
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    vertex_map: dict[int, int],
    dirichlet_global: set[int] | list[int] | np.ndarray,
    *,
    antiperiodic: bool = False,
) -> tuple[PeriodicReduction, np.ndarray]:
    """
    Удобный сбор для смешанной периодической задачи С Dirichlet на прочих гранях.
    `dirichlet_global` — глобальные DOF (рёбра + `n_a+вершина`), фиксируемые Dirichlet.
    Угловое правило: эти DOF исключаются из периодики (`excluded`), т.е. на пересечении
    периодической и Dirichlet-граней побеждает Dirichlet. Возвращает `(reduction,
    dirichlet_reduced_dofs)` — индексы Dirichlet-DOF в РЕДУЦИРОВАННОЙ системе (для
    `solve_periodic_nonlinear_mixed_picard`).
    """
    excl = set(int(g) for g in dirichlet_global)
    red = build_periodic_reduction(
        vector_space, scalar_space, vertex_map, antiperiodic=antiperiodic, excluded=excl
    )
    dir_reduced = sorted({int(red.reduced_of[g]) for g in excl if red.reduced_of[g] >= 0})
    return red, np.array(dir_reduced, dtype=int)


def reduce_system(
    M: np.ndarray, b: np.ndarray, reduction: PeriodicReduction
) -> tuple[np.ndarray, np.ndarray]:
    """Редуцировать `(M,b)` → `(TᵀMT, Tᵀb)` по периодическим связям."""
    T = reduction.T
    if M.shape != (reduction.n_full, reduction.n_full):
        raise ValueError("M shape does not match reduction.n_full.")
    if b.shape != (reduction.n_full,):
        raise ValueError("b shape does not match reduction.n_full.")
    return T.T @ M @ T, T.T @ b


def expand_solution(x_red: np.ndarray, reduction: PeriodicReduction) -> np.ndarray:
    """Восстановить полный вектор `x_full = T x_red` (ведомые DOF = sign·ведущие)."""
    x_red = np.asarray(x_red, dtype=float)
    if x_red.shape != (reduction.n_red,):
        raise ValueError("x_red shape does not match reduction.n_red.")
    return reduction.T @ x_red
