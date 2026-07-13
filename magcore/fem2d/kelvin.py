from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.fem2d.assembly import (
    assemble_current_rhs,
    assemble_magnetization_rhs,
    assemble_stiffness,
)
from magcore.fem2d.mesh_generators import DiskMesh
from magcore.fem2d.post import reconstruct_B_on_cells
from magcore.fem2d.solver import apply_dirichlet, solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.nonlinear.picard import resolve_magnetization, run_picard_fixed_point


@dataclass(frozen=True, slots=True)
class KelvinResult:
    a: np.ndarray            # (N,) A_z на РЕАЛЬНОМ диске
    B_cells: np.ndarray      # (n_cells, 2) — B на реальном диске


@dataclass(frozen=True, slots=True)
class KelvinPicardResult:
    a: np.ndarray                 # (N,) A_z на реальном диске
    B_cells: np.ndarray           # (n_cells, 2)
    H_cells: np.ndarray           # (n_cells, 2) — H=νB − νB_r (для risk-map)
    nu_cells: np.ndarray          # (n_cells,) — ν финальной сборки
    nu_br_cells: np.ndarray       # (n_cells, 2) — финальный источник ν·B_r
    n_iterations: int
    converged: bool
    rel_change_history: tuple[float, ...]


def _image_global_map(disk: DiskMesh) -> tuple[np.ndarray, int]:
    """
    Отображение узел-образа → глобальный DOF: граничные узлы РАЗДЕЛЕНЫ с реальным
    диском (склейка → сами себя в [0,N)); внутренние узлы образа получают новые
    индексы [N, 2N−n_theta). Возвращает (img_global(N,), total).
    """
    N = disk.mesh.n_vertices
    boundary = {int(b) for b in disk.boundary_nodes}
    img = np.empty(N, dtype=int)
    nxt = N
    for i in range(N):
        if i in boundary:
            img[i] = i
        else:
            img[i] = nxt
            nxt += 1
    return img, nxt


def solve_kelvin_magnetostatic(
    disk: DiskMesh,
    *,
    nu_real,
    j_fn=None,
    magnetization=None,
    nu0: float = 1.0,
    quadrature_order: int = 5,
) -> KelvinResult:
    """
    Планарная магнитостатика в ОТКРЫТОЙ области методом Kelvin-трансформации (2D-B).

    Реальный диск (физика ν, источники) склеивается по граничной окружности с
    образ-диском (воздух ν₀), представляющим внешность через инверсию R=a²/r.
    В 2D инверсия КОНФОРМНА ⇒ образ решает тот же Лаплас (ν₀·Δ), а на стыке
    ∂Â/∂R=−∂A/∂r ⇒ простое разделение граничных DOF даёт точную непрерывность
    потока. Центр образа = ∞ ⇒ пиннится A=0 (условие убывания). Точно, чистый FEM.

    nu_real : (n_cells,) физическая ν реального диска (воздух/сталь/магнит).
    j_fn : callable(x)->float внеплоскостной ток; magnetization : (n_cells,2) ν·B_r.
    """
    mesh = disk.mesh
    space = LagrangeP1Space2D(mesh)
    N = mesh.n_vertices
    n_cells = mesh.n_cells
    img, total = _image_global_map(disk)

    Kr = assemble_stiffness(space, nu_real)                          # физика
    Ki = assemble_stiffness(space, np.full(n_cells, float(nu0)))     # образ (воздух ν₀)

    K = np.zeros((total, total), dtype=float)
    K[:N, :N] += Kr
    K[np.ix_(img, img)] += Ki      # склейка: граничные DOF накапливают оба вклада (поток)

    f = np.zeros(total, dtype=float)
    if j_fn is not None:
        f[:N] += assemble_current_rhs(space, j_fn, quadrature_order=quadrature_order)
    if magnetization is not None:
        nu_br = np.asarray(magnetization, dtype=float)
        if nu_br.shape != (n_cells, 2):
            raise ValueError("magnetization must have shape (n_cells, 2).")
        f[:N] += assemble_magnetization_rhs(space, nu_br)

    gc = int(img[disk.center_node])                                  # образ-центр = ∞
    K_bc, f_bc = apply_dirichlet(K, f, [gc], 0.0)
    x = solve_scalar(K_bc, f_bc)
    a = x[:N]
    return KelvinResult(a=a, B_cells=reconstruct_B_on_cells(space, a))


def solve_nonlinear_kelvin_2d_picard(
    disk: DiskMesh,
    nu_of_B,
    *,
    nu_init,
    j_fn=None,
    magnetization=None,
    nu0: float = 1.0,
    max_iter: int = 80,
    tol: float = 1.0e-6,
    relaxation: float = 0.5,
    quadrature_order: int = 5,
) -> KelvinPicardResult:
    """
    Нелинейный Picard для планарной задачи в ОТКРЫТОЙ области (Kelvin) — магнит с
    состоянием (колено) и/или нелинейная сталь в свободном пространстве.

    Переиспользует РАЗМЕРНО-НЕЗАВИСИМОЕ ядро `run_picard_fixed_point` (2D-0) — тот же
    цикл, что драйвит 3D и планарный box; backend-шаг собирает Kelvin-связку с
    замороженной ν и источником магнита, решает, возвращает B. Демонстрирует демаг в
    открытой 2D-области. Образ-блок (воздух ν₀) ν-независим ⇒ собирается ОДИН раз.

    magnetization : None | (n_cells,2) ν·B_r | callable(B,H,ν)->(n_cells,2)
        (магнит с коленом — напр. `hybrid.magnet_demag.MagnetDemagPolicy(..., axis=(1,0))`).
    """
    mesh = disk.mesh
    space = LagrangeP1Space2D(mesh)
    N = mesh.n_vertices
    n_cells = mesh.n_cells
    img, total = _image_global_map(disk)
    gc = int(img[disk.center_node])

    mag_fn = resolve_magnetization(magnetization, n_cells, dim=2)
    f_current = (
        np.zeros(N, dtype=float)
        if j_fn is None
        else assemble_current_rhs(space, j_fn, quadrature_order=quadrature_order)
    )
    Ki = assemble_stiffness(space, np.full(n_cells, float(nu0)))     # образ (воздух ν₀) — const

    state: dict[str, object] = {
        "a": np.zeros(N, dtype=float),
        "B": np.zeros((n_cells, 2), dtype=float),
        "H": np.zeros((n_cells, 2), dtype=float),
        "nu_br": np.zeros((n_cells, 2), dtype=float),
    }

    def step(nu_frozen: np.ndarray) -> np.ndarray:
        nu_br = np.asarray(mag_fn(state["B"], state["H"], nu_frozen), dtype=float)
        if nu_br.shape != (n_cells, 2):
            raise ValueError("magnetization callable must return shape (n_cells, 2).")
        Kr = assemble_stiffness(space, nu_frozen)
        K = np.zeros((total, total), dtype=float)
        K[:N, :N] += Kr
        K[np.ix_(img, img)] += Ki
        f = np.zeros(total, dtype=float)
        f[:N] += f_current + assemble_magnetization_rhs(space, nu_br)
        K_bc, f_bc = apply_dirichlet(K, f, [gc], 0.0)
        a = solve_scalar(K_bc, f_bc)[:N]
        B = reconstruct_B_on_cells(space, a)
        H = nu_frozen[:, None] * B - nu_br
        state.update(a=a, B=B, H=H, nu_br=nu_br)
        return B

    loop = run_picard_fixed_point(
        nu_init=nu_init, nu_of_B=nu_of_B, step=step,
        max_iter=max_iter, tol=tol, relaxation=relaxation,
    )

    return KelvinPicardResult(
        a=state["a"], B_cells=loop.B_cells, H_cells=state["H"],
        nu_cells=loop.nu_cells, nu_br_cells=state["nu_br"],
        n_iterations=loop.n_iterations, converged=loop.converged,
        rel_change_history=loop.rel_change_history,
    )
