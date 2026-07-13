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


@dataclass(frozen=True, slots=True)
class KelvinResult:
    a: np.ndarray            # (N,) A_z на РЕАЛЬНОМ диске
    B_cells: np.ndarray      # (n_cells, 2) — B на реальном диске


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
