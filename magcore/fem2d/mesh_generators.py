from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.fem2d.mesh import TriangleMesh


def build_structured_rectangle_tri_mesh(
    nx: int,
    ny: int,
    *,
    x0: float = 0.0,
    x1: float = 1.0,
    y0: float = 0.0,
    y1: float = 1.0,
) -> TriangleMesh:
    """
    Структурированная триангуляция прямоугольника [x0,x1]×[y0,y1]: сетка (nx×ny)
    четырёхугольников, каждый разбит на 2 треугольника (CCW). Аналог
    `mesh_generators.build_structured_box_tetra_mesh` для 2D — опорная сетка для MMS.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1.")
    if not (x1 > x0 and y1 > y0):
        raise ValueError("require x1 > x0 and y1 > y0.")

    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")  # shape (ny+1, nx+1)
    vertices = np.column_stack((gx.ravel(), gy.ravel()))

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    cells: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v11 = vid(i + 1, j + 1)
            v01 = vid(i, j + 1)
            # Оба треугольника CCW (проверено ориентацией в TriangleMesh.validate).
            cells.append((v00, v10, v11))
            cells.append((v00, v11, v01))

    return TriangleMesh(vertices=vertices, cells=np.asarray(cells, dtype=int))


@dataclass(frozen=True)
class DiskMesh:
    """
    Структурированная триангуляция круга концентрическими кольцами.
    Знает центральный узел (=∞ под Kelvin-инверсией) и упорядоченные по углу узлы
    граничной окружности (для склейки реального и образ-диска, 2D-B).
    """

    mesh: TriangleMesh
    center_node: int
    boundary_nodes: np.ndarray     # (n_theta,) индексы внешнего кольца, по возрастанию θ
    boundary_angles: np.ndarray    # (n_theta,) углы θ_k
    radius: float


def build_disk_tri_mesh(
    radius: float, n_rings: int, n_theta: int, *, cx: float = 0.0, cy: float = 0.0
) -> DiskMesh:
    """
    Круг радиуса `radius`: узел-центр + `n_rings` колец по `n_theta` узлов.
    Узел (кольцо j=1..n_rings, угол k=0..n_theta−1) имеет индекс 1+(j−1)·n_theta+k;
    центр — индекс 0. Все треугольники CCW (проверяется TriangleMesh.validate).
    """
    if n_rings < 1 or n_theta < 3:
        raise ValueError("require n_rings >= 1 and n_theta >= 3.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    thetas = 2.0 * np.pi * np.arange(n_theta) / n_theta
    verts: list[tuple[float, float]] = [(cx, cy)]
    for j in range(1, n_rings + 1):
        rj = radius * j / n_rings
        for k in range(n_theta):
            verts.append((cx + rj * np.cos(thetas[k]), cy + rj * np.sin(thetas[k])))

    def node(j: int, k: int) -> int:
        return 1 + (j - 1) * n_theta + (k % n_theta)

    cells: list[tuple[int, int, int]] = []
    # Центральный веер (центр → кольцо 1).
    for k in range(n_theta):
        cells.append((0, node(1, k), node(1, k + 1)))
    # Кольцевые пояса j → j+1: квадрат (a=inner_k, b=inner_{k+1}, c=outer_k, d=outer_{k+1}).
    for j in range(1, n_rings):
        for k in range(n_theta):
            a, b = node(j, k), node(j, k + 1)
            c, d = node(j + 1, k), node(j + 1, k + 1)
            cells.append((a, c, d))
            cells.append((a, d, b))

    mesh = TriangleMesh(vertices=np.asarray(verts, dtype=float), cells=np.asarray(cells, dtype=int))
    boundary_nodes = np.array([node(n_rings, k) for k in range(n_theta)], dtype=int)
    return DiskMesh(
        mesh=mesh,
        center_node=0,
        boundary_nodes=boundary_nodes,
        boundary_angles=thetas.copy(),
        radius=float(radius),
    )
