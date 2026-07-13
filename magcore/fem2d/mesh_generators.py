from __future__ import annotations

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
