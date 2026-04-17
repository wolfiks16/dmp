from __future__ import annotations

import numpy as np

from magcore.mesh.mesh import TetraMesh, oriented_tetra_volume6


def _orient_tetra(vertices: np.ndarray, tet: list[int]) -> list[int]:
    tet_arr = np.asarray(tet, dtype=int)
    vol6 = oriented_tetra_volume6(vertices[tet_arr])
    if vol6 > 0.0:
        return tet
    return [tet[0], tet[2], tet[1], tet[3]]


def build_structured_box_tetra_mesh(
    nx: int,
    ny: int,
    nz: int,
    *,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
    zlim: tuple[float, float] = (0.0, 1.0),
) -> TetraMesh:
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny and nz must be at least 1.")

    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    z0, z1 = float(zlim[0]), float(zlim[1])

    if not (x1 > x0 and y1 > y0 and z1 > z0):
        raise ValueError("Each coordinate interval must satisfy upper > lower.")

    hx = (x1 - x0) / nx
    hy = (y1 - y0) / ny
    hz = (z1 - z0) / nz

    def vid(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    vertices = []
    for k in range(nz + 1):
        z = z0 + k * hz
        for j in range(ny + 1):
            y = y0 + j * hy
            for i in range(nx + 1):
                x = x0 + i * hx
                vertices.append([x, y, z])
    vertices = np.asarray(vertices, dtype=float)

    cells: list[list[int]] = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v000 = vid(i, j, k)
                v100 = vid(i + 1, j, k)
                v010 = vid(i, j + 1, k)
                v110 = vid(i + 1, j + 1, k)
                v001 = vid(i, j, k + 1)
                v101 = vid(i + 1, j, k + 1)
                v011 = vid(i, j + 1, k + 1)
                v111 = vid(i + 1, j + 1, k + 1)

                local_tets = [
                    [v000, v100, v110, v111],
                    [v000, v100, v101, v111],
                    [v000, v001, v101, v111],
                    [v000, v001, v011, v111],
                    [v000, v010, v011, v111],
                    [v000, v010, v110, v111],
                ]

                for tet in local_tets:
                    cells.append(_orient_tetra(vertices, tet))

    return TetraMesh(vertices=vertices, cells=np.asarray(cells, dtype=int))


def build_structured_unit_cube_tetra_mesh(n: int) -> TetraMesh:
    return build_structured_box_tetra_mesh(
        nx=n,
        ny=n,
        nz=n,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        zlim=(0.0, 1.0),
    )


def build_symmetric_unit_cube_tetra_mesh() -> TetraMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
            [0.5, 0.5, 0.5],  # 8
            [0.0, 0.5, 0.5],  # 9
            [1.0, 0.5, 0.5],  # 10
            [0.5, 0.0, 0.5],  # 11
            [0.5, 1.0, 0.5],  # 12
            [0.5, 0.5, 0.0],  # 13
            [0.5, 0.5, 1.0],  # 14
        ],
        dtype=float,
    )

    face_to_center = {
        "x0": 9,
        "x1": 10,
        "y0": 11,
        "y1": 12,
        "z0": 13,
        "z1": 14,
    }

    faces = {
        "x0": [0, 3, 7, 4],
        "x1": [1, 5, 6, 2],
        "y0": [0, 4, 5, 1],
        "y1": [3, 2, 6, 7],
        "z0": [0, 1, 2, 3],
        "z1": [4, 7, 6, 5],
    }

    cells: list[list[int]] = []

    for face_name, corners in faces.items():
        fc = face_to_center[face_name]

        face_tris = [
            [corners[0], corners[1], fc],
            [corners[1], corners[2], fc],
            [corners[2], corners[3], fc],
            [corners[3], corners[0], fc],
        ]

        for tri in face_tris:
            tet = [tri[0], tri[1], tri[2], 8]
            cells.append(_orient_tetra(vertices, tet))

    return TetraMesh(vertices=vertices, cells=np.asarray(cells, dtype=int))