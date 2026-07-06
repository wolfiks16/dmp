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


def build_ball_tetra_mesh(
    n: int,
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> TetraMesh:
    """
    Тетраэдральная сетка ШАРА через гладкое отображение куба `[−1,1]³ → шар`
    (гомеоморфизм L∞→L2):

        x_ball = x · (‖x‖_∞ / ‖x‖₂) · radius     (центр → центр).

    Отображение сохраняет радиальное направление (`x/‖x‖₂`) и переводит куб-оболочку
    `‖x‖_∞=c` в шар-оболочку `|x_ball|=c·radius` ⇒ граница куба → сфера радиуса `radius`.
    Переиспользует структурированную box-сетку и переориентирует ячейки (отображение
    меняет знак объёма у части тетраэдров). Чисто NumPy (без scipy/Delaunay).

    Для верификации (намагниченная/проницаемая сфера). Сетка не идеально однородна
    (искажение у диагоналей куба), но валидна и сходится при росте `n`.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    box = build_structured_box_tetra_mesh(
        nx=n, ny=n, nz=n, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0)
    )
    src = box.vertices
    c = np.asarray(center, dtype=float)

    l2 = np.linalg.norm(src, axis=1)
    linf = np.max(np.abs(src), axis=1)
    scale = np.where(l2 > 1e-14, linf / np.where(l2 > 1e-14, l2, 1.0), 0.0)
    mapped = c[None, :] + src * (scale * radius)[:, None]

    cells = [_orient_tetra(mapped, [int(v) for v in cell]) for cell in box.cells]
    return TetraMesh(vertices=mapped, cells=np.asarray(cells, dtype=int))


def build_annular_sector_tetra_mesh(
    nr: int,
    ntheta: int,
    nz: int,
    *,
    r_in: float,
    r_out: float,
    theta_seg: float,
    z_len: float = 1.0,
) -> TetraMesh:
    """
    Тетраэдральная сетка АННУЛЯРНОГО СЕКТОРА (полюсный сегмент машины): структурная
    решётка в `(r,θ,z)` → `(x=r·cosθ, y=r·sinθ, z)`, каждая ячейка-гексаэдр → 6 тетов.

    `r∈[r_in,r_out]`, `θ∈[0,θ_seg]`, `z∈[0,z_len]`. Узлы на разрезах θ=0 и θ=θ_seg имеют
    ОДИНАКОВУЮ (r,z)-разметку ⇒ сетка периодик-конформна (сопоставление поворотом на
    θ_seg вокруг оси z; см. `femcore.periodic.match_periodic_vertices`). Топология/
    нумерация — как у `build_structured_box_tetra_mesh`; отличается только отображение
    координат. Ячейки переориентируются (отображение меняет знак объёма у части тетов).
    """
    if nr < 1 or ntheta < 1 or nz < 1:
        raise ValueError("nr, ntheta and nz must be at least 1.")
    if not (r_out > r_in > 0.0):
        raise ValueError("Require r_out > r_in > 0.")
    if not (0.0 < theta_seg < 2.0 * np.pi):
        raise ValueError("theta_seg must be in (0, 2π).")
    if z_len <= 0.0:
        raise ValueError("z_len must be positive.")

    dr = (r_out - r_in) / nr
    dth = theta_seg / ntheta
    dz = z_len / nz

    def vid(i: int, j: int, k: int) -> int:
        return i + (nr + 1) * (j + (ntheta + 1) * k)

    vertices = []
    for k in range(nz + 1):
        z = k * dz
        for j in range(ntheta + 1):
            th = j * dth
            ct, st = np.cos(th), np.sin(th)
            for i in range(nr + 1):
                r = r_in + i * dr
                vertices.append([r * ct, r * st, z])
    vertices = np.asarray(vertices, dtype=float)

    cells: list[list[int]] = []
    for k in range(nz):
        for j in range(ntheta):
            for i in range(nr):
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


def tag_sector_regions_by_radius(
    mesh: TetraMesh, band_edges: tuple[float, ...], labels: tuple[int, ...]
) -> np.ndarray:
    """
    Пометить ячейки по радиусу центроида: `labels[k]` для `band_edges[k] ≤ r < band_edges[k+1]`.
    `len(labels) == len(band_edges)-1`. Возвращает массив меток (n_cells,). Для разбиения
    сегмента на пояса: ярмо / магнит / зазор / воздух.
    """
    edges = np.asarray(band_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("band_edges must be strictly increasing, length >= 2.")
    if len(labels) != edges.size - 1:
        raise ValueError("len(labels) must equal len(band_edges) - 1.")
    out = np.full(mesh.n_cells, -1, dtype=int)
    for c in range(mesh.n_cells):
        cen = mesh.cell_centroid(c)
        r = float(np.hypot(cen[0], cen[1]))
        idx = int(np.searchsorted(edges, r, side="right") - 1)
        if 0 <= idx < len(labels):
            out[c] = labels[idx]
    return out


def build_frame_tetra_mesh(
    n: int,
    hole_lo: float = 1.0 / 3.0,
    hole_hi: float = 2.0 / 3.0,
) -> TetraMesh:
    """
    НЕСТЯГИВАЕМАЯ тетра-сетка «рамки»: куб `[0,1]³` минус центральная колонна по (x,y)
    на всю высоту z (сквозное отверстие) ⇒ топологический солид-тор, `b₁=1`.

    Для теста когомологий/калибровки на нестягиваемой области (гармонические поля —
    дополнительные калибровочные нуль-моды). Чисто NumPy: фильтрует ячейки
    структурированной box-сетки по центроиду и переиндексирует вершины.
    """
    if n < 3:
        raise ValueError("n must be >= 3 to carve a through-hole.")
    box = build_structured_box_tetra_mesh(nx=n, ny=n, nz=n)
    verts = box.vertices

    kept: list[list[int]] = []
    for cell in box.cells:
        cx, cy, _cz = verts[np.asarray(cell, dtype=int)].mean(axis=0)
        in_hole = (hole_lo < cx < hole_hi) and (hole_lo < cy < hole_hi)
        if not in_hole:
            kept.append([int(v) for v in cell])

    kept_arr = np.asarray(kept, dtype=int)
    used = np.unique(kept_arr)
    remap = -np.ones(verts.shape[0], dtype=int)
    remap[used] = np.arange(used.shape[0])
    return TetraMesh(vertices=verts[used], cells=remap[kept_arr])


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