from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ЛЁГКИЙ ВЕКТОРНЫЙ контейнер тетраэдральной сетки для 3D-решателя свободной геометрии (3D-1).
# Инварианты — те же, что у `magcore.mesh.TetraMesh` старого ядра (форма, конечность, индексы,
# положительная ориентация, без повторов, грань не больше чем у двух ячеек), но проверяются
# массивно: у `TetraMesh` проверка идёт циклом Python по ячейкам и на сотнях тысяч ячеек
# занимает десятки секунд. Для сверки со старым ядром на малых сетках — `to_tetra_mesh()`.

# Локальные грани и рёбра тетраэдра — в той же нумерации, что у TetraMesh.
_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64)
_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)


def signed_volumes6(vertices: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Шестикратные ориентированные объёмы ячеек det[v1−v0, v2−v0, v3−v0], форма (M,)."""
    v = np.asarray(vertices, dtype=float)[np.asarray(cells, dtype=np.int64)]
    J = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], axis=2)
    return np.linalg.det(J)


def unique_rows(rows, *, return_inverse: bool = False, return_counts: bool = False):
    """
    То же, что np.unique(rows, axis=0, return_inverse=…, return_counts=…), для строк неотрицательных целых
    (грани — тройки номеров вершин, пары «регион, вершина»): уникальные строки в лексикографическом порядке,
    обратные номера, счётчики. Считается через числовой ключ строки Σ rows[:, j]·B^(k−1−j), B = max + 1: он
    строго растёт в лексикографическом порядке строк, поэтому результат тот же до бита, а сортировка —
    одномерная, по числам (на 625 тыс. ячеек поиск одинаковых граней по строкам занимал 3–4 с, треть времени
    открытия расчёта из файла). Если ключ не помещается в int64 (B^k ≥ 2⁶³) — прежний путь по строкам.
    """
    r = np.asarray(rows)
    flags = dict(return_inverse=return_inverse, return_counts=return_counts)
    if (r.ndim != 2 or r.shape[0] == 0 or r.shape[1] == 0 or not np.issubdtype(r.dtype, np.integer)
            or int(r.min()) < 0 or (int(r.max()) + 1) ** r.shape[1] >= 2 ** 63):
        return np.unique(r, axis=0, **flags)
    base = int(r.max()) + 1
    key = r[:, 0].astype(np.int64)
    for j in range(1, r.shape[1]):
        key = key * base + r[:, j].astype(np.int64)
    res = np.unique(key, return_index=True, **flags)
    out = [r[res[1]]] + list(res[2:])
    return out[0] if len(out) == 1 else tuple(out)


def _has_duplicate_rows(rows: np.ndarray) -> bool:
    """Есть ли одинаковые строки из четырёх неотрицательных целых (тетраэдры) — сортировкой по двум ключам-парам."""
    r = np.asarray(rows, dtype=np.int64)
    base = int(r.max()) + 1 if r.size else 1
    if base ** 2 >= 2 ** 63:
        return np.unique(r, axis=0).shape[0] != r.shape[0]
    k1, k2 = r[:, 0] * base + r[:, 1], r[:, 2] * base + r[:, 3]
    o = np.lexsort((k2, k1))
    return bool(np.any((k1[o][1:] == k1[o][:-1]) & (k2[o][1:] == k2[o][:-1])))


def orient_cells(vertices: np.ndarray, cells) -> np.ndarray:
    """Копия ячеек, где у отрицательно ориентированных переставлены вершины 1 ↔ 2."""
    cells = np.array(cells, dtype=np.int64, copy=True)
    neg = signed_volumes6(vertices, cells) < 0.0
    tmp = cells[neg, 1].copy()
    cells[neg, 1] = cells[neg, 2]
    cells[neg, 2] = tmp
    return cells


@dataclass(frozen=True, slots=True)
class TetMesh3D:
    """Тетраэдральная сетка: вершины (N,3) [м] + ячейки (M,4), все положительно ориентированы."""

    vertices: np.ndarray
    cells: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", np.ascontiguousarray(self.vertices, dtype=float))
        object.__setattr__(self, "cells", np.ascontiguousarray(self.cells, dtype=np.int64))
        self.validate()

    def validate(self) -> None:
        v, c = self.vertices, self.cells
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices должны иметь форму (N, 3).")
        if c.ndim != 2 or c.shape[1] != 4:
            raise ValueError("cells должны иметь форму (M, 4).")
        if v.shape[0] == 0 or c.shape[0] == 0:
            raise ValueError("сетка пустая.")
        if not np.isfinite(v).all():
            raise ValueError("координаты вершин должны быть конечными.")
        if c.min() < 0 or c.max() >= v.shape[0]:
            raise ValueError("индексы вершин в ячейках вне диапазона.")
        s = np.sort(c, axis=1)
        if np.any(s[:, 1:] == s[:, :-1]):
            raise ValueError("в ячейке повторяется вершина.")
        if _has_duplicate_rows(s):
            raise ValueError("в сетке повторяются тетраэдры.")
        if np.any(signed_volumes6(v, c) <= 0.0):
            raise ValueError("есть ячейки с неположительным ориентированным объёмом.")
        faces = np.sort(c[:, _FACES].reshape(-1, 3), axis=1)
        _, counts = unique_rows(faces, return_counts=True)
        if np.any(counts > 2):
            raise ValueError("неманифолдная сетка: грань принадлежит больше чем двум ячейкам.")

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    def cell_volumes(self) -> np.ndarray:
        """Объёмы ячеек [м³], форма (M,)."""
        return signed_volumes6(self.vertices, self.cells) / 6.0

    def cell_centroids(self) -> np.ndarray:
        """Центры ячеек, форма (M,3)."""
        return self.vertices[self.cells].mean(axis=1)

    def edge_lengths(self) -> np.ndarray:
        """Длины шести рёбер каждой ячейки, форма (M,6)."""
        v = self.vertices[self.cells]
        return np.linalg.norm(v[:, _EDGES[:, 1]] - v[:, _EDGES[:, 0]], axis=2)

    def quality(self) -> np.ndarray:
        """
        Качество ячейки q = 6√2·V / l³, где l — среднеквадратичная длина ребра: у правильного
        тетраэдра q = 1, у вырожденного (плоского) q → 0.
        """
        l_rms = np.sqrt(np.mean(self.edge_lengths() ** 2, axis=1))
        return 6.0 * np.sqrt(2.0) * self.cell_volumes() / l_rms ** 3

    def boundary_faces(self) -> np.ndarray:
        """Граничные грани (принадлежат одной ячейке) — отсортированные тройки вершин (K,3)."""
        faces = np.sort(self.cells[:, _FACES].reshape(-1, 3), axis=1)
        uniq, counts = unique_rows(faces, return_counts=True)
        return uniq[counts == 1]

    def boundary_faces_oriented(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Граничные грани в порядке вершин с НАРУЖНОЙ нормалью (K,3) и их векторы площади n·S (K,3).
        Локальный порядок граней (как у TetraMesh) у положительно ориентированной ячейки даёт
        нормаль (v_b − v_a)×(v_c − v_a) наружу.
        """
        f = self.cells[:, _FACES].reshape(-1, 3)
        _, inv, counts = unique_rows(np.sort(f, axis=1), return_inverse=True, return_counts=True)
        fb = f[counts[np.asarray(inv).reshape(-1)] == 1]
        v = self.vertices
        area_vec = 0.5 * np.cross(v[fb[:, 1]] - v[fb[:, 0]], v[fb[:, 2]] - v[fb[:, 0]])
        return fb, area_vec

    def to_tetra_mesh(self):
        """Сетка старого 3D-ядра (`magcore.mesh.TetraMesh`) — для сверки на МАЛЫХ сетках."""
        from magcore.mesh.mesh import TetraMesh

        return TetraMesh(vertices=self.vertices.copy(), cells=self.cells.copy())
