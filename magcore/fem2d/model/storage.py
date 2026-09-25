from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.fem2d.model.problem import Problem2D, Solution2D, restore_problem2d
from magcore.packing import pack, unpack

# РЕШЕНИЕ 2D В ФАЙЛЕ РАСЧЁТА — то же, что этап 3D-9 для 3D (fem3d/storage.py; решение Sergey 2026-09-25:
# «поле из файла в 2D — по аналогии с 3D»). Файл 2D и раньше хранил поле, но только для показа: B, H и
# A_z округлены, и открытый расчёт с текущей версией решателя не сверялся. Теперь в файле ещё и точная
# копия: сетка (узлы, ячейки, регион ячейки), узловой A_z, T и невязка при решении; у машины — ещё ось
# магнита и номер паза по ячейкам (это её геометрия, как сетка). Всё остальное — материалы, токи обмотки,
# оси намагничивания свободной геометрии, модель магнита, уравнения — при открытии строит ТЕКУЩИЙ код.
# ПРОВЕРКА ГОДНОСТИ (Л-80: числа, сохранённые без версии расчётной формулы, молча устаревают) — как в 3D:
# невязка уравнений текущего кода при сохранённом A_z, отнесённая к невязке начального приближения, как у
# решателя (`restore_problem2d`). Код тот же — она та же, что при решении (сохранена в файле); изменились
# уравнения — она больше. Поле принимается, если невязка не больше допуска решателя или не больше чем
# вдвое выше сохранённой (решатель мог остановиться чуть выше допуска; то же A даёт ту же невязку с
# точностью до порядка сложения). Иначе — пересчёт.

FIELD2D_FORMAT = "magfield-field2d"
FIELD2D_VERSION = 1
SOLVER_TOL = 1.0e-6          # допуск решателя по невязке (`solve_problem2d`, tol по умолчанию)
STALE_FACTOR = 2.0           # во сколько раз невязка может превысить сохранённую, прежде чем поле отвергается


def relative_residual(solution: Solution2D) -> float:
    """Невязка решения, отнесённая к невязке начального приближения, — по ней Ньютон ставит «сошлось»."""
    h = solution.field.rel_change_history
    if len(h) < 1 or not h[0] > 0.0:
        raise ValueError("у решения нет истории невязки метода Ньютона.")
    return float(h[-1] / h[0])


def field_payload2d(solution: Solution2D, *, magnet_axis=None, slot_id=None) -> dict:
    """
    Сетка и A_z решения для файла расчёта (массивы — base64, little-endian) + T и невязка при решении.
    `magnet_axis` (n_cells, 2) и `slot_id` (n_cells,) — геометрия машины (у свободной геометрии оси и
    токи строятся из объектов, их не передают). Решение — методом Ньютона с новым магнитом (без истории),
    как считает веб.
    """
    p = solution.problem
    mesh = p.mesh
    reg = np.asarray(p.cell_region)
    if reg.min() < 0 or reg.max() > np.iinfo(np.uint16).max or mesh.n_vertices > np.iinfo(np.int32).max:
        raise ValueError("модель слишком велика для файла расчёта.")
    out = {"format": FIELD2D_FORMAT, "version": FIELD2D_VERSION,
           "n_vertices": int(mesh.n_vertices), "n_cells": int(mesh.n_cells),
           "vertices": pack(mesh.vertices, "<f8"), "cells": pack(mesh.cells, "<i4"),
           "cell_region": pack(reg, "<u2"), "a": pack(solution.field.a, "<f8"),
           "T": float(p.T), "residual": relative_residual(solution)}
    if magnet_axis is not None:
        out["magnet_axis"] = pack(np.asarray(magnet_axis, dtype=float).reshape(mesh.n_cells, 2), "<f8")
    if slot_id is not None:
        out["slot_id"] = pack(np.asarray(slot_id).reshape(mesh.n_cells), "<i4")
    return out


@dataclass(frozen=True)
class SavedMesh2D:
    """Сетка, регион ячейки, A_z и условия из сохранённого поля (проверены на целостность)."""

    vertices: np.ndarray         # (n_vertices, 2) [м]
    cells: np.ndarray            # (n_cells, 3)
    cell_region: np.ndarray      # (n_cells,)
    a: np.ndarray                # (n_vertices,) узловой A_z [Вб/м]
    T: float                     # [°C]
    stored_residual: float       # невязка при решении


@dataclass(frozen=True)
class SavedField2D:
    """Решение из файла расчёта и вывод проверки годности."""

    solution: Solution2D
    residual: float              # невязка уравнений текущего кода при сохранённом A_z
    stored_residual: float       # невязка при решении (из файла)
    ok: bool                     # поле годится: residual ≤ max(SOLVER_TOL, STALE_FACTOR·stored_residual)


def saved_array(payload: dict, key: str, dtype, count: int) -> np.ndarray:
    """Массив `key` сохранённого поля: ровно `count` чисел, иначе — понятная ошибка."""
    try:
        a = unpack(str(payload[key]), dtype)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"в сохранённом поле нет массива {key!r} или он повреждён.") from e
    if a.size != count:
        raise ValueError(f"в сохранённом поле массив {key!r} другой длины ({a.size} вместо {count}).")
    return a


def saved_mesh2d(payload: dict) -> SavedMesh2D:
    """Сетка, регион ячейки, A_z, T и невязка из сохранённого поля (`field_payload2d`)."""
    if not isinstance(payload, dict) or payload.get("format") != FIELD2D_FORMAT:
        raise ValueError("это не сохранённое поле 2D.")
    if payload.get("version") != FIELD2D_VERSION:
        raise ValueError(f"версия сохранённого поля {payload.get('version')!r} не поддерживается.")
    try:
        nv, nc = int(payload["n_vertices"]), int(payload["n_cells"])
        T, stored = float(payload["T"]), float(payload["residual"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("в сохранённом поле нет условий расчёта или они повреждены.") from e
    if not (nv > 0 and nc > 0 and np.isfinite(T) and np.isfinite(stored) and stored >= 0.0):
        raise ValueError("условия расчёта в сохранённом поле не годятся.")
    return SavedMesh2D(
        vertices=np.array(saved_array(payload, "vertices", "<f8", 2 * nv).reshape(nv, 2), dtype=float),
        cells=saved_array(payload, "cells", "<i4", 3 * nc).reshape(nc, 3).astype(np.int64),
        cell_region=saved_array(payload, "cell_region", "<u2", nc).astype(np.int64),
        a=np.array(saved_array(payload, "a", "<f8", nv), dtype=float), T=T, stored_residual=stored)


def restore_saved_field2d(saved: SavedMesh2D, problem: Problem2D) -> SavedField2D:
    """
    Проверить сохранённый A_z на задаче той же модели, собранной текущим кодом на сетке из файла
    (`object_problem_from_mesh` или машина на своей геометрии). Вывод — в `ok`, см. шапку модуля.
    """
    if problem.mesh.n_vertices != saved.a.size:
        raise ValueError("задача собрана не на сетке сохранённого поля.")
    sol = restore_problem2d(problem, saved.a, tol=SOLVER_TOL)
    residual = relative_residual(sol)
    ok = bool(residual <= max(SOLVER_TOL, STALE_FACTOR * saved.stored_residual))
    return SavedField2D(solution=sol, residual=residual, stored_residual=saved.stored_residual, ok=ok)
