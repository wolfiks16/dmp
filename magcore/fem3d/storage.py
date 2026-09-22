from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from magcore.fem3d.nonlinear import restore_field3d
from magcore.fem3d.objects import GeoObject3D, object_problem3d_from_mesh
from magcore.fem3d.problem import Problem3D
from magcore.fem3d.scalar import ScalarField3D
from magcore.fem3d.scene import pack, unpack

# РЕШЕНИЕ 3D В ФАЙЛЕ РАСЧЁТА (этап 3D-9, план — docs/plan_3d_2026-09-11.md). Раньше файл хранил только
# сводку, и чтобы снова увидеть поле, сетку строили и задачу решали заново (на БПЛА32 — около 6 минут).
# Теперь в файле сетка (узлы, ячейки, регион ячейки) и узловой потенциал φ — то же, что хранит 2D. Всё
# остальное (H, B, проницаемость, коэнергия, карта риска, доля r) — одна оценка тех же законов при φ
# (`restore_field3d`), без итераций; совпадает с решением до бита.
# Регион ячейки — номер объекта в списке модели (0 — домен): материалы и оси намагничивания берутся из
# объектов файла, как при построении сетки (`object_problem3d_from_mesh`).
# ПРОВЕРКА ГОДНОСТИ (Л-80: числа, сохранённые без версии расчётной формулы, молча устаревают). Номер
# версии не ведём — сохранённый φ проверяется уравнениями ТЕКУЩЕГО кода: невязка при φ, отнесённая к
# невязке начального приближения, как у решателя. Код тот же — она та же, что при решении (сохранена в
# файле); изменились материалы, модель магнита, граница — она больше. Поле принимается, если невязка не
# больше допуска решателя или не больше чем вдвое выше сохранённой (решатель мог остановиться по малому
# шагу чуть выше допуска; вдвое — запас на порядок сложения, то же φ даёт ту же невязку с точностью
# до округления). Иначе — пересчёт.

FIELD3D_FORMAT = "magfield-field3d"
FIELD3D_VERSION = 1
SOLVER_TOL = 1.0e-9          # допуск решателя по невязке (`solve_nonlinear3d`, tol по умолчанию)
STALE_FACTOR = 2.0           # во сколько раз невязка может превысить сохранённую, прежде чем поле отвергается


def field_payload(field: ScalarField3D) -> dict:
    """
    Сетка и потенциал решения для файла расчёта (массивы — base64, порядок байтов little-endian) +
    условия расчёта (температура, граница, внешнее поле) и невязка при решении — для проверки годности.
    Решение — с новым магнитом (без истории до расчёта), как считает веб; поле, решённое с историей,
    при восстановлении не пройдёт проверку невязкой (уравнения другие) и будет пересчитано.
    """
    p = field.problem
    reg = np.asarray(p.cell_region)
    if reg.max() > np.iinfo(np.uint16).max or p.mesh.n_vertices > np.iinfo(np.int32).max:
        raise ValueError("модель слишком велика для файла расчёта.")
    return {"format": FIELD3D_FORMAT, "version": FIELD3D_VERSION,
            "n_vertices": p.mesh.n_vertices, "n_cells": p.mesh.n_cells,
            "vertices": pack(p.mesh.vertices, "<f8"), "cells": pack(p.mesh.cells, "<i4"),
            "cell_region": pack(reg, "<u2"), "phi": pack(field.phi, "<f8"),
            "T": float(p.T), "bc": field.bc, "applied_field": [float(v) for v in field.applied_field],
            "residual": float(field.residual)}


@dataclass(frozen=True)
class SavedField3D:
    """Модель и поле из файла расчёта и вывод проверки годности."""

    problem: Problem3D
    field: ScalarField3D
    residual: float              # невязка уравнений текущего кода при сохранённом φ
    stored_residual: float       # невязка при решении (из файла)
    ok: bool                     # поле годится: residual ≤ max(SOLVER_TOL, STALE_FACTOR·stored_residual)


def _array(payload: dict, key: str, dtype, n: int) -> np.ndarray:
    try:
        a = unpack(str(payload[key]), dtype)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"в сохранённом поле нет массива {key!r} или он повреждён.") from e
    if a.size != n:
        raise ValueError(f"в сохранённом поле массив {key!r} другой длины ({a.size} вместо {n}).")
    return a


def restore_saved_field(payload: dict, objects, domain: GeoObject3D) -> SavedField3D:
    """
    Модель и поле по сохранённым сетке и потенциалу (`field_payload`) и объектам того же расчёта.
    Проверка годности — невязкой уравнений текущего кода (см. шапку модуля); вывод — в `ok`.
    """
    if not isinstance(payload, dict) or payload.get("format") != FIELD3D_FORMAT:
        raise ValueError("это не сохранённое поле 3D.")
    if payload.get("version") != FIELD3D_VERSION:
        raise ValueError(f"версия сохранённого поля {payload.get('version')!r} не поддерживается.")
    try:
        nv, nc = int(payload["n_vertices"]), int(payload["n_cells"])
        T, bc = float(payload["T"]), str(payload["bc"])
        H0 = np.asarray(payload["applied_field"], dtype=float)
        stored = float(payload["residual"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("в сохранённом поле нет условий расчёта или они повреждены.") from e
    if not (nv > 0 and nc > 0 and np.isfinite(T) and np.isfinite(stored) and stored >= 0.0):
        raise ValueError("условия расчёта в сохранённом поле не годятся.")
    vertices = np.array(_array(payload, "vertices", "<f8", 3 * nv).reshape(nv, 3), dtype=float)
    cells = _array(payload, "cells", "<i4", 4 * nc).reshape(nc, 4).astype(np.int64)
    cell_region = _array(payload, "cell_region", "<u2", nc).astype(np.int64)
    phi = _array(payload, "phi", "<f8", nv)
    problem = object_problem3d_from_mesh(objects, domain, vertices, cells, cell_region, T=T)
    field = restore_field3d(problem, phi, bc=bc, applied_field=H0, tol=SOLVER_TOL)
    ok = bool(field.residual <= max(SOLVER_TOL, STALE_FACTOR * stored))
    return SavedField3D(problem=problem, field=replace(field, converged=ok), residual=float(field.residual),
                        stored_residual=stored, ok=ok)
