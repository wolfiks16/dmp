from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from magcore.fem2d.model.materials import (
    Air,
    LinearMaterial,
    MagnetMaterial,
    SteelMaterial,
)
from magcore.fem2d.model.problem import Problem2D, Solution2D

# Сериализация Problem2D → «сцена» (JSON) для интерфейса: реальная треугольная сетка +
# регионы по материалам + ось магнита (+ опц. поле из решения). Общая, для ЛЮБОЙ задачи —
# UI-вьюпорт рисует НАСТОЯЩУЮ сетку из решателя, а не стилизацию.


def _kind(material) -> str:
    if isinstance(material, Air):
        return "air"
    if isinstance(material, SteelMaterial):
        return "steel"
    if isinstance(material, MagnetMaterial):
        return "magnet"
    if isinstance(material, LinearMaterial):
        return "linear"
    return "other"


def problem_to_scene(
    problem: Problem2D,
    *,
    solution: Solution2D | None = None,
    scale_mm: float = 1000.0,
    ndigits: int = 4,
) -> dict:
    """
    Собрать сцену: узлы (мм), треугольники, регион на ячейку, словарь регионов (имя+тип
    материала), ось магнита. Если дано `solution` — добавить поячеечный |B| [Тл]. Координаты
    масштабируются в мм (scale_mm=1000 для м→мм) и округляются (компактный JSON).
    """
    mesh = problem.mesh
    V = np.asarray(mesh.vertices, dtype=float) * float(scale_mm)
    cells = np.asarray(mesh.cells, dtype=int)
    region = np.asarray(problem.cell_region, dtype=int)
    regions = {
        str(rid): {"name": r.name, "kind": _kind(r.material)}
        for rid, r in problem.regions.items()
    }
    scene: dict = {
        "units": "mm",
        "bounds": {
            "xmin": float(V[:, 0].min()), "xmax": float(V[:, 0].max()),
            "ymin": float(V[:, 1].min()), "ymax": float(V[:, 1].max()),
        },
        "vertices": np.round(V, ndigits).tolist(),
        "cells": cells.tolist(),
        "region": region.tolist(),
        "regions": regions,
    }
    if problem.magnet_axis is not None:
        scene["magnet_axis"] = np.round(np.asarray(problem.magnet_axis, dtype=float), 4).tolist()
    if solution is not None:
        B = solution.field.B_cells
        scene["Bmag"] = np.round(np.hypot(B[:, 0], B[:, 1]), 4).tolist()
    return scene


def write_scene_json(path: str | Path, scene: dict) -> Path:
    """Записать сцену в JSON-файл."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
    return p
