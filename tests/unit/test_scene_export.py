import json

import numpy as np

from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.model import (
    Air,
    Problem2D,
    Region2D,
    problem_to_scene,
    solve_problem2d,
    write_scene_json,
)

# Экспорт сцены Problem2D → JSON (реальная сетка + регионы) для UI-вьюпорта.


def _air_problem():
    mesh = build_structured_rectangle_tri_mesh(3, 3, x1=2.0, y1=1.0)
    nc = mesh.n_cells
    return Problem2D(mesh=mesh, cell_region=np.zeros(nc, int),
                     regions={0: Region2D(0, "air", Air())})


def test_scene_structure_and_units():
    p = _air_problem()
    s = problem_to_scene(p, scale_mm=1000.0)
    assert len(s["vertices"]) == p.mesh.n_vertices
    assert len(s["cells"]) == p.mesh.n_cells
    assert len(s["region"]) == p.mesh.n_cells
    assert s["regions"]["0"]["kind"] == "air"
    assert s["units"] == "mm"
    assert np.isclose(s["bounds"]["xmax"], 2000.0)   # 2 м → 2000 мм
    assert np.isclose(s["bounds"]["ymax"], 1000.0)


def test_scene_with_solution_carries_field():
    p = _air_problem()
    sol = solve_problem2d(p, max_iter=20)
    s = problem_to_scene(p, solution=sol)
    assert len(s["Bmag"]) == p.mesh.n_cells


def test_write_scene_json_roundtrip(tmp_path):
    p = _air_problem()
    out = write_scene_json(tmp_path / "scene.json", problem_to_scene(p))
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["cells"] and d["regions"] and d["vertices"]
