from __future__ import annotations

import numpy as np

from magcore.mesh.mesh import canonical_cell
from magcore.mesh.mesh_generators import (
    build_structured_box_tetra_mesh,
    build_structured_unit_cube_tetra_mesh,
    build_symmetric_unit_cube_tetra_mesh,
)


def _vertex_key(x: np.ndarray, decimals: int = 12) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(x, dtype=float), decimals=decimals))


def _build_vertex_reflection_map_x_midplane(vertices: np.ndarray) -> np.ndarray:
    lookup = {_vertex_key(vertices[i]): i for i in range(len(vertices))}
    reflection = np.zeros(len(vertices), dtype=int)

    for i, xyz in enumerate(vertices):
        xr = np.array([1.0 - xyz[0], xyz[1], xyz[2]], dtype=float)
        key = _vertex_key(xr)
        if key not in lookup:
            raise AssertionError("Reflected vertex not found.")
        reflection[i] = lookup[key]

    return reflection


def test_build_structured_unit_cube_tetra_mesh_n1_basic_counts() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(1)

    assert mesh.n_vertices == 8
    assert mesh.n_cells == 6
    assert mesh.boundary_face_count() > 0
    assert mesh.edge_count() > 0


def test_build_structured_unit_cube_tetra_mesh_counts_for_multiple_n() -> None:
    for n in (1, 2, 3):
        mesh = build_structured_unit_cube_tetra_mesh(n)

        assert mesh.n_vertices == (n + 1) ** 3
        assert mesh.n_cells == 6 * n**3
        assert mesh.boundary_face_count() > 0
        assert mesh.edge_count() > 0


def test_build_structured_box_tetra_mesh_respects_box_bounds() -> None:
    mesh = build_structured_box_tetra_mesh(
        nx=2,
        ny=3,
        nz=4,
        xlim=(-1.0, 2.0),
        ylim=(10.0, 13.0),
        zlim=(5.0, 9.0),
    )

    mins = mesh.vertices.min(axis=0)
    maxs = mesh.vertices.max(axis=0)

    assert np.allclose(mins, [-1.0, 10.0, 5.0])
    assert np.allclose(maxs, [2.0, 13.0, 9.0])


def test_build_structured_box_tetra_mesh_rejects_bad_input() -> None:
    bad_cases = [
        dict(nx=0, ny=1, nz=1),
        dict(nx=1, ny=0, nz=1),
        dict(nx=1, ny=1, nz=0),
    ]

    for kwargs in bad_cases:
        try:
            build_structured_box_tetra_mesh(**kwargs)
        except ValueError:
            continue
        assert False, "Expected ValueError for invalid mesh resolution."

    try:
        build_structured_box_tetra_mesh(1, 1, 1, xlim=(1.0, 1.0))
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for degenerate coordinate interval."


def test_build_symmetric_unit_cube_tetra_mesh_basic_counts() -> None:
    mesh = build_symmetric_unit_cube_tetra_mesh()

    assert mesh.n_vertices == 15
    assert mesh.n_cells == 24
    assert mesh.boundary_face_count() == 24
    assert mesh.edge_count() > 0


def test_symmetric_unit_cube_mesh_is_closed_under_x_reflection() -> None:
    mesh = build_symmetric_unit_cube_tetra_mesh()

    vertex_reflection = _build_vertex_reflection_map_x_midplane(mesh.vertices)

    edge_set = set(mesh.all_edges())
    for i, j in mesh.all_edges():
        ri = int(vertex_reflection[i])
        rj = int(vertex_reflection[j])
        reflected_edge = (ri, rj) if ri < rj else (rj, ri)
        assert reflected_edge in edge_set

    face_set = set(mesh.boundary_faces())
    for face in mesh.boundary_faces():
        rf = tuple(sorted(int(vertex_reflection[v]) for v in face))
        assert rf in face_set

    cell_set = {canonical_cell(tuple(int(v) for v in cell)) for cell in mesh.cells}
    for cell in mesh.cells:
        reflected = canonical_cell(tuple(int(vertex_reflection[int(v)]) for v in cell))
        assert reflected in cell_set