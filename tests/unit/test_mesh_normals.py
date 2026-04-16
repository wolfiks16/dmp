from __future__ import annotations

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.normals import (
    shared_edge_orientation,
    find_orientation_conflicts,
    orientability_check,
)


def make_consistent_square() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def make_inconsistent_square() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],  # shared edge (1,2) has same direction conflict
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_shared_edge_orientation_consistent_pair():
    mesh = make_consistent_square()
    sign = shared_edge_orientation(mesh.faces[0], mesh.faces[1])

    assert sign == -1


def test_shared_edge_orientation_conflicting_pair():
    mesh = make_inconsistent_square()
    sign = shared_edge_orientation(mesh.faces[0], mesh.faces[1])

    assert sign == +1


def test_find_orientation_conflicts_empty_for_consistent_patch():
    mesh = make_consistent_square()
    conflicts = find_orientation_conflicts(mesh, (0, 1))

    assert conflicts == ()


def test_find_orientation_conflicts_detects_conflict():
    mesh = make_inconsistent_square()
    conflicts = find_orientation_conflicts(mesh, (0, 1))

    assert conflicts == ((0, 1),)


def test_orientability_check_true_for_consistent_patch():
    mesh = make_consistent_square()
    assert orientability_check(mesh, (0, 1)) is True