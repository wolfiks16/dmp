from __future__ import annotations

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.quality import (
    face_edge_lengths,
    face_aspect_ratio,
    mesh_quality_summary,
    find_near_degenerate_faces,
    find_tiny_edges,
)


def make_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_face_edge_lengths_returns_three_positive_values():
    mesh = make_mesh()
    lengths = face_edge_lengths(mesh, 0)

    assert len(lengths) == 3
    assert all(l > 0.0 for l in lengths)


def test_face_aspect_ratio_is_at_least_one():
    mesh = make_mesh()
    ar = face_aspect_ratio(mesh, 0)

    assert ar >= 1.0


def test_mesh_quality_summary_contains_expected_keys():
    mesh = make_mesh()
    summary = mesh_quality_summary(mesh)

    assert summary["n_faces"] == 1
    assert "min_area" in summary
    assert "max_area" in summary
    assert "mean_area" in summary
    assert "min_edge" in summary
    assert "max_edge" in summary
    assert "max_aspect_ratio" in summary


def test_find_near_degenerate_faces_detects_none_for_normal_triangle():
    mesh = make_mesh()
    faces = find_near_degenerate_faces(mesh, area_threshold=1.0e-12)

    assert faces == ()


def test_find_tiny_edges_detects_expected_edge():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0e-15, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    mesh = SurfaceMesh(vertices=vertices, faces=faces)

    tiny = find_tiny_edges(mesh, edge_threshold=1.0e-12)
    assert (0, 1) in tiny