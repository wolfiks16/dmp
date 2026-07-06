from __future__ import annotations

import numpy as np

from magcore.hybrid.interface import CouplingInterface
from magcore.mesh.mesh_generators import (
    build_structured_box_tetra_mesh,
    build_structured_unit_cube_tetra_mesh,
)
from magcore.mesh.normals import orientability_check


def test_interface_unit_cube_basic_counts() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    ci = CouplingInterface.from_tetra_mesh(mesh)

    assert ci.n_faces == len(mesh.boundary_faces())
    assert ci.n_faces == 48  # 6 sides * 4 squares * 2 triangles
    assert ci.n_phi_dofs == len(mesh.boundary_vertices())
    assert ci.n_flux_dofs == ci.n_faces
    assert ci.surface_mesh.validate_basic() == ()
    assert abs(float(np.sum(ci.face_areas)) - 6.0) < 1e-9  # unit cube surface area


def test_interface_is_closed_and_normals_point_outward() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    ci = CouplingInterface.from_tetra_mesh(mesh)

    # Closed surface: area-weighted outward normals sum to zero (∮ n dS = 0).
    flux = (ci.outward_normals * ci.face_areas[:, None]).sum(axis=0)
    assert np.linalg.norm(flux) < 1e-9

    center = np.array([0.5, 0.5, 0.5])
    for f in range(ci.n_faces):
        c = ci.surface_mesh.face_centroid(f)
        # outward normal has positive projection onto (centroid - cube center)
        assert float(np.dot(ci.outward_normals[f], c - center)) > 0.0
        # SurfaceMesh winding reproduces the stored outward normal
        assert np.allclose(ci.surface_mesh.face_normal(f), ci.outward_normals[f], atol=1e-9)


def test_interface_surface_is_orientable() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    assert orientability_check(ci.surface_mesh, tuple(range(ci.n_faces)))


def test_interface_vertex_and_cell_maps_are_consistent() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    ci = CouplingInterface.from_tetra_mesh(mesh)

    g2s = ci.global_to_surface_vertex()
    assert len(g2s) == ci.n_phi_dofs
    assert set(g2s.values()) == set(range(ci.n_phi_dofs))

    for f in range(ci.n_faces):
        c = int(ci.face_to_cell[f])
        cell_verts = set(mesh.cell_vertex_indices(c))
        face_global = {int(ci.surface_to_global_vertex[v]) for v in ci.surface_mesh.faces[f]}
        assert face_global.issubset(cell_verts)


def test_interface_nonunit_box_areas_and_closure() -> None:
    # 2 x 1 x 1 box: surface area = 2*(2*1) + 2*(2*1) + 2*(1*1) = 10
    mesh = build_structured_box_tetra_mesh(2, 1, 1, xlim=(0.0, 2.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0))
    ci = CouplingInterface.from_tetra_mesh(mesh)

    flux = (ci.outward_normals * ci.face_areas[:, None]).sum(axis=0)
    assert np.linalg.norm(flux) < 1e-9
    assert abs(float(np.sum(ci.face_areas)) - 10.0) < 1e-9
