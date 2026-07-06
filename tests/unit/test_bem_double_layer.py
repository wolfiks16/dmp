from __future__ import annotations

import numpy as np
import pytest

from magcore.bem.double_layer import double_layer_potential_at_points
from magcore.hybrid.interface import CouplingInterface
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _unit_cube_interface(n: int) -> CouplingInterface:
    return CouplingInterface.from_tetra_mesh(build_structured_unit_cube_tetra_mesh(n))


def test_double_layer_solid_angle_identity() -> None:
    # (K·1)(x) = ∮_Γ ∂G/∂n_y ds_y = -1 for x inside Ω, 0 for x outside.
    ci = _unit_cube_interface(4)
    psi = np.ones(ci.surface_mesh.n_vertices)

    inside = double_layer_potential_at_points(
        ci.surface_mesh, psi, np.array([[0.5, 0.5, 0.5], [0.3, 0.4, 0.6]]), ci.outward_normals, 2
    )
    outside = double_layer_potential_at_points(
        ci.surface_mesh, psi, np.array([[3.0, 3.0, 3.0], [-1.0, 0.5, 2.0]]), ci.outward_normals, 2
    )

    assert np.allclose(inside, -1.0, atol=2e-3)
    assert np.allclose(outside, 0.0, atol=1e-5)


def test_double_layer_converges_to_solid_angle_under_refinement() -> None:
    pts = np.array([[0.5, 0.5, 0.5]])
    errs = []
    for n in (2, 4):
        ci = _unit_cube_interface(n)
        psi = np.ones(ci.surface_mesh.n_vertices)
        val = double_layer_potential_at_points(ci.surface_mesh, psi, pts, ci.outward_normals, 2)[0]
        errs.append(abs(val + 1.0))
    assert errs[1] < errs[0]


def test_double_layer_is_linear_in_psi() -> None:
    ci = _unit_cube_interface(2)
    rng = np.random.default_rng(0)
    psi1 = rng.standard_normal(ci.surface_mesh.n_vertices)
    psi2 = rng.standard_normal(ci.surface_mesh.n_vertices)
    pts = np.array([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]])

    k1 = double_layer_potential_at_points(ci.surface_mesh, psi1, pts, ci.outward_normals, 2)
    k2 = double_layer_potential_at_points(ci.surface_mesh, psi2, pts, ci.outward_normals, 2)
    k12 = double_layer_potential_at_points(ci.surface_mesh, 3.0 * psi1 - 2.0 * psi2, pts, ci.outward_normals, 2)

    assert np.allclose(k12, 3.0 * k1 - 2.0 * k2, atol=1e-12)


def test_double_layer_validates_shapes() -> None:
    ci = _unit_cube_interface(2)
    with pytest.raises(ValueError):
        double_layer_potential_at_points(
            ci.surface_mesh, np.ones(3), np.array([[0.5, 0.5, 0.5]]), ci.outward_normals, 2
        )
