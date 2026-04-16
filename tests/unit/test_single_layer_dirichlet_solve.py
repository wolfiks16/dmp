from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.bem.reference_fields import harmonic_linear_z
from magcore.bem.single_layer_solve import solve_single_layer_dirichlet_p0
from magcore.domain.fields import ExternalField
from magcore.domain.interfaces import InterfacePatch
from magcore.domain.materials import AirMaterial
from magcore.domain.problem import MagnetostaticProblem
from magcore.domain.regions import Region
from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.topology import RegionTopology


MU0 = 4.0e-7 * 3.141592653589793


def make_tetra_problem() -> MagnetostaticProblem:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=int,
    )
    mesh = SurfaceMesh(vertices=vertices, faces=faces)

    materials = (
        AirMaterial(material_id="mat_air", name="Air", mu=MU0),
    )
    regions = (
        Region(region_id="air", name="Air", material_id="mat_air", is_external=True),
        Region(region_id="scatterer", name="Scatterer", material_id="mat_air", is_external=False),
    )
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="gamma",
                name="closed_surface",
                region_minus_id="scatterer",
                region_plus_id="air",
                face_indices=(0, 1, 2, 3),
            ),
        )
    )

    return MagnetostaticProblem(
        problem_id="tetra_case",
        name="Closed tetra verification case",
        materials=materials,
        regions=regions,
        surface_mesh=mesh,
        topology=topology,
        external_field=ExternalField(h_ext=(0.0, 0.0, 0.0)),
    )


def test_single_layer_dirichlet_solve_runs():
    problem = make_tetra_problem()
    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    result = solve_single_layer_dirichlet_p0(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        ref_fn=harmonic_linear_z,
        adaptive_config=config,
    )

    assert result.sigma.shape == (4,)
    assert result.boundary_data.shape == (4,)
    assert result.face_centroids.shape == (4, 3)
    assert np.isfinite(result.sigma).all()
    assert np.isfinite(result.residual_norm)
    assert result.residual_norm < 1.0e-8


def test_single_layer_dirichlet_matrix_is_finite():
    problem = make_tetra_problem()
    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    result = solve_single_layer_dirichlet_p0(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        ref_fn=harmonic_linear_z,
        adaptive_config=config,
    )

    assert np.isfinite(result.matrix).all()


def test_single_layer_dirichlet_condition_number_is_finite():
    problem = make_tetra_problem()
    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    result = solve_single_layer_dirichlet_p0(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        ref_fn=harmonic_linear_z,
        adaptive_config=config,
    )

    assert np.isfinite(result.condition_number_est)
    assert result.condition_number_est > 0.0