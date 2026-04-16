from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.bem.reference_fields import harmonic_point_source
from magcore.bem.single_layer_solve import (
    compute_reference_error,
    reconstruct_potential_from_density,
    solve_single_layer_dirichlet_p0,
)
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
        problem_id="tetra_case_recon",
        name="Closed tetra reconstruction case",
        materials=materials,
        regions=regions,
        surface_mesh=mesh,
        topology=topology,
        external_field=ExternalField(h_ext=(0.0, 0.0, 0.0)),
    )


def make_reference_function():
    # Source is strictly inside the tetrahedron.
    source = np.array([0.1, 0.1, 0.1], dtype=float)
    return harmonic_point_source(source)


def test_reconstruction_runs_and_is_finite():
    problem = make_tetra_problem()
    ref_fn = make_reference_function()

    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    solve = solve_single_layer_dirichlet_p0(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        ref_fn=ref_fn,
        adaptive_config=config,
    )

    pts = np.array(
        [
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 1.5],
            [2.5, 3.5, 2.0],
        ],
        dtype=float,
    )

    pred = reconstruct_potential_from_density(
        mesh=problem.surface_mesh,
        face_indices=(0, 1, 2, 3),
        sigma=solve.sigma,
        target_points=pts,
        quadrature_order=2,
    )

    assert pred.shape == (3,)
    assert np.isfinite(pred).all()


def test_reconstruction_error_is_bounded_for_decaying_exterior_reference():
    problem = make_tetra_problem()
    ref_fn = make_reference_function()

    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    solve = solve_single_layer_dirichlet_p0(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        ref_fn=ref_fn,
        adaptive_config=config,
    )

    pts = np.array(
        [
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 1.5],
            [2.5, 3.5, 2.0],
            [4.0, 4.0, 3.0],
        ],
        dtype=float,
    )

    pred = reconstruct_potential_from_density(
        mesh=problem.surface_mesh,
        face_indices=(0, 1, 2, 3),
        sigma=solve.sigma,
        target_points=pts,
        quadrature_order=2,
    )

    err = compute_reference_error(pred, pts, ref_fn)

    assert np.isfinite(err.l2)
    assert np.isfinite(err.linf)
    # Verification gate: bounded reconstruction error against a decaying exterior harmonic reference
    assert err.linf < 0.2


def test_reconstruction_error_report_shapes():
    problem = make_tetra_problem()
    ref_fn = make_reference_function()

    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    solve = solve_single_layer_dirichlet_p0(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        ref_fn=ref_fn,
        adaptive_config=config,
    )

    pts = np.array(
        [
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 1.5],
        ],
        dtype=float,
    )

    pred = reconstruct_potential_from_density(
        mesh=problem.surface_mesh,
        face_indices=(0, 1, 2, 3),
        sigma=solve.sigma,
        target_points=pts,
        quadrature_order=2,
    )
    err = compute_reference_error(pred, pts, ref_fn)

    assert err.abs_errors.shape == (2,)
    assert err.prediction.shape == (2,)
    assert err.reference.shape == (2,)