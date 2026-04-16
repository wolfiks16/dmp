from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.bem.adjoint_trace_collocation import OffsetTraceConfig
from magcore.bem.background_fields import linear_background_potential, evaluate_background_on_points
from magcore.bem.transmission_solve import (
    TransmissionContrastConfig,
    reconstruct_transmission_potential,
    solve_linear_transmission_problem,
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
        Region(region_id="incl", name="Inclusion", material_id="mat_air", is_external=False),
    )
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="gamma",
                name="closed_surface",
                region_minus_id="incl",
                region_plus_id="air",
                face_indices=(0, 1, 2, 3),
            ),
        )
    )

    return MagnetostaticProblem(
        problem_id="trans_contrast",
        name="Transmission contrast",
        materials=materials,
        regions=regions,
        surface_mesh=mesh,
        topology=topology,
        external_field=ExternalField(h_ext=(0.0, 0.0, 0.0)),
    )


def make_cfg(mu_in: float, mu_out: float) -> TransmissionContrastConfig:
    return TransmissionContrastConfig(
        adaptive_config=AdaptiveIntegrationConfig(
            quadrature_order=2,
            near_factor=2.0,
            max_depth=5,
            self_max_depth=6,
        ),
        offset_config=OffsetTraceConfig(
            quadrature_order=2,
            offset_factor=0.05,
        ),
        mu_in=mu_in,
        mu_out=mu_out,
        H0=np.array([0.0, 0.0, 1.0], dtype=float),
    )


def test_transmission_solve_with_contrast_runs():
    problem = make_tetra_problem()
    cfg = make_cfg(2.0 * MU0, MU0)

    result = solve_linear_transmission_problem(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        cfg=cfg,
    )

    assert result.lambda_in.shape == (4,)
    assert result.lambda_out.shape == (4,)
    assert np.isfinite(result.lambda_in).all()
    assert np.isfinite(result.lambda_out).all()
    assert np.isfinite(result.condition_number_est)
    assert result.residual_norm < 1.0e-8


def test_transmission_solve_with_contrast_induced_response_is_nontrivial():
    problem = make_tetra_problem()
    cfg = make_cfg(2.0 * MU0, MU0)

    result = solve_linear_transmission_problem(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        cfg=cfg,
    )

    assert np.linalg.norm(result.lambda_in) > 1.0e-12
    assert np.linalg.norm(result.lambda_out) > 1.0e-12


def test_transmission_solve_with_contrast_exterior_field_differs_from_background():
    problem = make_tetra_problem()
    cfg = make_cfg(2.0 * MU0, MU0)

    result = solve_linear_transmission_problem(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        cfg=cfg,
    )

    pts = np.array(
        [
            [2.0, 2.0, 2.0],
            [3.0, 2.0, 1.0],
            [2.5, 3.5, 2.0],
        ],
        dtype=float,
    )

    pred = reconstruct_transmission_potential(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        result=result,
        target_points=pts,
        H0=cfg.H0,
        side="exterior",
        quadrature_order=2,
    )

    bg = evaluate_background_on_points(pts, linear_background_potential(cfg.H0))

    diff = np.linalg.norm(pred - bg)
    assert np.all(np.isfinite(pred))
    assert diff > 1.0e-10


def test_transmission_system_matrix_is_finite():
    problem = make_tetra_problem()
    cfg = make_cfg(2.0 * MU0, MU0)

    result = solve_linear_transmission_problem(
        problem=problem,
        face_indices=(0, 1, 2, 3),
        cfg=cfg,
    )

    assert result.system_matrix.shape == (8, 8)
    assert np.isfinite(result.system_matrix).all()