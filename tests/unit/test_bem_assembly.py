from __future__ import annotations

import numpy as np

from magcore.domain.fields import ExternalField
from magcore.domain.interfaces import InterfacePatch
from magcore.domain.materials import AirMaterial, PermanentMagnetMaterial
from magcore.domain.problem import MagnetostaticProblem
from magcore.domain.regions import Region
from magcore.domain.config import SolverConfig
from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.topology import RegionTopology
from magcore.bem.assembly import (
    build_trace_spaces,
    build_assembly_context,
    build_placeholder_operators,
    assemble_operator_placeholder,
    assemble_multitrace_system_placeholder,
)
from magcore.bem.solver import solve_surface_bem_placeholder


MU0 = 4.0e-7 * 3.141592653589793


def make_problem() -> MagnetostaticProblem:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)

    mesh = SurfaceMesh(vertices=vertices, faces=faces)

    materials = (
        AirMaterial(material_id="mat_air", name="Air", mu=MU0),
        PermanentMagnetMaterial(
            material_id="mat_pm",
            name="PM",
            mu=1.05 * MU0,
            br=(0.0, 0.0, 1.2),
        ),
    )

    regions = (
        Region(region_id="air", name="External Air", material_id="mat_air", is_external=True),
        Region(region_id="magnet", name="Magnet", material_id="mat_pm", is_external=False),
    )

    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="magnet_air_interface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
        )
    )

    return MagnetostaticProblem(
        problem_id="case_assembly",
        name="Assembly test case",
        materials=materials,
        regions=regions,
        surface_mesh=mesh,
        topology=topology,
        external_field=ExternalField(h_ext=(0.0, 0.0, 0.0)),
    )


def test_build_trace_spaces():
    problem = make_problem()
    config = SolverConfig()

    phi, flux = build_trace_spaces(problem, config)

    assert phi.ndofs == 3
    assert flux.ndofs == 1


def test_build_assembly_context():
    problem = make_problem()
    config = SolverConfig()

    ctx = build_assembly_context(problem, config)

    assert ctx.problem.problem_id == "case_assembly"
    assert ctx.phi_space.ndofs == 3
    assert ctx.flux_space.ndofs == 1


def test_build_placeholder_operators():
    problem = make_problem()
    config = SolverConfig()
    ctx = build_assembly_context(problem, config)

    ops = build_placeholder_operators(ctx)

    assert set(ops.keys()) == {"V", "K", "Kt", "D"}
    assert ops["V"].shape == (ctx.phi_space.ndofs, ctx.flux_space.ndofs)
    assert ops["D"].shape == (ctx.flux_space.ndofs, ctx.phi_space.ndofs)


def test_assemble_operator_placeholder():
    problem = make_problem()
    config = SolverConfig()
    ctx = build_assembly_context(problem, config)
    ops = build_placeholder_operators(ctx)

    V = assemble_operator_placeholder(ops["V"])
    assert V.shape == (ctx.phi_space.ndofs, ctx.flux_space.ndofs)
    assert np.allclose(V, 0.0)


def test_assemble_multitrace_system_placeholder():
    problem = make_problem()
    config = SolverConfig()

    system = assemble_multitrace_system_placeholder(problem, config)

    n_phi = system.spaces["phi"].ndofs
    n_flux = system.spaces["flux"].ndofs

    assert system.matrix.shape == (n_phi + n_flux, n_phi + n_flux)
    assert system.rhs.shape == (n_phi + n_flux,)
    assert system.metadata["placeholder"] is True


def test_solve_surface_bem_placeholder():
    problem = make_problem()
    config = SolverConfig()

    solution = solve_surface_bem_placeholder(problem, config)

    assert solution.problem_id == "case_assembly"
    assert solution.phi_trace_coeffs.shape == (3,)
    assert solution.flux_trace_coeffs.shape == (1,)
    assert solution.solve_result.converged is True
    assert np.isclose(solution.solve_result.residual_norm, 0.0)