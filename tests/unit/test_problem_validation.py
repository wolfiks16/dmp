from __future__ import annotations

import numpy as np

from magcore.domain.fields import ExternalField
from magcore.domain.interfaces import InterfacePatch
from magcore.domain.materials import AirMaterial, PermanentMagnetMaterial
from magcore.domain.problem import MagnetostaticProblem
from magcore.domain.regions import Region
from magcore.domain.validation_checks import validate_problem
from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.topology import RegionTopology


MU0 = 4.0e-7 * 3.141592653589793


def make_valid_problem() -> MagnetostaticProblem:
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
        Region(
            region_id="air",
            name="External Air",
            material_id="mat_air",
            is_external=True,
        ),
        Region(
            region_id="magnet",
            name="Magnet",
            material_id="mat_pm",
            is_external=False,
        ),
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
        problem_id="case_001",
        name="Minimal valid case",
        materials=materials,
        regions=regions,
        surface_mesh=mesh,
        topology=topology,
        external_field=ExternalField(h_ext=(0.0, 0.0, 0.0)),
    )


def test_valid_problem_has_no_errors():
    problem = make_valid_problem()
    report = validate_problem(problem)

    assert not report.has_errors()
    assert len(report.errors()) == 0


def test_valid_problem_maps_are_accessible():
    problem = make_valid_problem()

    assert problem.get_region("air").name == "External Air"
    assert problem.get_region("magnet").material_id == "mat_pm"
    assert problem.get_material("mat_air").name == "Air"
    assert problem.external_region().region_id == "air"


def test_valid_problem_raise_if_errors_does_not_raise():
    problem = make_valid_problem()
    report = validate_problem(problem)

    report.raise_if_errors()