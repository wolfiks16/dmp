from __future__ import annotations

import numpy as np
import pytest

from magcore.domain.fields import ExternalField
from magcore.domain.interfaces import InterfacePatch
from magcore.domain.materials import (
    AirMaterial,
    LinearMagneticMaterial,
    PermanentMagnetMaterial,
)
from magcore.domain.problem import MagnetostaticProblem
from magcore.domain.regions import Region
from magcore.domain.validation_checks import validate_problem
from magcore.exceptions import DomainValidationError
from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.topology import RegionTopology


MU0 = 4.0e-7 * 3.141592653589793


def issue_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def make_mesh_one_face() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return SurfaceMesh(vertices=vertices, faces=faces)


def make_mesh_two_faces() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
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


def make_materials_duplicate_ids():
    return (
        AirMaterial(material_id="dup", name="Air A", mu=MU0),
        LinearMagneticMaterial(material_id="dup", name="Linear B", mu=1.2 * MU0),
    )


def make_materials_standard():
    return (
        AirMaterial(material_id="mat_air", name="Air", mu=MU0),
        PermanentMagnetMaterial(
            material_id="mat_pm",
            name="PM",
            mu=1.05 * MU0,
            br=(0.0, 0.0, 1.2),
        ),
    )


def make_regions_standard():
    return (
        Region(region_id="air", name="Air", material_id="mat_air", is_external=True),
        Region(region_id="magnet", name="Magnet", material_id="mat_pm", is_external=False),
    )


def make_problem(
    *,
    materials,
    regions,
    mesh,
    topology,
    problem_id="p",
    name="problem",
    external_field=(0.0, 0.0, 0.0),
) -> MagnetostaticProblem:
    return MagnetostaticProblem(
        problem_id=problem_id,
        name=name,
        materials=materials,
        regions=regions,
        surface_mesh=mesh,
        topology=topology,
        external_field=ExternalField(h_ext=external_field),
    )


def test_duplicate_material_ids_are_reported():
    mesh = make_mesh_one_face()
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
        )
    )
    regions = (
        Region(region_id="air", name="Air", material_id="dup", is_external=True),
        Region(region_id="magnet", name="Magnet", material_id="dup", is_external=False),
    )

    problem = make_problem(
        materials=make_materials_duplicate_ids(),
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "materials.ids.duplicate" in issue_codes(report)


def test_duplicate_region_ids_are_reported():
    mesh = make_mesh_one_face()
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="air",
                region_plus_id="magnet",
                face_indices=(0,),
            ),
        )
    )
    materials = make_materials_standard()
    regions = (
        Region(region_id="dup", name="Air", material_id="mat_air", is_external=True),
        Region(region_id="dup", name="Magnet", material_id="mat_pm", is_external=False),
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "regions.ids.duplicate" in issue_codes(report)


def test_missing_material_reference_is_reported():
    mesh = make_mesh_one_face()
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
        )
    )
    materials = (
        AirMaterial(material_id="mat_air", name="Air", mu=MU0),
    )
    regions = (
        Region(region_id="air", name="Air", material_id="mat_air", is_external=True),
        Region(region_id="magnet", name="Magnet", material_id="mat_missing", is_external=False),
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "regions.material.missing" in issue_codes(report)


def test_missing_external_region_is_reported():
    mesh = make_mesh_one_face()
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
        )
    )
    materials = make_materials_standard()
    regions = (
        Region(region_id="air", name="Air", material_id="mat_air", is_external=False),
        Region(region_id="magnet", name="Magnet", material_id="mat_pm", is_external=False),
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "regions.external.count" in issue_codes(report)


def test_two_external_regions_are_reported():
    mesh = make_mesh_one_face()
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
        )
    )
    materials = make_materials_standard()
    regions = (
        Region(region_id="air", name="Air", material_id="mat_air", is_external=True),
        Region(region_id="magnet", name="Magnet", material_id="mat_pm", is_external=True),
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "regions.external.count" in issue_codes(report)


def test_degenerate_face_is_reported():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    mesh = SurfaceMesh(vertices=vertices, faces=faces)

    materials = make_materials_standard()
    regions = make_regions_standard()
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
        )
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "mesh.face.degenerate" in issue_codes(report)


def test_patch_face_out_of_range_is_reported():
    mesh = make_mesh_one_face()
    materials = make_materials_standard()
    regions = make_regions_standard()
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(3,),  # out of range
            ),
        )
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "topology.patch.face.out_of_range" in issue_codes(report)


def test_uncovered_mesh_faces_are_reported():
    mesh = make_mesh_two_faces()
    materials = make_materials_standard()
    regions = make_regions_standard()
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),  # face 1 uncovered
            ),
        )
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "topology.faces.uncovered" in issue_codes(report)


def test_repeated_face_assignment_is_reported():
    mesh = make_mesh_two_faces()
    materials = make_materials_standard()
    regions = make_regions_standard()
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface_1",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
            InterfacePatch(
                patch_id="p2",
                name="iface_2",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0, 1),  # face 0 repeated
            ),
        )
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "topology.face.repeated" in issue_codes(report)


def test_external_region_cannot_use_permanent_magnet():
    mesh = make_mesh_one_face()
    materials = (
        PermanentMagnetMaterial(
            material_id="mat_pm",
            name="PM",
            mu=1.05 * MU0,
            br=(0.0, 0.0, 1.2),
        ),
    )
    regions = (
        Region(region_id="air", name="External", material_id="mat_pm", is_external=True),
        Region(region_id="magnet", name="Inner", material_id="mat_pm", is_external=False),
    )
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
        )
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    assert report.has_errors()
    assert "problem.external_region.permanent_magnet" in issue_codes(report)


def test_raise_if_errors_raises_domain_validation_error():
    mesh = make_mesh_one_face()
    materials = make_materials_standard()
    regions = (
        Region(region_id="air", name="Air", material_id="mat_air", is_external=False),
        Region(region_id="magnet", name="Magnet", material_id="mat_pm", is_external=False),
    )
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
        )
    )

    problem = make_problem(
        materials=materials,
        regions=regions,
        mesh=mesh,
        topology=topology,
    )
    report = validate_problem(problem)

    with pytest.raises(DomainValidationError):
        report.raise_if_errors()