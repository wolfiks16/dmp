from __future__ import annotations

import numpy as np

from magcore.bem.magnet_zone_maps import (
    build_face_region_maps,
    build_face_zone_maps,
    build_mu_side_vectors,
    build_source_jump_vector,
    zone_face_groups,
)
from magcore.bem.magnet_linearization import MagnetLinearizationConfig, build_zone_state_map
from magcore.domain.fields import ExternalField
from magcore.domain.interfaces import InterfacePatch
from magcore.domain.magnet_curves import DemagnetizationCurveBH
from magcore.domain.magnet_zones import MagnetAssemblySpec, MagnetZoneSpec
from magcore.domain.materials import AirMaterial
from magcore.domain.problem import MagnetostaticProblem
from magcore.domain.regions import Region
from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.topology import RegionTopology


MU0 = 4.0e-7 * 3.141592653589793


def make_problem_two_internal_regions() -> MagnetostaticProblem:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
            [0.0, 0.0, -1.0], # 4
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],  # face 0 : magnet1 / air
            [0, 1, 3],  # face 1 : magnet1 / air
            [1, 2, 3],  # face 2 : magnet2 / air
            [2, 0, 4],  # face 3 : magnet2 / air
        ],
        dtype=int,
    )
    mesh = SurfaceMesh(vertices=vertices, faces=faces)

    materials = (
        AirMaterial(material_id="mat_air", name="Air", mu=MU0),
    )
    regions = (
        Region(region_id="air", name="Air", material_id="mat_air", is_external=True),
        Region(region_id="mag1", name="Magnet 1", material_id="mat_air", is_external=False),
        Region(region_id="mag2", name="Magnet 2", material_id="mat_air", is_external=False),
    )
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="mag1_air",
                region_minus_id="mag1",
                region_plus_id="air",
                face_indices=(0, 1),
            ),
            InterfacePatch(
                patch_id="p2",
                name="mag2_air",
                region_minus_id="mag2",
                region_plus_id="air",
                face_indices=(2, 3),
            ),
        )
    )

    return MagnetostaticProblem(
        problem_id="zone_map_case",
        name="Zone map case",
        materials=materials,
        regions=regions,
        surface_mesh=mesh,
        topology=topology,
        external_field=ExternalField(h_ext=(0.0, 0.0, 0.0)),
    )


def make_magnet_assembly() -> MagnetAssemblySpec:
    z1 = MagnetZoneSpec(
        zone_id="z1",
        region_id="mag1",
        curve_id="curve1",
        easy_axis=(1.0, 0.0, 0.0),
    )
    z2 = MagnetZoneSpec(
        zone_id="z2",
        region_id="mag2",
        curve_id="curve1",
        easy_axis=(0.0, 1.0, 0.0),
    )
    return MagnetAssemblySpec(zones=(z1, z2))


def make_zone_state_map():
    curve = DemagnetizationCurveBH(
        curve_id="curve1",
        name="test curve",
        H_values=np.array([-1000.0, -500.0, 0.0], dtype=float),
        B_values=np.array([0.2, 0.8, 1.2], dtype=float),
        temperature_c=20.0,
    )
    asm = make_magnet_assembly()
    cfg = MagnetLinearizationConfig(mode="tangent")

    return build_zone_state_map(
        zone_specs=asm.zones,
        curve_map={"curve1": curve},
        H_parallel_map={"z1": -750.0, "z2": -500.0},
        cfg=cfg,
    )


def test_build_face_region_maps():
    problem = make_problem_two_internal_regions()
    fr = build_face_region_maps(problem, (0, 1, 2, 3))

    assert fr.face_indices == (0, 1, 2, 3)
    assert fr.region_minus_ids == ("mag1", "mag1", "mag2", "mag2")
    assert fr.region_plus_ids == ("air", "air", "air", "air")


def test_build_face_zone_maps():
    problem = make_problem_two_internal_regions()
    asm = make_magnet_assembly()

    fzm = build_face_zone_maps(problem, asm, (0, 1, 2, 3))

    assert fzm.face_indices == (0, 1, 2, 3)
    assert fzm.zone_minus_ids == ("z1", "z1", "z2", "z2")
    assert fzm.zone_plus_ids == (None, None, None, None)
    assert fzm.normals.shape == (4, 3)


def test_zone_face_groups():
    problem = make_problem_two_internal_regions()
    asm = make_magnet_assembly()

    groups = zone_face_groups(problem, asm, (0, 1, 2, 3))

    assert groups["z1"] == (0, 1)
    assert groups["z2"] == (2, 3)


def test_build_mu_side_vectors():
    problem = make_problem_two_internal_regions()
    asm = make_magnet_assembly()
    states = make_zone_state_map()

    fzm = build_face_zone_maps(problem, asm, (0, 1, 2, 3))
    mu_minus, mu_plus = build_mu_side_vectors(fzm, states, mu_air=MU0)

    assert mu_minus.shape == (4,)
    assert mu_plus.shape == (4,)
    assert np.all(mu_minus > 0.0)
    assert np.allclose(mu_plus, MU0)


def test_build_source_jump_vector():
    problem = make_problem_two_internal_regions()
    asm = make_magnet_assembly()
    states = make_zone_state_map()

    fzm = build_face_zone_maps(problem, asm, (0, 1, 2, 3))
    s = build_source_jump_vector(fzm, states)

    assert s.shape == (4,)
    assert np.isfinite(s).all()


def test_build_face_region_maps_rejects_missing_face_in_topology():
    problem = make_problem_two_internal_regions()

    try:
        build_face_region_maps(problem, (0, 99))
    except KeyError:
        assert True
        return

    assert False, "Expected KeyError for face not covered by topology."