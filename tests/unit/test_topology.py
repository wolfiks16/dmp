from __future__ import annotations

from magcore.domain.interfaces import InterfacePatch
from magcore.mesh.topology import RegionTopology


def make_topology() -> RegionTopology:
    return RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface_1",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0, 1),
            ),
            InterfacePatch(
                patch_id="p2",
                name="iface_2",
                region_minus_id="core",
                region_plus_id="air",
                face_indices=(2,),
            ),
        )
    )


def test_patch_map_contains_all_patches():
    topology = make_topology()
    patch_map = topology.patch_map()

    assert set(patch_map.keys()) == {"p1", "p2"}
    assert patch_map["p1"].name == "iface_1"
    assert patch_map["p2"].region_minus_id == "core"


def test_all_face_indices_collects_all_faces():
    topology = make_topology()
    all_faces = topology.all_face_indices()

    assert all_faces == (0, 1, 2)


def test_region_adjacency_is_built_correctly():
    topology = make_topology()
    adj = topology.region_adjacency()

    assert adj["magnet"] == {"air"}
    assert adj["core"] == {"air"}
    assert adj["air"] == {"magnet", "core"}


def test_validate_basic_detects_duplicate_patch_ids():
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="dup",
                name="iface_1",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0,),
            ),
            InterfacePatch(
                patch_id="dup",
                name="iface_2",
                region_minus_id="core",
                region_plus_id="air",
                face_indices=(1,),
            ),
        )
    )

    issues = topology.validate_basic()
    codes = {i.code for i in issues}

    assert "topology.patch_ids.duplicate" in codes


def test_validate_basic_detects_repeated_face_assignment():
    topology = RegionTopology(
        interface_patches=(
            InterfacePatch(
                patch_id="p1",
                name="iface_1",
                region_minus_id="magnet",
                region_plus_id="air",
                face_indices=(0, 1),
            ),
            InterfacePatch(
                patch_id="p2",
                name="iface_2",
                region_minus_id="core",
                region_plus_id="air",
                face_indices=(1, 2),
            ),
        )
    )

    issues = topology.validate_basic()
    codes = {i.code for i in issues}

    assert "topology.face.repeated" in codes