from __future__ import annotations

import numpy as np

from magcore.domain.magnet_zones import MagnetAssemblySpec, MagnetZoneSpec


def test_magnet_zone_normalizes_easy_axis():
    zone = MagnetZoneSpec(
        zone_id="z1",
        region_id="magnet_region_1",
        curve_id="curve_1",
        easy_axis=(0.0, 0.0, 5.0),
    )

    a = zone.easy_axis_array
    assert np.allclose(a, [0.0, 0.0, 1.0])
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_magnet_zone_rejects_zero_easy_axis():
    try:
        MagnetZoneSpec(
            zone_id="z1",
            region_id="magnet_region_1",
            curve_id="curve_1",
            easy_axis=(0.0, 0.0, 0.0),
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for zero easy axis."


def test_magnet_zone_rejects_wrong_axis_shape():
    try:
        MagnetZoneSpec(
            zone_id="z1",
            region_id="magnet_region_1",
            curve_id="curve_1",
            easy_axis=(1.0, 0.0),
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong axis shape."


def test_magnet_assembly_zone_map_and_lookup():
    z1 = MagnetZoneSpec(
        zone_id="z1",
        region_id="r1",
        curve_id="c1",
        easy_axis=(1.0, 0.0, 0.0),
    )
    z2 = MagnetZoneSpec(
        zone_id="z2",
        region_id="r2",
        curve_id="c1",
        easy_axis=(0.0, 1.0, 0.0),
    )

    asm = MagnetAssemblySpec(zones=(z1, z2))

    zm = asm.zone_map()
    assert set(zm.keys()) == {"z1", "z2"}
    assert asm.zone_by_region_id("r1") == z1
    assert asm.zone_by_region_id("r2") == z2
    assert asm.zone_by_region_id("missing") is None


def test_magnet_assembly_rejects_duplicate_zone_ids():
    z1 = MagnetZoneSpec(
        zone_id="z",
        region_id="r1",
        curve_id="c1",
        easy_axis=(1.0, 0.0, 0.0),
    )
    z2 = MagnetZoneSpec(
        zone_id="z",
        region_id="r2",
        curve_id="c1",
        easy_axis=(0.0, 1.0, 0.0),
    )

    try:
        MagnetAssemblySpec(zones=(z1, z2))
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for duplicate zone_id."


def test_magnet_assembly_rejects_duplicate_region_ids():
    z1 = MagnetZoneSpec(
        zone_id="z1",
        region_id="r",
        curve_id="c1",
        easy_axis=(1.0, 0.0, 0.0),
    )
    z2 = MagnetZoneSpec(
        zone_id="z2",
        region_id="r",
        curve_id="c1",
        easy_axis=(0.0, 1.0, 0.0),
    )

    try:
        MagnetAssemblySpec(zones=(z1, z2))
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for duplicate region_id."


def test_magnet_assembly_ids_accessors():
    z1 = MagnetZoneSpec(
        zone_id="z1",
        region_id="r1",
        curve_id="c1",
        easy_axis=(1.0, 0.0, 0.0),
    )
    z2 = MagnetZoneSpec(
        zone_id="z2",
        region_id="r2",
        curve_id="c2",
        easy_axis=(0.0, 1.0, 0.0),
    )

    asm = MagnetAssemblySpec(zones=(z1, z2))

    assert asm.zone_ids() == ("z1", "z2")
    assert asm.region_ids() == ("r1", "r2")