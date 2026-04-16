from __future__ import annotations

import numpy as np

from magcore.preprocess.magnet_generators import (
    generate_halbach_ring_zone_axes,
    generate_radial_ring_zone_axes,
    make_segmented_ring_zone_specs,
)


def test_generate_radial_ring_zone_axes_returns_unit_vectors():
    axes = generate_radial_ring_zone_axes(n_sectors=8)

    assert len(axes) == 8
    for a in axes:
        assert a.shape == (3,)
        assert np.isclose(np.linalg.norm(a), 1.0)
        assert np.isclose(a[2], 0.0)


def test_generate_radial_ring_zone_axes_adjacent_axes_differ():
    axes = generate_radial_ring_zone_axes(n_sectors=8)

    diffs = [np.linalg.norm(axes[i] - axes[(i + 1) % 8]) for i in range(8)]
    assert all(d > 0.0 for d in diffs)


def test_generate_halbach_ring_zone_axes_returns_unit_vectors():
    axes = generate_halbach_ring_zone_axes(n_sectors=12, pole_pairs=2, rotation_sign=+1)

    assert len(axes) == 12
    for a in axes:
        assert a.shape == (3,)
        assert np.isclose(np.linalg.norm(a), 1.0)
        assert np.isclose(a[2], 0.0)


def test_generate_halbach_ring_zone_axes_changes_with_rotation_sign():
    axes_pos = generate_halbach_ring_zone_axes(n_sectors=8, pole_pairs=1, rotation_sign=+1)
    axes_neg = generate_halbach_ring_zone_axes(n_sectors=8, pole_pairs=1, rotation_sign=-1)

    mismatch = sum(
        np.linalg.norm(a1 - a2) > 1.0e-12
        for a1, a2 in zip(axes_pos, axes_neg, strict=False)
    )
    assert mismatch > 0


def test_make_segmented_ring_zone_specs_radial():
    region_ids = ("r0", "r1", "r2", "r3")
    zones = make_segmented_ring_zone_specs(
        region_ids=region_ids,
        curve_id="curve_1",
        generator="radial",
        zone_prefix="rad",
    )

    assert len(zones) == 4
    assert tuple(z.region_id for z in zones) == region_ids
    assert tuple(z.curve_id for z in zones) == ("curve_1",) * 4
    assert zones[0].zone_id.startswith("rad_")


def test_make_segmented_ring_zone_specs_halbach():
    region_ids = ("r0", "r1", "r2", "r3", "r4", "r5")
    zones = make_segmented_ring_zone_specs(
        region_ids=region_ids,
        curve_id="curve_h",
        generator="halbach",
        pole_pairs=2,
        rotation_sign=+1,
        zone_prefix="hb",
    )

    assert len(zones) == 6
    assert tuple(z.region_id for z in zones) == region_ids
    for z in zones:
        a = np.asarray(z.easy_axis, dtype=float)
        assert np.isclose(np.linalg.norm(a), 1.0)


def test_make_segmented_ring_zone_specs_rejects_unknown_generator():
    try:
        make_segmented_ring_zone_specs(
            region_ids=("r0", "r1"),
            curve_id="curve",
            generator="unknown",
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for unknown generator."