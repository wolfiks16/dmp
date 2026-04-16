from __future__ import annotations

import numpy as np

from magcore.bem.magnet_linearization import (
    MagnetLinearizationConfig,
    build_zone_state_map,
    linearize_zone_from_curve,
)
from magcore.domain.magnet_curves import DemagnetizationCurveBH
from magcore.domain.magnet_zones import MagnetZoneSpec


def make_linear_curve() -> DemagnetizationCurveBH:
    H = np.array([-1000.0, -500.0, 0.0], dtype=float)
    B = np.array([0.2, 0.8, 1.2], dtype=float)
    return DemagnetizationCurveBH(
        curve_id="curve_linear",
        name="linear curve",
        H_values=H,
        B_values=B,
        temperature_c=20.0,
    )


def test_linearize_zone_from_curve_tangent_mode():
    curve = make_linear_curve()
    cfg = MagnetLinearizationConfig(mode="tangent")

    state = linearize_zone_from_curve(
        zone_id="z1",
        curve=curve,
        H_parallel=-750.0,
        easy_axis=np.array([0.0, 0.0, 2.0], dtype=float),
        cfg=cfg,
    )

    expected_B = 0.5
    expected_mu = (0.8 - 0.2) / 500.0
    expected_src = expected_B - expected_mu * (-750.0)

    assert np.isclose(state.B_parallel, expected_B)
    assert np.isclose(state.mu_eff, expected_mu)
    assert np.isclose(state.B_src_scalar, expected_src)
    assert np.allclose(state.easy_axis, [0.0, 0.0, 1.0])
    assert np.allclose(state.B_src_vector, expected_src * np.array([0.0, 0.0, 1.0]))


def test_linearize_zone_from_curve_fixed_recoil_mode():
    curve = make_linear_curve()
    cfg = MagnetLinearizationConfig(
        mode="fixed_recoil",
        fixed_recoil_mu={"curve_linear": 0.002},
    )

    state = linearize_zone_from_curve(
        zone_id="z1",
        curve=curve,
        H_parallel=-750.0,
        easy_axis=np.array([1.0, 0.0, 0.0], dtype=float),
        cfg=cfg,
    )

    expected_B = 0.5
    expected_mu = 0.002
    expected_src = expected_B - expected_mu * (-750.0)

    assert np.isclose(state.B_parallel, expected_B)
    assert np.isclose(state.mu_eff, expected_mu)
    assert np.isclose(state.B_src_scalar, expected_src)
    assert np.allclose(state.B_src_vector, expected_src * np.array([1.0, 0.0, 0.0]))


def test_linearize_zone_from_curve_rejects_zero_axis():
    curve = make_linear_curve()
    cfg = MagnetLinearizationConfig(mode="tangent")

    try:
        linearize_zone_from_curve(
            zone_id="z1",
            curve=curve,
            H_parallel=-500.0,
            easy_axis=np.array([0.0, 0.0, 0.0], dtype=float),
            cfg=cfg,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for zero easy axis."


def test_linearize_zone_from_curve_rejects_nonpositive_mu_eff():
    curve = make_linear_curve()
    cfg = MagnetLinearizationConfig(
        mode="fixed_recoil",
        fixed_recoil_mu={"curve_linear": -1.0},
    )

    try:
        linearize_zone_from_curve(
            zone_id="z1",
            curve=curve,
            H_parallel=-500.0,
            easy_axis=np.array([1.0, 0.0, 0.0], dtype=float),
            cfg=cfg,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for nonpositive mu_eff."


def test_build_zone_state_map_constructs_all_states():
    curve = make_linear_curve()
    zone1 = MagnetZoneSpec(
        zone_id="z1",
        region_id="r1",
        curve_id="curve_linear",
        easy_axis=(1.0, 0.0, 0.0),
    )
    zone2 = MagnetZoneSpec(
        zone_id="z2",
        region_id="r2",
        curve_id="curve_linear",
        easy_axis=(0.0, 1.0, 0.0),
    )

    cfg = MagnetLinearizationConfig(mode="tangent")
    states = build_zone_state_map(
        zone_specs=(zone1, zone2),
        curve_map={"curve_linear": curve},
        H_parallel_map={"z1": -750.0, "z2": -500.0},
        cfg=cfg,
    )

    assert set(states.keys()) == {"z1", "z2"}
    assert np.isfinite(states["z1"].B_parallel)
    assert np.isfinite(states["z2"].B_parallel)


def test_build_zone_state_map_checks_missing_curve():
    zone = MagnetZoneSpec(
        zone_id="z1",
        region_id="r1",
        curve_id="curve_missing",
        easy_axis=(1.0, 0.0, 0.0),
    )
    cfg = MagnetLinearizationConfig(mode="tangent")

    try:
        build_zone_state_map(
            zone_specs=(zone,),
            curve_map={},
            H_parallel_map={"z1": -500.0},
            cfg=cfg,
        )
    except KeyError:
        assert True
        return

    assert False, "Expected KeyError for missing curve."


def test_build_zone_state_map_checks_missing_H_parallel():
    curve = make_linear_curve()
    zone = MagnetZoneSpec(
        zone_id="z1",
        region_id="r1",
        curve_id="curve_linear",
        easy_axis=(1.0, 0.0, 0.0),
    )
    cfg = MagnetLinearizationConfig(mode="tangent")

    try:
        build_zone_state_map(
            zone_specs=(zone,),
            curve_map={"curve_linear": curve},
            H_parallel_map={},
            cfg=cfg,
        )
    except KeyError:
        assert True
        return

    assert False, "Expected KeyError for missing H_parallel."