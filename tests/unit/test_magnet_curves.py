from __future__ import annotations

import numpy as np

from magcore.domain.magnet_curves import (
    DemagnetizationCurveBH,
    demag_curve_from_br_hcb_hcj,
)


def make_linear_curve() -> DemagnetizationCurveBH:
    H = np.array([-1000.0, -500.0, 0.0], dtype=float)
    B = np.array([0.2, 0.8, 1.2], dtype=float)
    return DemagnetizationCurveBH(
        curve_id="c1",
        name="linear demag",
        H_values=H,
        B_values=B,
        temperature_c=20.0,
    )


def test_demagnetization_curve_basic_properties():
    curve = make_linear_curve()

    assert curve.n_points == 3
    assert np.isclose(curve.H_min, -1000.0)
    assert np.isclose(curve.H_max, 0.0)


def test_B_of_H_interpolates_linearly():
    curve = make_linear_curve()

    val = curve.B_of_H(-750.0)
    assert np.isclose(val, 0.5)


def test_slope_dBdH_returns_segment_slope():
    curve = make_linear_curve()

    slope_left = curve.slope_dBdH(-900.0)
    slope_right = curve.slope_dBdH(-100.0)

    assert np.isclose(slope_left, (0.8 - 0.2) / (500.0))
    assert np.isclose(slope_right, (1.2 - 0.8) / (500.0))


def test_clamp_H_clips_to_bounds():
    curve = make_linear_curve()

    assert np.isclose(curve.clamp_H(-5000.0), curve.H_min)
    assert np.isclose(curve.clamp_H(100.0), curve.H_max)


def test_segment_index_returns_valid_index():
    curve = make_linear_curve()

    idx1 = curve.segment_index(-900.0)
    idx2 = curve.segment_index(-100.0)

    assert idx1 == 0
    assert idx2 == 1


def test_B_of_H_clamps_outside_range():
    curve = make_linear_curve()

    assert np.isclose(curve.B_of_H(-5000.0), curve.B_values[0])
    assert np.isclose(curve.B_of_H(100.0), curve.B_values[-1])


def test_curve_rejects_nonmonotone_H():
    H = np.array([-1000.0, -200.0, -500.0], dtype=float)
    B = np.array([0.2, 0.8, 1.2], dtype=float)

    try:
        DemagnetizationCurveBH(
            curve_id="bad_H",
            name="bad H",
            H_values=H,
            B_values=B,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for nonmonotone H_values."


def test_curve_rejects_decreasing_B():
    H = np.array([-1000.0, -500.0, 0.0], dtype=float)
    B = np.array([0.2, 1.2, 0.8], dtype=float)

    try:
        DemagnetizationCurveBH(
            curve_id="bad_B",
            name="bad B",
            H_values=H,
            B_values=B,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for decreasing B_values."


def test_curve_rejects_mismatched_lengths():
    H = np.array([-1000.0, -500.0, 0.0], dtype=float)
    B = np.array([0.2, 1.2], dtype=float)

    try:
        DemagnetizationCurveBH(
            curve_id="bad_len",
            name="bad len",
            H_values=H,
            B_values=B,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for mismatched array lengths."


def test_demag_curve_from_br_hcb_hcj_builds_valid_curve():
    curve = demag_curve_from_br_hcb_hcj(
        curve_id="imported",
        name="datasheet curve",
        Br=1.2,
        HcB=900000.0,
        HcJ=1200000.0,
        n_points=64,
        temperature_c=20.0,
    )

    assert curve.n_points == 64
    assert np.isfinite(curve.H_values).all()
    assert np.isfinite(curve.B_values).all()
    assert np.all(np.diff(curve.H_values) > 0.0)
    assert np.all(np.diff(curve.B_values) >= 0.0)


def test_demag_curve_from_br_hcb_hcj_hits_Br_at_zero():
    curve = demag_curve_from_br_hcb_hcj(
        curve_id="imported2",
        name="datasheet curve 2",
        Br=1.1,
        HcB=800000.0,
        HcJ=800000.0,
        n_points=32,
    )

    assert np.isclose(curve.B_of_H(0.0), 1.1)