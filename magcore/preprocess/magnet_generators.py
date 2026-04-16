from __future__ import annotations

import math
import numpy as np

from magcore.domain.magnet_zones import MagnetZoneSpec


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        raise ValueError("Vector must be nonzero.")
    return v / nrm


def generate_radial_ring_zone_axes(
    n_sectors: int,
    clockwise: bool = False,
    z_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, ...]:
    """
    Generate radial easy-axis directions in the xy-plane for a segmented ring.
    """
    if n_sectors < 1:
        raise ValueError("n_sectors must be positive.")

    sign = -1.0 if clockwise else 1.0
    axes = []

    for k in range(n_sectors):
        theta = sign * 2.0 * math.pi * (k + 0.5) / n_sectors
        a = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
        axes.append(_unit(a))

    return tuple(axes)


def generate_halbach_ring_zone_axes(
    n_sectors: int,
    pole_pairs: int = 1,
    rotation_sign: int = +1,
) -> tuple[np.ndarray, ...]:
    """
    Generate piecewise-constant easy axes for a segmented 2D Halbach-like ring
    in the xy-plane.

    This is a preprocessing generator only. The solver still sees ordinary zones
    with constant easy axes.
    """
    if n_sectors < 1:
        raise ValueError("n_sectors must be positive.")
    if pole_pairs < 1:
        raise ValueError("pole_pairs must be positive.")
    if rotation_sign not in (-1, +1):
        raise ValueError("rotation_sign must be -1 or +1.")

    axes = []
    for k in range(n_sectors):
        theta = 2.0 * math.pi * (k + 0.5) / n_sectors
        alpha = rotation_sign * pole_pairs * theta
        a = np.array([math.cos(alpha), math.sin(alpha), 0.0], dtype=float)
        axes.append(_unit(a))

    return tuple(axes)


def make_segmented_ring_zone_specs(
    region_ids: tuple[str, ...],
    curve_id: str,
    generator: str = "radial",
    pole_pairs: int = 1,
    clockwise: bool = False,
    rotation_sign: int = +1,
    zone_prefix: str = "zone",
) -> tuple[MagnetZoneSpec, ...]:
    """
    Build MagnetZoneSpec entries for a segmented ring.

    Parameters
    ----------
    region_ids:
        One solver region per magnetic sector.
    curve_id:
        Demagnetization curve shared by the zones.
    generator:
        "radial" or "halbach"
    """
    n = len(region_ids)
    if n < 1:
        raise ValueError("region_ids must be nonempty.")

    if generator == "radial":
        axes = generate_radial_ring_zone_axes(n_sectors=n, clockwise=clockwise)
    elif generator == "halbach":
        axes = generate_halbach_ring_zone_axes(
            n_sectors=n,
            pole_pairs=pole_pairs,
            rotation_sign=rotation_sign,
        )
    else:
        raise ValueError("generator must be 'radial' or 'halbach'.")

    zones = []
    for k, (rid, a) in enumerate(zip(region_ids, axes, strict=False)):
        zones.append(
            MagnetZoneSpec(
                zone_id=f"{zone_prefix}_{k:03d}",
                region_id=rid,
                curve_id=curve_id,
                easy_axis=(float(a[0]), float(a[1]), float(a[2])),
            )
        )

    return tuple(zones)