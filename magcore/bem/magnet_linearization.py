from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.domain.magnet_curves import DemagnetizationCurveBH


@dataclass(frozen=True, slots=True)
class MagnetLinearizationConfig:
    """
    Configuration of zonewise magnet linearization.

    mode:
        "tangent"      -> mu_eff = dB/dH at current operating point
        "fixed_recoil" -> mu_eff = fixed value from fixed_recoil_mu map
    """
    mode: str = "tangent"
    fixed_recoil_mu: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"tangent", "fixed_recoil"}:
            raise ValueError("mode must be 'tangent' or 'fixed_recoil'.")
        if self.mode == "fixed_recoil" and self.fixed_recoil_mu is None:
            raise ValueError("fixed_recoil_mu must be provided when mode='fixed_recoil'.")


@dataclass(frozen=True, slots=True)
class ZoneOperatingPoint:
    zone_id: str
    curve_id: str
    H_parallel: float
    B_parallel: float
    mu_eff: float
    B_src_scalar: float
    B_src_vector: np.ndarray
    easy_axis: np.ndarray

    def __post_init__(self) -> None:
        a = np.asarray(self.easy_axis, dtype=float)
        bsrc = np.asarray(self.B_src_vector, dtype=float)

        object.__setattr__(self, "easy_axis", a)
        object.__setattr__(self, "B_src_vector", bsrc)

        if a.shape != (3,):
            raise ValueError("easy_axis must have shape (3,).")
        if bsrc.shape != (3,):
            raise ValueError("B_src_vector must have shape (3,).")
        if not np.isfinite(a).all():
            raise ValueError("easy_axis must be finite.")
        if not np.isfinite(bsrc).all():
            raise ValueError("B_src_vector must be finite.")
        if not np.isfinite(self.H_parallel):
            raise ValueError("H_parallel must be finite.")
        if not np.isfinite(self.B_parallel):
            raise ValueError("B_parallel must be finite.")
        if not np.isfinite(self.mu_eff):
            raise ValueError("mu_eff must be finite.")
        if self.mu_eff <= 0.0:
            raise ValueError("mu_eff must be positive.")
        if not np.isfinite(self.B_src_scalar):
            raise ValueError("B_src_scalar must be finite.")


def _normalize_easy_axis(easy_axis: np.ndarray) -> np.ndarray:
    a = np.asarray(easy_axis, dtype=float)
    if a.shape != (3,):
        raise ValueError("easy_axis must have shape (3,).")
    if not np.isfinite(a).all():
        raise ValueError("easy_axis must be finite.")

    nrm = float(np.linalg.norm(a))
    if nrm == 0.0:
        raise ValueError("easy_axis must be nonzero.")
    return a / nrm


def _mu_eff_from_curve(
    curve: DemagnetizationCurveBH,
    H_parallel: float,
    cfg: MagnetLinearizationConfig,
) -> float:
    if cfg.mode == "tangent":
        return float(curve.slope_dBdH(H_parallel))

    raise RuntimeError("Use fixed-recoil path with explicit curve_id lookup.")


def linearize_zone_from_curve(
    zone_id: str,
    curve: DemagnetizationCurveBH,
    H_parallel: float,
    easy_axis: np.ndarray,
    cfg: MagnetLinearizationConfig,
) -> ZoneOperatingPoint:
    """
    Linearize one magnetic zone at a current operating field H_parallel.

    Along the easy axis:
        B_parallel = B(H_parallel)

    Current-iteration linearized equivalent:
        B_parallel ≈ mu_eff * H_parallel + B_src_scalar

    and vector source term:
        B_src_vector = B_src_scalar * a
    """
    a = _normalize_easy_axis(easy_axis)
    H_parallel = float(H_parallel)
    if not np.isfinite(H_parallel):
        raise ValueError("H_parallel must be finite.")

    B_parallel = float(curve.B_of_H(H_parallel))

    if cfg.mode == "tangent":
        mu_eff = float(curve.slope_dBdH(H_parallel))
    elif cfg.mode == "fixed_recoil":
        mu_map = cfg.fixed_recoil_mu or {}
        if curve.curve_id not in mu_map:
            raise KeyError(f"Missing fixed recoil mu for curve_id='{curve.curve_id}'.")
        mu_eff = float(mu_map[curve.curve_id])
    else:
        raise ValueError("Unsupported linearization mode.")

    if mu_eff <= 0.0:
        raise ValueError("mu_eff must be positive.")

    B_src_scalar = float(B_parallel - mu_eff * H_parallel)
    B_src_vector = B_src_scalar * a

    return ZoneOperatingPoint(
        zone_id=zone_id,
        curve_id=curve.curve_id,
        H_parallel=H_parallel,
        B_parallel=B_parallel,
        mu_eff=mu_eff,
        B_src_scalar=B_src_scalar,
        B_src_vector=B_src_vector,
        easy_axis=a,
    )


def build_zone_state_map(
    zone_specs,
    curve_map: dict[str, DemagnetizationCurveBH],
    H_parallel_map: dict[str, float],
    cfg: MagnetLinearizationConfig,
) -> dict[str, ZoneOperatingPoint]:
    """
    Build current operating states for all magnetic zones.

    Parameters
    ----------
    zone_specs:
        Iterable of objects with fields:
            zone_id, curve_id, easy_axis
    curve_map:
        curve_id -> DemagnetizationCurveBH
    H_parallel_map:
        zone_id -> current operating H_parallel
    cfg:
        linearization config
    """
    state_map: dict[str, ZoneOperatingPoint] = {}

    for zone in zone_specs:
        if zone.curve_id not in curve_map:
            raise KeyError(f"Missing curve for curve_id='{zone.curve_id}'.")
        if zone.zone_id not in H_parallel_map:
            raise KeyError(f"Missing operating H value for zone_id='{zone.zone_id}'.")

        curve = curve_map[zone.curve_id]
        H_parallel = float(H_parallel_map[zone.zone_id])

        state_map[zone.zone_id] = linearize_zone_from_curve(
            zone_id=zone.zone_id,
            curve=curve,
            H_parallel=H_parallel,
            easy_axis=np.asarray(zone.easy_axis, dtype=float),
            cfg=cfg,
        )

    return state_map