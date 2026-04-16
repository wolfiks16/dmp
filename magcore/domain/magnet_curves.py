from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class DemagnetizationCurveBH:
    curve_id: str
    name: str
    H_values: np.ndarray
    B_values: np.ndarray
    temperature_c: float | None = None

    def __post_init__(self) -> None:
        H = np.asarray(self.H_values, dtype=float)
        B = np.asarray(self.B_values, dtype=float)

        object.__setattr__(self, "H_values", H)
        object.__setattr__(self, "B_values", B)

        self.validate()

    def validate(self) -> None:
        H = self.H_values
        B = self.B_values

        if H.ndim != 1 or B.ndim != 1:
            raise ValueError("H_values and B_values must be 1D arrays.")
        if len(H) != len(B):
            raise ValueError("H_values and B_values must have the same length.")
        if len(H) < 2:
            raise ValueError("Demagnetization curve must contain at least two points.")
        if not np.isfinite(H).all() or not np.isfinite(B).all():
            raise ValueError("Curve arrays must contain only finite values.")

        dH = np.diff(H)
        if not np.all(dH > 0.0):
            raise ValueError("H_values must be strictly increasing.")

        # We work with second-quadrant style data, but do not enforce a rigid sign box
        # because vendor data may include a small range around zero.
        if np.any(np.diff(B) < 0.0):
            raise ValueError("B_values must be monotone nondecreasing with H_values.")

    @property
    def n_points(self) -> int:
        return len(self.H_values)

    @property
    def H_min(self) -> float:
        return float(self.H_values[0])

    @property
    def H_max(self) -> float:
        return float(self.H_values[-1])

    def clamp_H(self, H: float) -> float:
        return float(np.clip(H, self.H_min, self.H_max))

    def segment_index(self, H: float) -> int:
        Hc = self.clamp_H(H)
        idx = int(np.searchsorted(self.H_values, Hc, side="right") - 1)
        return max(0, min(idx, self.n_points - 2))

    def B_of_H(self, H: float) -> float:
        Hc = self.clamp_H(H)
        return float(np.interp(Hc, self.H_values, self.B_values))

    def slope_dBdH(self, H: float) -> float:
        idx = self.segment_index(H)
        h0 = self.H_values[idx]
        h1 = self.H_values[idx + 1]
        b0 = self.B_values[idx]
        b1 = self.B_values[idx + 1]
        return float((b1 - b0) / (h1 - h0))


def demag_curve_from_br_hcb_hcj(
    curve_id: str,
    name: str,
    Br: float,
    HcB: float,
    HcJ: float,
    n_points: int = 64,
    temperature_c: float | None = None,
) -> DemagnetizationCurveBH:
    """
    Build an approximate second-quadrant B(H) curve from datasheet parameters.

    Policy:
    - use Br at H = 0
    - use HcB as the point where B = 0
    - optionally extend slightly toward HcJ with a softened tail if HcJ > HcB

    This is an import/helper representation, not the canonical source of truth.
    """
    if n_points < 4:
        raise ValueError("n_points must be at least 4.")
    if Br <= 0.0:
        raise ValueError("Br must be positive.")
    if HcB <= 0.0 or HcJ <= 0.0:
        raise ValueError("HcB and HcJ must be positive.")
    if HcJ < HcB:
        raise ValueError("Expected HcJ >= HcB.")

    # Use second-quadrant convention: H is negative in the demag region.
    H_left = -HcJ
    H_right = 0.0
    H = np.linspace(H_left, H_right, n_points, dtype=float)

    B = np.empty_like(H)

    if np.isclose(HcJ, HcB):
        # essentially linear recoil-like curve
        slope = Br / HcB
        B[:] = Br + slope * H
    else:
        # piecewise: steeper tail near intrinsic coercivity, linear toward Br
        H_knee = -HcB
        slope_main = Br / HcB

        for i, h in enumerate(H):
            if h >= H_knee:
                B[i] = Br + slope_main * h
            else:
                # soften toward near-zero B in the far-left tail
                t = (h - H_left) / (H_knee - H_left)
                t = float(np.clip(t, 0.0, 1.0))
                B_knee = 0.0
                B_tail = 0.0
                B[i] = (1.0 - t) * B_tail + t * B_knee

    # enforce monotonicity
    B = np.maximum.accumulate(B)

    return DemagnetizationCurveBH(
        curve_id=curve_id,
        name=name,
        H_values=H,
        B_values=B,
        temperature_c=temperature_c,
    )