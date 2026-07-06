from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0


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


# 1 Oe = 1e6/(4*pi) A/m  (т.е. 1 кЭ = 79577.47 А/м); 1 Гс = 1e-4 Тл.
_KOE_TO_A_PER_M: float = 1.0e6 / (4.0 * np.pi)
_GAUSS_TO_TESLA: float = 1.0e-4


def demag_curve_from_datasheet(
    curve_id: str,
    name: str,
    Br: float,
    Hcb: float,
    Hk: float,
    Hcj: float,
    n_below: int = 33,
    n_above: int = 33,
    temperature_c: float | None = None,
) -> DemagnetizationCurveBH:
    """
    НОРМАЛЬНАЯ кривая размагничивания B(H) (2-й квадрант) из 4 даташит-параметров (СИ):
    Br [Тл], |H_cB|, |H_k|, |H_cJ| [А/м] (положительные модули).

    Форма C¹ (см. docs/math/nonlinear_materials.md §4):
      mu_rec = Br/(mu0*Hcb);  s_J = mu0*(mu_rec-1)  (интринзик-наклон);
      интринзик J(H): прямая J=Br+s_J*H выше колена H_k=-|H_k|; ниже — парабола
        a*H^2+b*H+c, КАСАТЕЛЬНАЯ к прямой в колене (C^1) и J(-|H_cJ|)=0;
      нормальная кривая B(H)=J(H)+mu0*H  (выше колена даёт recoil Br+mu0*mu_rec*H,
      ноль при H=-|H_cB|; ниже колена уходит в минус — это нормально для нормальной
      кривой высококоэрцитивных магнитов).

    Совпадает с независимой инженерной Excel-методикой (прямая + касательная парабола,
    2 температурных коэффициента). Чистый NumPy.
    """
    if not (Br > 0.0):
        raise ValueError("Br must be positive.")
    if not (Hcb > 0.0 and Hk > 0.0 and Hcj > 0.0):
        raise ValueError("Hcb, Hk, Hcj must be positive magnitudes [A/m].")
    if not (Hk < Hcj):
        raise ValueError("Knee |H_k| must be below intrinsic coercivity |H_cJ|.")

    mu_rec = Br / (MU0 * Hcb)
    s_J = MU0 * (mu_rec - 1.0)
    Hk_s = -Hk
    Hcj_s = -Hcj
    J_k = Br + s_J * Hk_s
    d = Hk_s - Hcj_s  # = Hcj - Hk > 0
    a = -(J_k - s_J * d) / (d * d)
    b = s_J - 2.0 * a * Hk_s
    c = J_k - a * Hk_s * Hk_s - b * Hk_s

    H_below = np.linspace(Hcj_s, Hk_s, n_below)
    H_above = np.linspace(Hk_s, 0.0, n_above)
    H = np.concatenate([H_below[:-1], H_above])
    J = np.where(H >= Hk_s, Br + s_J * H, a * H * H + b * H + c)
    B = J + MU0 * H
    B = np.maximum.accumulate(B)  # страховка от микро-немонотонности у краёв

    return DemagnetizationCurveBH(
        curve_id=curve_id,
        name=name,
        H_values=H,
        B_values=B,
        temperature_c=temperature_c,
    )


def demag_curve_from_datasheet_cgs(
    curve_id: str,
    name: str,
    Br_gauss: float,
    Hcb_kOe: float,
    Hk_kOe: float,
    Hcj_kOe: float,
    temperature_c: float | None = None,
) -> DemagnetizationCurveBH:
    """
    Как `demag_curve_from_datasheet`, но даташит в CGS (Гс, кЭ) — удобно для ввода
    с паспортов магнитов. 1 Гс = 1e-4 Тл, 1 кЭ = 1e6/(4*pi) А/м.
    """
    return demag_curve_from_datasheet(
        curve_id,
        name,
        Br=Br_gauss * _GAUSS_TO_TESLA,
        Hcb=Hcb_kOe * _KOE_TO_A_PER_M,
        Hk=Hk_kOe * _KOE_TO_A_PER_M,
        Hcj=Hcj_kOe * _KOE_TO_A_PER_M,
        temperature_c=temperature_c,
    )