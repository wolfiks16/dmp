from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.machines.rotor_sweep import (
    RotorDamage,
    RotorSweepResult,
    electrical_period_angles,
    sweep_rotor,
)

# ПОТЕРИ В ЖЕЛЕЗЕ (статор) по модели Штейнмеца с РАЗДЕЛЕНИЕМ гистерезис/вихревые.
# Именно ради честного расчёта этих потерь вращение делалось ПЕРЕД железом: имея волну B(θ)
# в неподвижных точках статора за электрический период, берём НАСТОЯЩИЙ размах и производную
# поля по элементу, а не оценку «B_peak из одного снимка × синусоида».
#
# Удельные потери [Вт/кг] на элемент:
#   гистерезис  p_h = k_h · f · B_m^α          (B_m — пик |B| за период)
#   вихревые    p_e = k_e · f² · B_m²           (для СИНУСОИДЫ; калибровка коэффициента)
# Но реальная волна в зубце НЕ синусоида (проходящие зубцы дают гармоники), поэтому вихревые
# считаются из фактической производной: классически P_e ∝ ⟨(dB/dt)²⟩. Через безразмерную
# электрическую фазу θ_e = p·θ_mech = 2πf·t это даёт
#   p_e = 2·k_e·f²·⟨(dB/dθ_e)²⟩,
# что для чистой синусоиды B_m·sinθ_e в точности сводится к k_e·f²·B_m² (⟨cos²⟩=½), а на
# гармониках корректно РАСТЁТ (∝ n² по номеру гармоники) — этого снимок дать не может.
#
# ГРАНИЦА: считается железо СТАТОРА (зубцы+ярмо), стационарное в лаборатории. Ротор в СВОЕЙ
# системе видит квазипостоянное поле (едет вместе с магнитами) ⇒ его потери малы и требуют
# иной трактовки (в системе материала); здесь не считаются. Модель гистерезиса — классическая
# скалярная (по пику |B|); вращательный гистерезис (петля под вращающимся B) не разделяется —
# представительное инженерное приближение. Коэффициенты k_h,α,k_e — ПРЕДСТАВИТЕЛЬНЫЕ, сверять
# с кривой удельных потерь конкретной марки.

_STATOR_IRON = (int(Region.STATOR_YOKE), int(Region.TOOTH))


@dataclass(frozen=True, slots=True)
class SteinmetzCoefficients:
    """
    Коэффициенты Штейнмеца [удельные потери в Вт/кг, f в Гц, B в Тл] + плотность стали.

    p = k_hyst·f·B^alpha + k_eddy·f²·B²  (сепарация гистерезис/вихревые).
    """

    k_hyst: float          # Вт·с/(кг·Тл^alpha)
    alpha: float           # показатель Штейнмеца (обычно 1.6…2.2)
    k_eddy: float          # Вт·с²/(кг·Тл²)
    density: float         # плотность стали [кг/м³]

    def __post_init__(self) -> None:
        for nm in ("k_hyst", "k_eddy", "density"):
            if getattr(self, nm) <= 0.0:
                raise ValueError(f"{nm} must be positive.")
        if not (1.0 <= self.alpha <= 3.0):
            raise ValueError("alpha должен быть в [1, 3].")

    @staticmethod
    def m270_35a() -> "SteinmetzCoefficients":
        """
        Представительный M270-35A. ⚠ ОРИЕНТИРОВОЧНО: подобрано под ~2.3 Вт/кг при 1.5 Тл/50 Гц
        (доля вихревых ~15 %) — сверить с кривой удельных потерь марки перед использованием
        в отчёте. Плотность электротехнической стали ≈ 7650 кг/м³.
        """
        return SteinmetzCoefficients(k_hyst=0.017, alpha=2.0, k_eddy=4.3e-5, density=7650.0)

    def specific_hysteresis(self, B_peak, freq: float):
        """Удельные гистерезисные потери k_h·f·B_m^α [Вт/кг] (векторизуемо по B_peak)."""
        B = np.asarray(B_peak, dtype=float)
        return self.k_hyst * float(freq) * np.power(np.maximum(B, 0.0), self.alpha)

    def specific_eddy_sinusoid(self, B_peak, freq: float):
        """Удельные вихревые потери СИНУСОИДЫ k_e·f²·B_m² [Вт/кг] (эталон калибровки)."""
        B = np.asarray(B_peak, dtype=float)
        return self.k_eddy * float(freq) ** 2 * B * B


def electrical_frequency(params: OutrunnerPMSMParams, speed_rpm: float) -> float:
    """Электрическая частота f = p·n/60 [Гц] по механической скорости [об/мин]."""
    return (params.n_poles // 2) * float(speed_rpm) / 60.0


def stator_iron_probes(params: OutrunnerPMSMParams):
    """
    Неподвижные пробные точки в железе СТАТОРА (центроиды его ячеек эталонной геометрии)
    и их площади. B в этих точках снимается на каждом угле ротора (см. `sweep_rotor`).
    Возвращает (points:(P,2), areas:(P,)).
    """
    geo = build_outrunner_spm_pmsm(params)
    idx = np.where(np.isin(geo.region, _STATOR_IRON))[0]
    pts = np.array([geo.mesh.cell_centroid(int(c)) for c in idx], dtype=float)
    areas = np.array([geo.mesh.cell_area(int(c)) for c in idx], dtype=float)
    return pts, areas


def eddy_specific_from_waveform(B_series: np.ndarray, freq: float,
                                coeffs: SteinmetzCoefficients) -> np.ndarray:
    """
    Удельные ВИХРЕВЫЕ потери [Вт/кг] по элементу из ВОЛНЫ B(θ_e) за ОДИН электрический период.

    `B_series` — (N, P, 2): B в P точках на N РАВНОМЕРНЫХ углах ровно одного электрического
    периода (θ_e = 0…2π). p_e = 2·k_e·f²·⟨(dB/dθ_e)²⟩, производная — центральной разностью на
    ПЕРИОДИЧЕСКОЙ сетке (обе компоненты B). Захватывает гармоники зубцовой волны.
    """
    B = np.asarray(B_series, dtype=float)
    n = B.shape[0]
    dtheta = 2.0 * math.pi / n
    dB = (np.roll(B, -1, axis=0) - np.roll(B, 1, axis=0)) / (2.0 * dtheta)   # (N,P,2), периодич.
    mean_sq = (dB ** 2).sum(axis=2).mean(axis=0)                             # ⟨(dB/dθ_e)²⟩ по точке
    return 2.0 * coeffs.k_eddy * float(freq) ** 2 * mean_sq


@dataclass(frozen=True, slots=True)
class IronLossResult:
    """Потери в железе статора за электрический период."""

    hysteresis: float             # суммарные гистерезисные потери [Вт]
    eddy: float                   # суммарные вихревые потери [Вт]
    freq: float                   # электрическая частота [Гц]
    iron_mass: float              # масса железа статора [кг]

    @property
    def total(self) -> float:
        return float(self.hysteresis + self.eddy)


def iron_loss_from_probe_waveform(
    B_series: np.ndarray, probe_areas: np.ndarray, *, freq: float,
    axial_length: float, coeffs: SteinmetzCoefficients,
) -> IronLossResult:
    """
    Потери в железе из готовой волны B(θ_e) в пробных точках статора.

    `B_series` — (N,P,2) за один электрический период; `probe_areas` — (P,) площади ячеек.
    Масса элемента = ρ·area·L; удельные потери × масса, суммируются по точкам.
    """
    B = np.asarray(B_series, dtype=float)
    areas = np.asarray(probe_areas, dtype=float).reshape(-1)
    if B.ndim != 3 or B.shape[1] != areas.size or B.shape[2] != 2:
        raise ValueError("B_series должно быть (N, P, 2) с P = число площадей.")

    Bmag = np.hypot(B[:, :, 0], B[:, :, 1])          # |B|(θ) по точкам
    B_peak = Bmag.max(axis=0)                         # пик за период
    p_h = coeffs.specific_hysteresis(B_peak, freq)             # Вт/кг
    p_e = eddy_specific_from_waveform(B, freq, coeffs)         # Вт/кг

    mass = coeffs.density * areas * float(axial_length)        # кг по элементу
    return IronLossResult(
        hysteresis=float((p_h * mass).sum()),
        eddy=float((p_e * mass).sum()),
        freq=float(freq),
        iron_mass=float(mass.sum()),
    )


def efficiency(torque_mean: float, speed_rpm: float,
               copper_loss_w: float, iron_loss_w: float) -> float:
    """
    КПД двигателя η = P_вых / (P_вых + P_медь + P_железо).

    P_вых = |M_ср|·ω_мех, ω_мех = 2π·n/60. Берётся модуль момента (режим двигателя). Механические
    потери (трение/вентиляция) сюда НЕ входят — их можно добавить в сумму потерь отдельно.
    Потери должны быть в тех же ВАТТАХ, что и мощность (не «на единицу осевой длины»).
    """
    if speed_rpm <= 0.0:
        raise ValueError("speed_rpm must be positive.")
    if copper_loss_w < 0.0 or iron_loss_w < 0.0:
        raise ValueError("потери не могут быть отрицательными.")
    omega = 2.0 * math.pi * float(speed_rpm) / 60.0
    p_out = abs(float(torque_mean)) * omega
    p_in = p_out + float(copper_loss_w) + float(iron_loss_w)
    return float(p_out / p_in) if p_in > 0.0 else 0.0


def stator_iron_loss(
    params: OutrunnerPMSMParams,
    magnet: AnisotropicBHTMagnet,
    steel: SteelBHCurve,
    *,
    speed_rpm: float,
    i_peak: float = 0.0,
    gamma_elec: float = 0.0,
    turns_per_slot: float = 0.0,
    T: float = 20.0,
    damage: RotorDamage | None = None,
    coeffs: SteinmetzCoefficients | None = None,
    n_positions: int = 24,
    relaxation: float = 0.1,
    max_iter: int = 300,
) -> tuple[IronLossResult, RotorSweepResult]:
    """
    Потери в железе статора: прогнать ротор на один электрический период, снять волну B в
    неподвижных точках статора, применить Штейнмец. Возвращает (потери, результат прогонки —
    там же момент/пульсации, чтобы не гонять сетку дважды).

    Скорость `speed_rpm` задаёт частоту f = p·n/60. `damage` — повреждение магнита, если
    считаем машину ПОСЛЕ перегрузки. Один электрический период достаточен для СИММЕТРИЧНОЙ
    (целой) машины; при несимметричном повреждении картина ротора всё равно p-периодична по
    механике, а частота поля в статоре = электрическая, поэтому период тот же.
    """
    cf = coeffs or SteinmetzCoefficients.m270_35a()
    pts, areas = stator_iron_probes(params)
    angles = electrical_period_angles(params, int(n_positions), periods=1)

    sweep = sweep_rotor(
        params, magnet, steel, angles=angles, i_peak=i_peak, gamma_elec=gamma_elec,
        turns_per_slot=turns_per_slot, T=T, damage=damage, no_load=False,
        probe_points=pts, relaxation=relaxation, max_iter=max_iter,
    )
    freq = electrical_frequency(params, speed_rpm)
    loss = iron_loss_from_probe_waveform(
        sweep.probe_B, areas, freq=freq, axial_length=params.axial_length, coeffs=cf
    )
    return loss, sweep
