from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry, Region

# Рабочая точка магнита ПО ВСЕМУ ОБЪЁМУ (поячеечно). Из решённого поля для каждой ячейки
# магнита: рабочее поле H_op вдоль лёгкой оси (второй квадрант, <0), рабочая индукция B_op
# вдоль оси, коэффициент проницаемости P_c = B_op/(μ₀|H_op|) (дескриптор «высоты» рабочей
# точки — чем выше P_c, тем выше рабочая точка и тем труднее размагнитить), объём ячейки
# (area×L). Плюс ОБЪЁМНО-ВЗВЕШЕННОЕ распределение: какая доля объёма работает в какой точке.
# В 2D поле инвариантно по z ⇒ ячейка сечения = призма объёма area×L с той же рабочей точкой,
# так что это рабочая точка по всему объёму. Структура переносится в 3D (ячейка → тетраэдр).


@dataclass(frozen=True, slots=True)
class MagnetOperatingPoint:
    cell_indices: np.ndarray    # (n_mag,) индексы ячеек магнита
    H_op: np.ndarray            # (n_mag,) рабочее поле вдоль e [А/м] (demag → <0)
    B_op: np.ndarray            # (n_mag,) рабочая индукция вдоль e [Тл]
    permeance: np.ndarray       # (n_mag,) P_c = B_op/(μ0|H_op|), безразм. (>0)
    cell_volume: np.ndarray     # (n_mag,) объём ячейки [м³] = area×L
    knee_field: float           # H_knee(T) [А/м] (<0)
    T: float

    @property
    def total_volume(self) -> float:
        return float(self.cell_volume.sum())

    def volume_weighted_mean_H_op(self) -> float:
        return float(np.average(self.H_op, weights=self.cell_volume))

    def volume_weighted_mean_permeance(self) -> float:
        return float(np.average(self.permeance, weights=self.cell_volume))

    def worst_H_op(self) -> float:
        """Наименьшее (наиболее размагничивающее) рабочее поле по объёму [А/м]."""
        return float(self.H_op.min())

    def volume_fraction_below(self, H: float) -> float:
        """Доля объёма магнита с рабочим полем H_op < H (напр. H=H_knee → доля за коленом)."""
        below = self.cell_volume[self.H_op < float(H)].sum()
        return float(below / self.cell_volume.sum())

    def percentiles_H_op(self, qs) -> np.ndarray:
        """Объёмно-взвешенные перцентили H_op: значение, ниже которого работает доля q объёма."""
        order = np.argsort(self.H_op)
        h = self.H_op[order]
        cw = np.cumsum(self.cell_volume[order])
        cw = cw / cw[-1]
        return np.interp(np.asarray(qs, dtype=float), cw, h)

    def knee_margin(self) -> np.ndarray:
        """Поячеечный запас до колена m = H_op − H_knee(T) [А/м] (m<0 ⇒ за коленом)."""
        return self.H_op - self.knee_field


def magnet_operating_point(
    geometry: MachineGeometry,
    result,
    magnet: AnisotropicBHTMagnet,
    *,
    T: float,
    axis=None,
    axial_length: float | None = None,
    mu0: float = MU0,
) -> MagnetOperatingPoint:
    """
    Построить поле рабочей точки по объёму магнита из решённого поля.
    `result` — объект с `.H_cells` и `.B_cells` (MachineStaticResult / любой 2D-результат).
    H_op = (H_solver·e)/μ0 [А/м]; B_op = B·e [Тл]; P_c = B_op/(μ0|H_op|). Ось — поячеечная
    радиальная (geometry.magnet_easy_axis) или переданная.
    """
    mesh = geometry.mesh
    region = geometry.region
    idx = np.where(region == int(Region.MAGNET))[0]
    ax = geometry.magnet_easy_axis if axis is None else np.asarray(axis, dtype=float)
    axes = ax[idx] if ax.ndim == 2 else np.broadcast_to(ax.reshape(-1), (idx.size, ax.shape[-1]))

    H_op = np.einsum("ij,ij->i", result.H_cells[idx], axes) / float(mu0)
    B_op = np.einsum("ij,ij->i", result.B_cells[idx], axes)
    L = geometry.params.axial_length if axial_length is None else float(axial_length)
    areas = np.array([mesh.cell_area(int(c)) for c in idx], dtype=float)
    denom = mu0 * np.abs(H_op)
    permeance = np.where(denom > 0.0, B_op / denom, np.inf)
    return MagnetOperatingPoint(
        cell_indices=idx,
        H_op=H_op,
        B_op=B_op,
        permeance=permeance,
        cell_volume=areas * L,
        knee_field=float(magnet.knee_field(T)),
        T=float(T),
    )
