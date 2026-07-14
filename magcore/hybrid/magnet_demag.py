from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet

if TYPE_CHECKING:  # только для аннотаций: не тянуть 3D-драйвер в runtime (переиспользуемо в 2D)
    from magcore.hybrid.nonlinear import CoupledPicardResult


class MagnetDemagPolicy:
    """
    Состояние-зависимый источник намагниченности для `solve_coupled_nonlinear_picard`
    (ядро новизны 2): магнит с КОЛЕНОМ. Реализует callable `magnetization(B,H,ν)`.

    Для каждой ячейки магнита берёт рабочее поле вдоль лёгкой оси (в физических А/м,
    через мост `H_phys = H_solver/μ₀`), вычисляет эффективную ремнантность
    `B_r_eff = magnet.effective_Br(H_eval, T)` (необратимая потеря ниже колена) и
    возвращает источник решателя `ν·B_r_eff·e` (ν = 1/μ_rec, относительная).

    Семантика H_eval:
      * `track_worst_point=False` (по умолчанию, ОДИНОЧНАЯ статическая нагрузка):
        H_eval = ТЕКУЩЕЕ H_par ⇒ Picard сходится к рабочей точке на ГЛАВНОЙ кривой
        (физически верный equilibrium одиночного нагружения).
      * `track_worst_point=True` (ИСТОРИЯ нагружения, фаза Ca): H_eval = наименьшее
        достигнутое H_par (latching-минимум) ⇒ необратимость сохраняется при снятии поля.

    Под-релаксация `relaxation∈(0,1]` по B_r_eff гасит осцилляции demag-итерации.
    """

    def __init__(
        self,
        magnet: AnisotropicBHTMagnet,
        magnet_mask: np.ndarray,
        T: float,
        n_cells: int,
        *,
        mu0: float = MU0,
        relaxation: float = 0.5,
        track_worst_point: bool = False,
        axis=None,
    ) -> None:
        mask = np.asarray(magnet_mask, dtype=bool).reshape(-1)
        if mask.shape != (n_cells,):
            raise ValueError("magnet_mask must have shape (n_cells,).")
        if not (0.0 < relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1].")
        self.magnet = magnet
        self.mask = mask
        self.idx = np.where(mask)[0]
        self.T = float(T)
        self.n_cells = int(n_cells)
        self.mu0 = float(mu0)
        self.omega = float(relaxation)
        self.track = bool(track_worst_point)
        # Ось проекции. Два режима:
        #   * (dim,)          — ОДНА ось на все ячейки магнита (3D easy_axis / плоская (1,0));
        #   * (n_cells, dim)  — ПОЯЧЕЕЧНАЯ ось (реальная машина: радиальная ось·полярность,
        #                       своя у каждой ячейки — geometry.magnet_easy_axis).
        # Размерность источника наследуется из axis. Вне магнита строки не используются (idx).
        ax = (
            np.asarray(magnet.easy_axis, dtype=float)
            if axis is None
            else np.asarray(axis, dtype=float)
        )
        if ax.ndim == 1:
            self.per_cell = False
            self.axis = ax
            self.dim = int(ax.shape[0])
        elif ax.ndim == 2:
            if ax.shape[0] != n_cells:
                raise ValueError("поячеечная axis должна иметь форму (n_cells, dim).")
            self.per_cell = True
            self.axis = ax
            self.dim = int(ax.shape[1])
        else:
            raise ValueError("axis must be (dim,) or (n_cells, dim).")
        self.nu_rec = 1.0 / magnet.mu_rec
        self.h_worst = np.zeros(n_cells, dtype=float)
        self._br_prev: np.ndarray | None = None

    def _axes_at_idx(self) -> np.ndarray:
        """Оси лёгкого намагничивания в ячейках магнита, (n_mag, dim)."""
        if self.per_cell:
            return self.axis[self.idx]
        return np.broadcast_to(self.axis, (self.idx.size, self.dim))

    def __call__(self, B_cells: np.ndarray, H_cells: np.ndarray, nu_cells: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n_cells, self.dim), dtype=float)
        if self.idx.size == 0:
            return out

        axes = self._axes_at_idx()  # (n_mag, dim), поячеечная ось (или broadcast одной)
        # Рабочее поле вдоль лёгкой оси: H_solver[Тл] · e (построчно), затем мост в А/м.
        h_par_phys = np.einsum("ij,ij->i", H_cells[self.idx], axes) / self.mu0  # (n_mag,)
        if self.track:
            self.h_worst[self.idx] = np.minimum(self.h_worst[self.idx], h_par_phys)
            h_eval = self.h_worst[self.idx]
        else:
            h_eval = h_par_phys

        br_eff = np.asarray(self.magnet.effective_Br(h_eval, self.T), dtype=float)  # Тл
        if self._br_prev is None:
            br = br_eff  # 1-я итерация: H=0 ⇒ номинальная Br(T), релаксация не нужна
        else:
            br = (1.0 - self.omega) * self._br_prev + self.omega * br_eff
        self._br_prev = br

        out[self.idx] = (self.nu_rec * br)[:, None] * axes  # ν·B_r_eff·e [Тл]
        return out


@dataclass(frozen=True)
class DemagRiskMap:
    """Карта риска размагничивания по ячейкам магнита (deliverable новизны 2)."""

    cell_indices: np.ndarray   # (n_mag,) индексы ячеек магнита
    H_par: np.ndarray          # (n_mag,) рабочее поле вдоль e [А/м] (demag → < 0)
    margin: np.ndarray         # (n_mag,) маржа m = H_par − H_knee(T); m<0 ⇒ за коленом
    Br_eff: np.ndarray         # (n_mag,) эффективная ремнантность [Тл]
    loss: np.ndarray           # (n_mag,) необратимая потеря Br(T) − Br_eff [Тл] (≥0)
    demagnetized: np.ndarray   # (n_mag,) bool: margin < 0
    T: float
    Br_nominal: float          # Br(T) без потерь [Тл]
    knee_field: float          # H_knee(T) [А/м] (<0)

    @property
    def n_demagnetized(self) -> int:
        return int(np.count_nonzero(self.demagnetized))

    @property
    def worst_margin(self) -> float:
        return float(self.margin.min()) if self.margin.size else float("nan")

    @property
    def total_loss(self) -> float:
        return float(self.loss.sum())


def compute_demag_risk_map(
    magnet: AnisotropicBHTMagnet,
    result: "CoupledPicardResult",
    magnet_mask: np.ndarray,
    T: float,
    *,
    mu0: float = MU0,
    axis=None,
) -> DemagRiskMap:
    """
    Построить карту риска из сошедшегося решения: на каждой ячейке магнита взять
    рабочее поле H_par (мост H_solver/μ₀), маржу к колену, эффективную ремнантность
    и необратимую потерю. См. docs/math/nonlinear_materials.md §7.

    `result` — любой объект с полем `.H_cells` (3D CoupledPicardResult или 2D-результат).
    `axis` — ось проекции (по умолчанию 3D easy_axis; для 2D передаётся плоскостная).
    """
    mask = np.asarray(magnet_mask, dtype=bool).reshape(-1)
    idx = np.where(mask)[0]
    ax = np.asarray(magnet.easy_axis if axis is None else axis, dtype=float)
    axes = ax[idx] if ax.ndim == 2 else np.broadcast_to(ax.reshape(-1), (idx.size, ax.shape[-1]))

    h_par = np.einsum("ij,ij->i", result.H_cells[idx], axes) / float(mu0)   # А/м
    margin = np.asarray(magnet.risk_margin(h_par, T), dtype=float)
    br_eff = np.asarray(magnet.effective_Br(h_par, T), dtype=float)
    br_nom = float(magnet.Br(T))
    loss = br_nom - br_eff
    return DemagRiskMap(
        cell_indices=idx,
        H_par=h_par,
        margin=margin,
        Br_eff=br_eff,
        loss=loss,
        demagnetized=margin < 0.0,
        T=float(T),
        Br_nominal=br_nom,
        knee_field=float(magnet.knee_field(T)),
    )
