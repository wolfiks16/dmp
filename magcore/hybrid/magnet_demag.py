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
    """
    Карта риска размагничивания по ячейкам магнита (deliverable новизны 2).

    В модели из нескольких марок колено и номинальная B_r у каждой ячейки свои (своей марки) —
    `knee_field_cells`, `Br_nominal_cells`; общие скаляры `knee_field`, `Br_nominal` есть, только когда
    они одни на все ячейки, иначе None — чтобы код, ждущий одну марку, не считал молча по чужому колену.
    """

    cell_indices: np.ndarray   # (n_mag,) индексы ячеек магнита
    H_par: np.ndarray          # (n_mag,) рабочее поле вдоль e [А/м] (demag → < 0)
    margin: np.ndarray         # (n_mag,) маржа m = H_par − H_knee(T) своей марки; m<0 ⇒ за коленом
    Br_eff: np.ndarray         # (n_mag,) эффективная ремнантность [Тл]
    loss: np.ndarray           # (n_mag,) необратимая потеря Br(T) − Br_eff [Тл] (≥0)
    demagnetized: np.ndarray   # (n_mag,) bool: margin < 0
    T: float
    Br_nominal: float | None   # Br(T) без потерь [Тл]; None — в карте марки с разной Br(T)
    knee_field: float | None   # H_knee(T) [А/м] (<0); None — в карте марки с разным коленом
    retention: np.ndarray | None = None    # (n_mag,) сохранённая доля ремнантности r_eff = min(история, r_now) —
                                           # по ней считана потеря: Br_eff = r_eff·Br(T) (Л-100)
    beyond_hcj: np.ndarray | None = None   # (n_mag,) bool: r_eff = 0 — поле хоть раз было ниже −H_cJ; модель
                                           # магнита там не определена, потеря принята полной (как стоп «каскад» в 2D)
    Br_nominal_cells: np.ndarray | None = None   # (n_mag,) Br(T) марки ячейки [Тл]; не задано — из общего скаляра
    knee_field_cells: np.ndarray | None = None   # (n_mag,) H_knee(T) марки ячейки [А/м]; не задано — из скаляра

    def __post_init__(self) -> None:
        n = np.asarray(self.cell_indices).size
        for cells, common in (("Br_nominal_cells", self.Br_nominal), ("knee_field_cells", self.knee_field)):
            if getattr(self, cells) is None:
                if common is None:
                    raise ValueError(f"{cells}: общего значения на все ячейки нет — нужно значение по ячейкам.")
                object.__setattr__(self, cells, np.full(n, float(common)))
            elif np.asarray(getattr(self, cells)).shape != (n,):
                raise ValueError(f"{cells} — по числу на ячейку карты.")

    @property
    def n_damaged(self) -> int:
        """Ячеек с необратимой потерей — сейчас или на прежних нагружениях, при любой температуре: r_eff < 1."""
        if self.retention is None:
            return int(np.count_nonzero(self.H_par < self.knee_field_cells))
        return int(np.count_nonzero(self.retention < 1.0))

    @property
    def n_beyond_hcj(self) -> int:
        return 0 if self.beyond_hcj is None else int(np.count_nonzero(self.beyond_hcj))

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
    retention=None,
) -> DemagRiskMap:
    """
    Построить карту риска из сошедшегося решения: на каждой ячейке магнита взять
    рабочее поле H_par (мост H_solver/μ₀), маржу к колену, сохранённую долю ремнантности
    и необратимую потерю. См. docs/math/nonlinear_materials.md §7.

    `result` — любой объект с полем `.H_cells` (3D CoupledPicardResult или 2D-результат).
    `axis` — ось проекции (по умолчанию 3D easy_axis; для 2D передаётся плоскостная).
    `retention` — для решения с историей нагружения: сохранённая доля ремнантности r ∈ [0, 1]
    после прежних нагружений (массив по всем ячейкам; вне магнита не используется). Потеря
    считается по r_eff = min(r, r_now(H∥ сейчас, T)) — магнит, вернувшийся по линии возврата,
    сохраняет потерю, в том числе при другой температуре (Л-100); маржа и `demagnetized` — по
    текущему полю («за коленом сейчас»). None — истории нет, потеря по текущему полю.
    """
    mask = np.asarray(magnet_mask, dtype=bool).reshape(-1)
    idx = np.where(mask)[0]
    ax = np.asarray(magnet.easy_axis if axis is None else axis, dtype=float)
    axes = ax[idx] if ax.ndim == 2 else np.broadcast_to(ax.reshape(-1), (idx.size, ax.shape[-1]))

    h_par = np.einsum("ij,ij->i", result.H_cells[idx], axes) / float(mu0)   # А/м
    margin = np.asarray(magnet.risk_margin(h_par, T), dtype=float)
    r_eff = np.asarray(magnet.retention_now(h_par, T), dtype=float)
    if retention is not None:
        r_eff = np.minimum(r_eff, np.asarray(retention, dtype=float).reshape(-1)[idx])
    br_nom = float(magnet.Br(T))
    br_eff = r_eff * br_nom
    return DemagRiskMap(
        cell_indices=idx,
        H_par=h_par,
        margin=margin,
        Br_eff=br_eff,
        loss=br_nom - br_eff,
        demagnetized=margin < 0.0,
        T=float(T),
        Br_nominal=br_nom,
        knee_field=float(magnet.knee_field(T)),
        retention=r_eff,
        beyond_hcj=r_eff == 0.0,
    )


def merge_risk_maps(maps) -> DemagRiskMap:
    """
    Карта риска модели из нескольких марок: карты марок (у каждой свои ячейки) — в одну, ячейки по
    возрастанию номера, как в `magnet_mask`. Колено и номинальная B_r остаются по ячейкам — у каждой марки
    свои; общий скаляр — только если он у всех марок один, иначе None.
    """
    maps = list(maps)
    if not maps:
        raise ValueError("нет карт для объединения.")
    if len({m.T for m in maps}) != 1:
        raise ValueError("карты марок посчитаны при разных температурах.")
    for opt in ("retention", "beyond_hcj"):
        if len({getattr(m, opt) is None for m in maps}) != 1:
            raise ValueError(f"{opt} есть не у всех карт марок.")
    idx = np.concatenate([m.cell_indices for m in maps])
    if np.unique(idx).size != idx.size:
        raise ValueError("ячейки карт марок пересекаются — у ячейки должна быть одна марка.")
    order = np.argsort(idx, kind="stable")

    def cat(name):
        if getattr(maps[0], name) is None:
            return None
        return np.concatenate([getattr(m, name) for m in maps])[order]

    def common(name):
        vals = {getattr(m, name) for m in maps}
        return vals.pop() if len(vals) == 1 else None

    return DemagRiskMap(
        cell_indices=idx[order], H_par=cat("H_par"), margin=cat("margin"), Br_eff=cat("Br_eff"),
        loss=cat("loss"), demagnetized=cat("demagnetized"), T=maps[0].T,
        Br_nominal=common("Br_nominal"), knee_field=common("knee_field"),
        retention=cat("retention"), beyond_hcj=cat("beyond_hcj"),
        Br_nominal_cells=cat("Br_nominal_cells"), knee_field_cells=cat("knee_field_cells"),
    )
