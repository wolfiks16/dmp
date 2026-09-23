from __future__ import annotations

import numpy as np

from magcore.constants import MU0

# Закон магнита в ПЛАНАРНОЙ постановке (неизвестное — A_z, значит индукция) для метода Ньютона.
# Вдоль лёгкой оси — рабочая ветвь закона с памятью (общая модель магнита: при новой необратимой
# потере главная кривая, иначе линия возврата с сохранённой долей r), поперёк — постоянная материала
# μ⊥. Касательная берётся у ТОЙ ЖЕ ветви, поэтому итерация ньютоновская: внешнего цикла по источнику
# и подбора релаксации не нужно (Л-21, Л-93). За коленом главная кривая в разы круче линии возврата,
# и замороженный источник с релаксацией там не сжимает — это и было причиной несходимости в 2D.


class MagnetLaw2D:
    """
    H(B) и касательная dH/dB по ячейкам магнита. Единицы решателя: H_реш = μ₀·H [Тл], ν — относительная.

    `axis` — ось лёгкого намагничивания: один вектор (2,) на весь магнит или поячеечный (n_cells, 2)
    (реальная машина: радиальная ось·полярность). `retention` — сохранённая доля ремнантности r ∈ [0, 1]
    после прежних нагружений (массив по ВСЕМ ячейкам; вне магнита не используется); None — новый магнит.
    """

    def __init__(self, magnet, magnet_mask, n_cells: int, *, T: float, axis, retention=None, mu0: float = MU0):
        mask = np.asarray(magnet_mask, dtype=bool).reshape(-1)
        if mask.shape != (int(n_cells),):
            raise ValueError("magnet_mask must have shape (n_cells,).")
        self.magnet = magnet
        self.idx = np.where(mask)[0]
        self.T = float(T)
        self.mu0 = float(mu0)
        ax = np.asarray(magnet.easy_axis[:2] if axis is None else axis, dtype=float)
        if ax.ndim == 1:
            e = np.broadcast_to(ax.reshape(1, -1), (self.idx.size, ax.size)).astype(float)
        elif ax.ndim == 2 and ax.shape[0] == int(n_cells):
            e = ax[self.idx].astype(float)
        else:
            raise ValueError("axis must be (2,) or (n_cells, 2).")
        if e.shape[1] != 2:
            raise ValueError("ось намагничивания в планарной задаче — вектор из двух чисел.")
        norm = np.linalg.norm(e, axis=1)
        if ax.ndim == 1 and self.idx.size and not np.all(norm > 0.0):
            raise ValueError("ось намагничивания не должна быть нулевой.")
        # Поячеечная ось бывает не определена в отдельной ячейке (радиальное намагничивание в центре):
        # там намагниченности нет, тело ведёт себя как линейное с μ_rec — как и в прежней схеме.
        self.defined = norm > 0.0
        self.axes = e / np.where(self.defined, norm, 1.0)[:, None]
        self.nu_rec = 1.0 / magnet.mu_rec
        self.nu_perp = 1.0 / magnet.mu_perp
        self.retention = (np.ones(self.idx.size, dtype=float) if retention is None
                          else np.asarray(retention, dtype=float).reshape(-1)[self.idx])
        if np.any(self.retention < 0.0) or np.any(self.retention > 1.0):
            raise ValueError("сохранённая доля ремнантности r должна лежать в [0, 1].")
        self.H_par = np.zeros(self.idx.size, dtype=float)      # рабочее поле вдоль оси [А/м]
        self.slope = np.full(self.idx.size, mu0 * magnet.mu_rec, dtype=float)   # наклон ветви [Тл·м/А]

    def __call__(self, B_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """По полю B в ячейках магнита: H [Тл] (n_mag, 2) и касательная dH/dB (n_mag, 2, 2)."""
        if self.idx.size == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0, 2, 2), dtype=float)
        B = np.asarray(B_cells, dtype=float)[self.idx]
        b_par = np.einsum("ij,ij->i", B, self.axes)
        h_par, slope = self.magnet.branch_parallel_inverse(b_par, self.T, self.retention)
        self.H_par, self.slope = h_par, slope
        b_perp = B - b_par[:, None] * self.axes
        H = (self.mu0 * h_par)[:, None] * self.axes + self.nu_perp * b_perp
        ee = self.axes[:, :, None] * self.axes[:, None, :]
        eye = np.broadcast_to(np.eye(2), (self.idx.size, 2, 2))
        D = (self.mu0 / slope)[:, None, None] * ee + self.nu_perp * (eye - ee)
        if not np.all(self.defined):                       # ячейки без направления оси — линейный μ_rec
            H[~self.defined] = self.nu_rec * B[~self.defined]
            D[~self.defined] = self.nu_rec * eye[~self.defined]
        return H, D

    def retention_now(self) -> np.ndarray:
        """Доля ремнантности, которую оставило текущее поле (для истории нагружения): min(r, r_now)."""
        r_now = np.asarray(self.magnet.retention_now(self.H_par, self.T), dtype=float)
        return np.minimum(self.retention, r_now)
