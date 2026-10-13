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

    `T` — температура: одно число на весь магнит или поячеечно (n_cells,) — тогда у каждой ячейки кривая
    своей температуры (транзиент К6′: магнит прогрет неравномерно). `T_bin` — шаг сетки температур: T ячейки
    округляется до T_bin·round(T/T_bin), и кривые строятся только в узлах сетки — как в `IrreversibleMagnetState`
    ядра К6′, поэтому оба метода решают один и тот же закон; погрешность закона ≤ |dB_r/dT|·T_bin/2 и → 0 с
    T_bin. None — точные значения (своя кривая на каждое различное T). Ветвь при каждой температуре считает
    модель магнита (`branch_parallel_inverse`) — реализация одна (Л-92, Л-100).
    """

    def __init__(self, magnet, magnet_mask, n_cells: int, *, T, axis, retention=None, mu0: float = MU0,
                 T_bin: float | None = None):
        mask = np.asarray(magnet_mask, dtype=bool).reshape(-1)
        if mask.shape != (int(n_cells),):
            raise ValueError("magnet_mask must have shape (n_cells,).")
        self.magnet = magnet
        self.idx = np.where(mask)[0]
        T_arr = np.asarray(T, dtype=float)
        if T_arr.ndim == 0:
            self.T = float(T_arr)
            t = np.full(self.idx.size, self.T, dtype=float)
        elif T_arr.shape == (int(n_cells),):
            self.T = T_arr
            t = T_arr[self.idx].astype(float)
        else:
            raise ValueError("T — одно число или по ячейкам (n_cells,).")
        if not np.all(np.isfinite(t)):
            raise ValueError("температура магнита должна быть конечной.")
        if T_bin is not None:
            if not float(T_bin) > 0.0:
                raise ValueError("T_bin must be positive.")
            t = np.round(t / float(T_bin)) * float(T_bin)
        self.T_cells = t                                       # температура закона по ячейкам магнита
        # Ячейки одной температуры — одна кривая: группы строятся один раз, T закона за время его жизни постоянна.
        temps, inv = np.unique(t, return_inverse=True)
        self._groups = [(float(Tg), np.where(inv == k)[0]) for k, Tg in enumerate(temps)]
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
        h_par = np.empty_like(b_par)
        slope = np.empty_like(b_par)
        for Tg, sel in self._by_temperature():
            h_par[sel], slope[sel] = self.magnet.branch_parallel_inverse(b_par[sel], Tg, self.retention[sel])
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

    def _by_temperature(self):
        """
        Группы ячеек одной температуры: (T, выборка). Одна группа — выборка `slice(None)`: вызов на полном массиве,
        ровно как при одной T на весь магнит (бит в бит).
        """
        if len(self._groups) == 1:
            return [(self._groups[0][0], slice(None))]
        return self._groups

    def retention_now(self) -> np.ndarray:
        """
        Доля ремнантности, которую оставило текущее поле (для истории нагружения): min(r, r_now). У ячейки без
        направления оси намагниченности нет — терять нечего, доля остаётся прежней.
        """
        r_now = np.empty_like(self.H_par)
        for Tg, sel in self._by_temperature():
            r_now[sel] = self.magnet.retention_now(self.H_par[sel], Tg)
        r_now = np.where(self.defined, r_now, 1.0)
        return np.minimum(self.retention, r_now)

    def beyond_hcj(self) -> np.ndarray:
        """
        Ячейки, где текущее поле ниже −H_cJ своей температуры (H∥ < −H_cJ): модель магнита там не определена,
        закон доопределён консервативно — полная потеря (r = 0). Ячейки без оси — нет.
        """
        below = np.zeros(self.H_par.shape, dtype=bool)
        for Tg, sel in self._by_temperature():
            below[sel] = self.H_par[sel] < -self.magnet.Hcj(Tg)
        return below & self.defined


class MagnetLaws2D:
    """
    Закон магнитов нескольких марок в одной задаче: у каждой марки свой `MagnetLaw2D` на своих ячейках. Для
    решателя — то же, что один закон: `idx` — ячейки всех марок подряд, H и касательная — в том же порядке.
    """

    def __init__(self, laws):
        self.laws = list(laws)
        if not self.laws:
            raise ValueError("нужен хотя бы один закон марки.")
        self.idx = np.concatenate([law.idx for law in self.laws])
        if np.unique(self.idx).size != self.idx.size:
            raise ValueError("ячейки марок пересекаются — у ячейки должен быть один закон.")

    def __call__(self, B_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parts = [law(B_cells) for law in self.laws]
        return np.concatenate([H for H, _ in parts]), np.concatenate([D for _, D in parts])

    def retention_now(self) -> np.ndarray:
        """Доля ремнантности после текущего поля — по ячейкам `idx`, у каждой марки по её кривой."""
        return np.concatenate([law.retention_now() for law in self.laws])
