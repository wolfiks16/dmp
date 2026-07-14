from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry

# Раскладка трёхфазной обмотки по пазам: какой фазе (A/B/C) и с каким знаком (направление
# тока ±z) принадлежит каждый паз. Нужна для P3 (ток якоря из параметров катушки → J_z).
# Здесь — только РАЗМЕТКА (геометрия обмотки), сами токи/поле — в P3.

PHASE_NAMES = ("A", "B", "C")


@dataclass(frozen=True)
class WindingLayout:
    """Фаза (0=A,1=B,2=C) и знак (±1) каждого паза. Однослойная (n_layers=1) в первой версии."""

    n_slots: int
    phase_of_slot: np.ndarray      # (n_slots,) в {0,1,2}
    sign_of_slot: np.ndarray       # (n_slots,) в {+1,-1}
    n_layers: int = 1

    def __post_init__(self) -> None:
        ph = np.asarray(self.phase_of_slot, dtype=int).reshape(-1)
        sg = np.asarray(self.sign_of_slot, dtype=int).reshape(-1)
        object.__setattr__(self, "phase_of_slot", ph)
        object.__setattr__(self, "sign_of_slot", sg)
        if ph.shape != (self.n_slots,) or sg.shape != (self.n_slots,):
            raise ValueError("phase_of_slot/sign_of_slot должны иметь длину n_slots.")
        if not set(np.unique(ph)).issubset({0, 1, 2}):
            raise ValueError("phase_of_slot ∈ {0,1,2}.")
        if not set(np.unique(sg)).issubset({-1, 1}):
            raise ValueError("sign_of_slot ∈ {-1,+1}.")

    def phase_slot_counts(self) -> np.ndarray:
        """Число пазов на фазу (для проверки симметрии — ожидается n_slots/3)."""
        return np.array([int((self.phase_of_slot == m).sum()) for m in range(3)])

    def is_balanced(self) -> bool:
        """Существенная симметрия обмотки: равное число пазов на каждую фазу."""
        counts = self.phase_slot_counts()
        return bool(np.all(counts == counts[0]))


def star_of_slots_layout(n_slots: int, n_poles: int) -> WindingLayout:
    """
    Симметричная ОДНОСЛОЙНАЯ раскладка по «звезде пазов»: паз i относят к фазе с наибольшей
    проекцией ЭДС-фазора (угол i·p·2π/Ns) на оси фаз A/B/C (0/120/240°), знак = знак
    проекции. Для дробных обмоток (напр. 12/14) часть пазов лежит на границе (двухслойная
    природа) — эти неоднозначные пазы разрешаются ТОЧНЫМ перебором до сбалансированной
    раскладки (равное число пазов на фазу + нулевой суммарный знак). ⚠ однослойная,
    первая версия; реальную/двухслойную раскладку можно задать явно через `WindingLayout`.
    """
    if n_slots < 3 or n_poles < 2 or n_poles % 2 != 0:
        raise ValueError("n_slots>=3 и n_poles>=2 чётное.")
    if n_slots % 3 != 0:
        raise ValueError("n_slots должно делиться на 3 (три фазы).")
    p = n_poles // 2
    proj = np.array([
        [math.cos(i * p * 2.0 * math.pi / n_slots - m * 2.0 * math.pi / 3.0) for m in range(3)]
        for i in range(n_slots)
    ])

    fixed_phase: dict[int, int] = {}
    fixed_sign: dict[int, int] = {}
    ties: list[tuple[int, list[int]]] = []
    for i in range(n_slots):
        ap = np.abs(proj[i])
        cand = [m for m in range(3) if ap[m] >= ap.max() - 1e-9]
        if len(cand) == 1:
            m = cand[0]
            fixed_phase[i] = m
            fixed_sign[i] = 1 if proj[i, m] >= 0 else -1
        else:
            ties.append((i, cand))

    if len(ties) > 16:
        raise ValueError("слишком много неоднозначных пазов — задайте раскладку явно.")

    target = n_slots // 3
    fallback: WindingLayout | None = None   # первая счёт-сбалансированная (если знак-баланс недостижим)
    for choice in product(*[cand for _, cand in ties]):
        phase = dict(fixed_phase)
        sign = dict(fixed_sign)
        for (i, _), m in zip(ties, choice):
            phase[i] = m
            sign[i] = 1 if proj[i, m] >= 0 else -1
        counts = [sum(phase[i] == m for i in range(n_slots)) for m in range(3)]
        if not all(c == target for c in counts):
            continue
        ph = np.array([phase[i] for i in range(n_slots)], dtype=int)
        sg = np.array([sign[i] for i in range(n_slots)], dtype=int)
        lay = WindingLayout(n_slots=n_slots, phase_of_slot=ph, sign_of_slot=sg)
        # Предпочитаем знак-сбалансированную (нулевой суммарный знак фазы = чище go/return).
        if all(sg[ph == m].sum() == 0 for m in range(3)):
            return lay
        if fallback is None:
            fallback = lay

    if fallback is not None:
        return fallback
    raise ValueError(
        "не удалось построить сбалансированную однослойную раскладку для %d/%d — "
        "задайте её явно через WindingLayout (возможно, нужна двухслойная)."
        % (n_slots, n_poles)
    )


def cell_phase_sign(
    geometry: MachineGeometry, layout: WindingLayout
) -> tuple[np.ndarray, np.ndarray]:
    """
    Поячеечная разметка обмотки: (phase_of_cell, sign_of_cell). Для ячеек паза — фаза/знак
    его паза; вне пазов phase=-1, sign=0. Основа для сборки токового источника J_z (P3).
    """
    if layout.n_slots != geometry.params.n_slots:
        raise ValueError("layout.n_slots не совпадает с геометрией.")
    nc = geometry.mesh.n_cells
    phase = np.full(nc, -1, dtype=int)
    sign = np.zeros(nc, dtype=int)
    sid = geometry.slot_id
    in_slot = sid >= 0
    phase[in_slot] = layout.phase_of_slot[sid[in_slot]]
    sign[in_slot] = layout.sign_of_slot[sid[in_slot]]
    return phase, sign
