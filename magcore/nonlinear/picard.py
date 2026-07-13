from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class PicardLoopResult:
    """
    Итог размерно-независимого Picard-цикла (общий для 2D/3D backends).

    Содержит только величины, которыми владеет сам цикл: поле B на последней
    итерации, замороженную ν финальной сборки и метаданные сходимости. Всё
    остальное (a, p, ψ, λ, H, источник ν·B_r, невязка) backend сохраняет в своём
    замыкании `step` и собирает в собственный результат после цикла.
    """

    B_cells: np.ndarray            # (n_cells, dim) — B=curl A на ячейку, последняя итерация
    nu_cells: np.ndarray           # (n_cells,) — ν, на которой собрана финальная система
    n_iterations: int
    converged: bool
    rel_change_history: tuple[float, ...]


def resolve_magnetization(magnetization, n_cells: int, *, dim: int = 3) -> Callable:
    """
    Привести источник намагниченности к функции состояния `mag_fn(B,H,ν) -> (n_cells, dim)`.

    Общая для связанного/периодического (и будущего 2D) драйверов:
      * None                      → нулевой источник (магнита нет);
      * массив (n_cells, dim)     → статический ν·B_r (константа между итерациями);
      * callable(B,H,ν)->(n,dim)  → магнит с состоянием (колено/возврат), как есть.
    """
    if magnetization is None:
        def mag_fn(_B, _H, _nu):
            return np.zeros((n_cells, dim), dtype=float)
        return mag_fn
    if callable(magnetization):
        return magnetization
    static_mag = np.asarray(magnetization, dtype=float)
    if static_mag.shape != (n_cells, dim):
        raise ValueError(f"static magnetization must have shape (n_cells, {dim}).")

    def mag_fn(_B, _H, _nu):
        return static_mag

    return mag_fn


def run_picard_fixed_point(
    *,
    nu_init,
    nu_of_B: Callable,
    step: Callable,
    max_iter: int = 50,
    tol: float = 1.0e-6,
    relaxation: float = 1.0,
) -> PicardLoopResult:
    """
    Размерно-независимый хордовый Picard для нелинейной магнитостатики — ядро,
    общее для 2D/3D backends (см. docs/math/nonlinear_materials.md §5, §8).

    Цикл владеет ТОЛЬКО под-релаксацией ν и критерием сходимости по полю B;
    сборку системы, наложение BC, решатель и извлечение B=curl A инкапсулирует
    backend через `step`.

    Параметры
    ---------
    nu_init : (n_cells,)
        Стартовая поячеечная релуктивность.
    nu_of_B : callable(B_cells:(n_cells,dim)) -> (n_cells,)
        Хордовая ν(|B|,…) по ячейке (воздух=const, сталь=SteelBHCurve.nu_chord,
        магнит=1/μ_rec). Объединяет все материальные области.
    step : callable(nu_cells:(n_cells,)) -> B_cells:(n_cells,dim)
        Замкнутая на backend функция: собирает систему с ЗАМОРОЖЕННОЙ ν, решает,
        возвращает B=curl A по ячейке. Любое дополнительное состояние (a, p, ψ, H,
        источник намагниченности, невязка) backend сохраняет как side effect.
    max_iter, tol, relaxation : критерии цикла; relaxation∈(0,1].

    Итерация k: B^{k+1}=step(ν^k) → критерий ‖B^{k+1}−B^k‖/‖B^k‖<tol → иначе
    ν^{k+1}=(1−ω)ν^k+ω·nu_of_B(B^{k+1}). На сошедшейся итерации выходим ДО обновления
    ν, поэтому `nu_cells` результата = ν финальной (сошедшейся) сборки.
    """
    if not callable(nu_of_B):
        raise ValueError("nu_of_B must be callable.")
    if not callable(step):
        raise ValueError("step must be callable.")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1].")

    nu_cells = np.asarray(nu_init, dtype=float).copy()
    if nu_cells.ndim != 1:
        raise ValueError("nu_init must have shape (n_cells,).")
    n_cells = nu_cells.shape[0]

    B_prev: np.ndarray | None = None
    B_cells = np.zeros((n_cells, 0), dtype=float)
    history: list[float] = []
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        B_cells = np.asarray(step(nu_cells), dtype=float)

        if B_prev is not None:
            denom = float(np.linalg.norm(B_prev))
            rel = float(np.linalg.norm(B_cells - B_prev)) / max(denom, 1.0e-30)
            history.append(rel)
            if rel < tol:
                converged = True
                break
        B_prev = B_cells

        nu_new = np.asarray(nu_of_B(B_cells), dtype=float)
        if nu_new.shape != (n_cells,):
            raise ValueError("nu_of_B must return an array of shape (n_cells,).")
        nu_cells = (1.0 - relaxation) * nu_cells + relaxation * nu_new

    return PicardLoopResult(
        B_cells=B_cells,
        nu_cells=nu_cells,
        n_iterations=it,
        converged=converged,
        rel_change_history=tuple(history),
    )
