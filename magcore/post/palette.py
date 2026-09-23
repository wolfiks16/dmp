from __future__ import annotations

import numpy as np

# ПАЛИТРА ПОЛЯ в проекте одна — радуга (решение Sergey 2026-09-23). Та же формула, что `rainbow` в
# webapp/static/index.html (ею же красит вид 3D в ws3d.js), чтобы рисунок и экран красили поле одинаково:
#     R = 1,5 − |4t − 3|,   G = 1,5 − |4t − 2|,   B = 1,5 − |4t − 1|,   каждое обрезано в [0, 1].
# У встроенного `jet` в matplotlib точки излома другие (0,11 / 0,34 / 0,65 …), поэтому он не годится.
# Отдельные шкалы — не поле: знакопеременные составляющие (синий–белый–красный), карта риска,
# температура; их эта палитра не касается. Здесь только числа (без matplotlib): ядро рисунков не строит.


def field_rainbow(t) -> np.ndarray:
    """Цвет поля по доле шкалы t ∈ [0, 1] (вне отрезка — обрезается): массив (..., 3) RGB в [0, 1]."""
    x = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)[..., None]
    return np.clip(1.5 - np.abs(4.0 * x - np.array([3.0, 2.0, 1.0])), 0.0, 1.0)


def field_rainbow_cmap(n: int = 256):
    """Палитра поля для matplotlib (ListedColormap из `field_rainbow`); matplotlib нужен только здесь."""
    from matplotlib.colors import ListedColormap

    return ListedColormap(field_rainbow(np.linspace(0.0, 1.0, int(n))), name="field_rainbow")
