"""
Визуализация полей на треугольной сетке (matplotlib, backend Agg → сохранение в PNG).

Рендерит результаты верифицированного 2D-ядра для РУЧНОЙ проверки: поле B, карту
риска размагничивания, температурное поле, сравнение материалов. Ничего в `magcore`
не меняет — только читает результаты решателей.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: только сохранение в файл, без окон
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402


def _triang(mesh) -> Triangulation:
    v = np.asarray(mesh.vertices, dtype=float)
    return Triangulation(v[:, 0], v[:, 1], np.asarray(mesh.cells, dtype=int))


def _centroids(mesh) -> np.ndarray:
    return mesh.vertices[mesh.cells].mean(axis=1)


def _save(fig, save_path: str | Path | None):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    return save_path


def plot_cell_scalar(mesh, values, *, title="", label="", cmap="viridis", save_path=None):
    """Поячеечное скалярное поле (напр. |B|) — заливка tripcolor (shading='flat')."""
    fig, ax = plt.subplots(figsize=(6, 5))
    tpc = ax.tripcolor(_triang(mesh), facecolors=np.asarray(values, dtype=float),
                       cmap=cmap, shading="flat")
    fig.colorbar(tpc, ax=ax, label=label)
    ax.set_aspect("equal"); ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    return _save(fig, save_path)


def plot_node_scalar(mesh, values, *, title="", label="", cmap="inferno", save_path=None):
    """Узловое скалярное поле (напр. температура T или потенциал A_z) — tricontourf."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.tricontourf(_triang(mesh), np.asarray(values, dtype=float), levels=20, cmap=cmap)
    fig.colorbar(cf, ax=ax, label=label)
    ax.set_aspect("equal"); ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    return _save(fig, save_path)


def plot_B_field(mesh, B_cells, *, title="Поле B", save_path=None):
    """Модуль |B| (заливка) + направление (стрелки в центроидах)."""
    B = np.asarray(B_cells, dtype=float)
    mag = np.linalg.norm(B, axis=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    tpc = ax.tripcolor(_triang(mesh), facecolors=mag, cmap="viridis", shading="flat")
    fig.colorbar(tpc, ax=ax, label="|B|, Тл")
    c = _centroids(mesh)
    ax.quiver(c[:, 0], c[:, 1], B[:, 0], B[:, 1], color="white", alpha=0.7,
              scale_units="xy", angles="xy")
    ax.set_aspect("equal"); ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    return _save(fig, save_path)


def plot_demag_risk(mesh, magnet_mask, risk, *, title=None, save_path=None):
    """
    Карта риска размагничивания (ГЛАВНЫЙ визуал новизны): ячейки магнита окрашены по
    марже к колену (диверг. шкала, центр 0: >0 безопасно, <0 за коленом). Немагнитные
    ячейки — серым. `risk` = DemagRiskMap; margin по cell_indices.
    """
    mask = np.asarray(magnet_mask, dtype=bool)
    n_cells = mesh.n_cells
    facec = np.full(n_cells, np.nan)
    facec[risk.cell_indices] = risk.margin

    fig, ax = plt.subplots(figsize=(6, 5))
    # серый фон (немагнитные)
    ax.tripcolor(_triang(mesh), facecolors=np.where(mask, np.nan, 0.0),
                 cmap="Greys", shading="flat", alpha=0.25)
    vmax = float(np.nanmax(np.abs(facec))) if np.isfinite(facec).any() else 1.0
    tpc = ax.tripcolor(_triang(mesh), facecolors=facec, cmap="RdBu",
                       shading="flat", vmin=-vmax, vmax=vmax)
    fig.colorbar(tpc, ax=ax, label="маржа к колену H_par−H_knee, А/м  (<0 = размагничен)")
    if title is None:
        title = ("Карта риска демага: T=%.0f°C, за коленом %d/%d ячеек"
                 % (risk.T, risk.n_demagnetized, risk.cell_indices.size))
    ax.set_aspect("equal"); ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    return _save(fig, save_path)


def plot_magnet_cell_field(mesh, cell_indices, values, *, title="", label="",
                           cmap="plasma", save_path=None):
    """
    Поле по ячейкам МАГНИТА (напр. рабочая точка H_op или коэфф. проницаемости P_c по всему
    объёму): окрашивает только ячейки магнита `cell_indices` значениями `values`, остальное —
    бледным фоном. Прямая визуализация рабочей точки по объёму магнита.
    """
    facec = np.full(mesh.n_cells, np.nan)
    facec[np.asarray(cell_indices, dtype=int)] = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tripcolor(_triang(mesh), facecolors=np.ones(mesh.n_cells),
                 cmap="Greys", vmin=0, vmax=6, shading="flat")
    tpc = ax.tripcolor(_triang(mesh), facecolors=facec, cmap=cmap, shading="flat")
    fig.colorbar(tpc, ax=ax, label=label)
    ax.set_aspect("equal"); ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    return _save(fig, save_path)


def plot_material_comparison(magnets_named, H_op, *, T_range=(20, 300), save_path=None):
    """
    Сравнение материалов: маржа к колену vs температура при фиксированном демаг-поле H_op.
    Показывает, при какой T каждый магнит уходит за колено (маржа<0). `magnets_named` =
    [(name, magnet), ...]. Прямая визуализация T-стойкости (NdFeB vs SmCo).
    """
    Ts = np.linspace(T_range[0], T_range[1], 200)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, mag in magnets_named:
        margin = np.array([float(mag.risk_margin(H_op, T)) for T in Ts])
        line, = ax.plot(Ts, margin / 1e3, label=name, lw=2)
        below = np.where(margin < 0.0)[0]
        if below.size:
            T_onset = Ts[below[0]]
            ax.axvline(T_onset, color=line.get_color(), ls="--", alpha=0.6)
            ax.annotate("%s: %.0f°C" % (name, T_onset), (T_onset, 0),
                        textcoords="offset points", xytext=(4, 10),
                        color=line.get_color())
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xlabel("Температура магнита, °C")
    ax.set_ylabel("Маржа к колену, кА/м  (>0 безопасно)")
    ax.set_title("T-стойкость к размагничиванию при H_op=%.2e А/м" % H_op)
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, save_path)
