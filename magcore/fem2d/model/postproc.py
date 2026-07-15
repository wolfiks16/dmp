from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.fem2d.model.problem import Solution2D

# ОБЩИЙ пост-процессинг на Problem2D — величины для ЛЮБОЙ 2D-задачи, без привязки к «ротору»:
# энергия поля, поток (через разность A_z), момент (Арккио по кольцу), рабочая точка магнита
# по объёму. Машинные величины (момент/ЭДС/потери) — частный случай этих на сетке машины.


def cell_areas(solution: Solution2D) -> np.ndarray:
    mesh = solution.problem.mesh
    return np.array([mesh.cell_area(c) for c in range(mesh.n_cells)], dtype=float)


def magnetic_energy(solution: Solution2D, *, axial_length: float = 1.0) -> float:
    """
    Энергия магнитного поля [Дж]: W = L·∫ ½·ν·|B|²/μ₀ dA (полевая форма ½∫H·B; ν — относит.).
    Положительна; ∝ осевой длине; для линейной задачи ∝ |B|² (∝ квадрату тока).
    """
    B = solution.field.B_cells
    nu = solution.field.nu_cells
    w = 0.5 * nu * (B[:, 0] ** 2 + B[:, 1] ** 2) / MU0
    return float((w * cell_areas(solution)).sum() * float(axial_length))


def _bary(p, v):
    d = (v[1, 1] - v[2, 1]) * (v[0, 0] - v[2, 0]) + (v[2, 0] - v[1, 0]) * (v[0, 1] - v[2, 1])
    if abs(d) < 1e-18:
        return None
    l0 = ((v[1, 1] - v[2, 1]) * (p[0] - v[2, 0]) + (v[2, 0] - v[1, 0]) * (p[1] - v[2, 1])) / d
    l1 = ((v[2, 1] - v[0, 1]) * (p[0] - v[2, 0]) + (v[0, 0] - v[2, 0]) * (p[1] - v[2, 1])) / d
    return (l0, l1, 1.0 - l0 - l1)


def interpolate_Az(solution: Solution2D, point) -> float:
    """Значение векторного потенциала A_z в произвольной точке (P1-интерполяция по ячейке)."""
    mesh = solution.problem.mesh
    a = solution.field.a
    p = np.asarray(point, dtype=float)
    for c in range(mesh.n_cells):
        verts = mesh.cell_vertices(c)
        lam = _bary(p, verts)
        if lam is None:
            continue
        if min(lam) >= -1e-9:
            idx = mesh.cell_vertex_indices(c)
            return float(lam[0] * a[idx[0]] + lam[1] * a[idx[1]] + lam[2] * a[idx[2]])
    raise ValueError("точка вне сетки.")


def flux_between_points(solution: Solution2D, p1, p2, *, axial_length: float = 1.0) -> float:
    """
    Магнитный поток [Вб] через линию между точками p1→p2 (2D): Φ = (A_z(p2) − A_z(p1))·L.
    Антисимметричен по перестановке точек; ∝ осевой длине.
    """
    return float((interpolate_Az(solution, p2) - interpolate_Az(solution, p1)) * float(axial_length))


def torque_arkkio(
    solution: Solution2D,
    r_inner: float,
    r_outer: float,
    *,
    center=(0.0, 0.0),
    axial_length: float = 1.0,
) -> float:
    """
    Момент вокруг оси (0,0)/center по методу Арккио [Н·м]: усреднённый тензор Максвелла по
    кольцу r∈[r_inner,r_outer] (обычно зазор): T = L/(μ₀(r_o−r_i))·Σ r·B_r·B_θ·area. Общий:
    работает для любой Problem2D с заданным кольцом (не требует понятия «машина»).
    """
    mesh = solution.problem.mesh
    B = solution.field.B_cells
    cx, cy = float(center[0]), float(center[1])
    cen = np.array([mesh.cell_centroid(c) for c in range(mesh.n_cells)]) - np.array([cx, cy])
    r = np.hypot(cen[:, 0], cen[:, 1])
    band = np.where((r >= r_inner) & (r <= r_outer))[0]
    if band.size == 0:
        return 0.0
    rc = r[band]
    rhat = cen[band] / rc[:, None]
    that = np.stack([-cen[band, 1], cen[band, 0]], axis=1) / rc[:, None]
    Br = B[band, 0] * rhat[:, 0] + B[band, 1] * rhat[:, 1]
    Bth = B[band, 0] * that[:, 0] + B[band, 1] * that[:, 1]
    areas = cell_areas(solution)[band]
    integral = float(np.sum(rc * Br * Bth * areas))
    return float(axial_length) / (MU0 * (r_outer - r_inner)) * integral


@dataclass(frozen=True, slots=True)
class OperatingPointField:
    """Рабочая точка магнита по объёму (общая, из Problem2D-решения)."""
    cell_indices: np.ndarray
    H_op: np.ndarray            # (n_mag,) А/м
    B_op: np.ndarray            # (n_mag,) Тл
    permeance: np.ndarray       # (n_mag,) P_c
    cell_volume: np.ndarray     # (n_mag,) м³
    knee_field: float           # А/м
    T: float

    @property
    def total_volume(self) -> float:
        return float(self.cell_volume.sum())

    def volume_weighted_mean_B_op(self) -> float:
        return float(np.average(self.B_op, weights=self.cell_volume))

    def volume_weighted_mean_H_op(self) -> float:
        return float(np.average(self.H_op, weights=self.cell_volume))

    def worst_H_op(self) -> float:
        return float(self.H_op.min())

    def volume_fraction_below(self, H: float) -> float:
        return float(self.cell_volume[self.H_op < float(H)].sum() / self.cell_volume.sum())


def operating_point(solution: Solution2D, *, axial_length: float = 1.0) -> OperatingPointField:
    """Рабочая точка (H_op, B_op, P_c) по объёму магнита из общего решения (нужен магнит в задаче)."""
    problem = solution.problem
    magnet = problem.magnet()
    if magnet is None:
        raise ValueError("в задаче нет магнита — рабочая точка не определена.")
    idx = np.where(problem.magnet_mask())[0]
    axes = np.asarray(problem.magnet_axis)[idx]
    H = solution.field.H_cells[idx]
    B = solution.field.B_cells[idx]
    H_op = np.einsum("ij,ij->i", H, axes) / MU0
    B_op = np.einsum("ij,ij->i", B, axes)
    denom = MU0 * np.abs(H_op)
    permeance = np.where(denom > 0.0, B_op / denom, np.inf)
    areas = cell_areas(solution)[idx]
    return OperatingPointField(
        cell_indices=idx, H_op=H_op, B_op=B_op, permeance=permeance,
        cell_volume=areas * float(axial_length),
        knee_field=float(magnet.knee_field(problem.T)), T=float(problem.T),
    )
