from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.fem2d.model.materials import MagnetMaterial, SteelMaterial
from magcore.fem3d.fieldlines import _seed_cells
from magcore.fem3d.scalar import ScalarField3D

# ИЗМЕРЕНИЯ ПО РЕШЕНИЮ 3D (пункт 6 плана интерфейса): точка, линия, окружность, среднее по телу (рабочая
# точка магнита), насыщение стали. Значения — из поля В ЯЧЕЙКАХ: решение кусочно-постоянно по ячейке, и число в
# точке — значение ячейки, в которой точка лежит; так же считает 2D (Л-103: числа — по ячейкам, линии для
# рисунка — по восстановленному полю). Точка вне сетки — NaN, а не ближайшая ячейка.


@dataclass(frozen=True)
class Samples3D:
    """Значения поля в точках."""

    points: np.ndarray   # (N, 3) [м]
    cells: np.ndarray    # (N,) ячейка точки; −1 — точка вне сетки
    B: np.ndarray        # (N, 3) [Тл]; NaN вне сетки
    H: np.ndarray        # (N, 3) [А/м]; NaN вне сетки
    body: list           # (N,) имя тела (региона) в точке; '' вне сетки


def sample(field: ScalarField3D, points) -> Samples3D:
    """Поле в точках `points` (N, 3) [м]: значение ячейки, в которой лежит точка."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("точки — по три конечных координаты.")
    problem = field.problem
    cells, _, hit = _seed_cells(problem.mesh, pts, k=64)
    cells = np.where(hit, cells, -1)
    B = np.full((pts.shape[0], 3), np.nan)
    H = np.full((pts.shape[0], 3), np.nan)
    B[hit] = field.B_cells[cells[hit]]
    H[hit] = field.H_cells[cells[hit]]
    reg = np.asarray(problem.cell_region)
    names = {rid: r.name for rid, r in problem.regions.items()}
    body = [names[int(reg[c])] if c >= 0 else "" for c in cells]
    return Samples3D(points=pts, cells=cells, B=B, H=H, body=body)


def line_points(p1, p2, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n точек на отрезке p1→p2 (концы включены) и расстояние каждой от p1 [м]."""
    a, b = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or int(n) < 2:
        raise ValueError("отрезок — две точки по три координаты, точек не меньше двух.")
    t = np.linspace(0.0, 1.0, int(n))
    return a[None, :] + t[:, None] * (b - a)[None, :], t * float(np.linalg.norm(b - a))


def circle_basis(normal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Единичная нормаль n и базис плоскости окружности (e1, e2), e1 × e2 = n; угол θ отсчитывается от e1 к e2.
    Для осей координат — как в 2D: нормаль Z — θ от +X к +Y; нормаль X — от +Y к +Z; нормаль Y — от +Z к +X.
    """
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.linalg.norm(n) > 0.0:
        raise ValueError("нормаль окружности — ненулевой вектор из трёх чисел.")
    n = n / np.linalg.norm(n)
    k = int(np.argmax(np.abs(n)))
    if abs(abs(n[k]) - 1.0) < 1e-12:                    # вдоль оси координат — базис по круговой перестановке осей
        s = np.sign(n[k])
        e1 = np.eye(3)[(k + 1) % 3]
        e2 = s * np.eye(3)[(k + 2) % 3]
        return n, e1, e2
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - n * (a @ n)
    e1 /= np.linalg.norm(e1)
    return n, e1, np.cross(n, e1)


@dataclass(frozen=True)
class CircleSamples3D:
    """Поле на окружности: составляющие в её плоскости — радиальная и касательная, и по нормали."""

    theta: np.ndarray    # (N,) угол [рад], от e1 к e2
    samples: Samples3D
    Br: np.ndarray       # (N,) радиальная B [Тл]
    Bt: np.ndarray       # (N,) касательная B [Тл], положительна по направлению роста θ
    Bn: np.ndarray       # (N,) B по нормали окружности [Тл]


def circle(field: ScalarField3D, center, normal, radius: float, n: int) -> CircleSamples3D:
    """Поле в n точках окружности (центр [м], нормаль, радиус [м]); точки равномерно, первая — на e1."""
    if not float(radius) > 0.0 or int(n) < 3:
        raise ValueError("радиус окружности — положительный, точек не меньше трёх.")
    nn, e1, e2 = circle_basis(normal)
    th = 2.0 * np.pi * np.arange(int(n)) / int(n)
    er = np.cos(th)[:, None] * e1[None, :] + np.sin(th)[:, None] * e2[None, :]
    et = -np.sin(th)[:, None] * e1[None, :] + np.cos(th)[:, None] * e2[None, :]
    pts = np.asarray(center, dtype=float)[None, :] + float(radius) * er
    s = sample(field, pts)
    return CircleSamples3D(theta=th, samples=s, Br=np.einsum("ij,ij->i", s.B, er),
                           Bt=np.einsum("ij,ij->i", s.B, et), Bn=s.B @ nn)


@dataclass(frozen=True)
class BodyMean3D:
    """Средние по объёму тела; у магнита — рабочая точка вдоль оси намагничивания ячейки."""

    name: str
    volume: float            # [м³]
    B_mean: np.ndarray       # (3,) ⟨B⟩ [Тл]
    B_abs_mean: float        # ⟨|B|⟩ [Тл]
    H_mean: np.ndarray       # (3,) ⟨H⟩ [А/м]
    magnet: bool
    B_par: float | None      # ⟨B·e⟩ [Тл] — рабочая точка магнита, B_d
    H_par: float | None      # ⟨H·e⟩ [А/м] — H_d (в разомкнутой цепи < 0)
    permeance: float | None  # P_c = B_d / (μ₀·|H_d|)


def body_mean(field: ScalarField3D, name: str) -> BodyMean3D:
    """Средние по объёму тела `name` (с весом объёма ячеек); у магнита — B_d, H_d и P_c вдоль оси ячейки."""
    problem = field.problem
    rid = next((i for i, r in problem.regions.items() if r.name == name), None)
    if rid is None:
        raise ValueError(f"нет тела {name!r}.")
    sel = np.asarray(problem.cell_region) == rid
    if not sel.any():
        raise ValueError(f"у тела {name!r} нет ячеек.")
    v = field.volumes[sel]
    V = float(v.sum())
    B, H = field.B_cells[sel], field.H_cells[sel]
    mean = lambda x: (x * v[:, None]).sum(axis=0) / V              # noqa: E731
    is_mag = isinstance(problem.regions[rid].material, MagnetMaterial)
    bp = hp = pc = None
    if is_mag:
        e = np.asarray(problem.magnet_axis, dtype=float)[sel]
        bp = float((np.einsum("ij,ij->i", B, e) * v).sum() / V)
        hp = float((np.einsum("ij,ij->i", H, e) * v).sum() / V)
        pc = bp / (MU0 * abs(hp)) if hp != 0.0 else float("inf")
    return BodyMean3D(name=name, volume=V, B_mean=mean(B), B_abs_mean=float((np.linalg.norm(B, axis=1) * v).sum() / V),
                      H_mean=mean(H), magnet=is_mag, B_par=bp, H_par=hp, permeance=pc)


@dataclass(frozen=True)
class Saturation3D:
    """Насыщение одного стального тела."""

    name: str
    volume: float            # [м³]
    B_max: float             # наибольшая |B| в ячейке [Тл] — одна ячейка, у острых углов зависит от сетки
    fraction_above: float    # доля объёма с |B| выше порога
    threshold: float         # порог [Тл]


def steel_saturation(field: ScalarField3D, threshold: float = 1.6) -> list[Saturation3D]:
    """По каждому стальному телу: наибольшая |B| и доля объёма, где |B| выше порога `threshold` [Тл]."""
    if not float(threshold) >= 0.0:
        raise ValueError("порог индукции — неотрицательное число, Тл.")
    problem = field.problem
    reg = np.asarray(problem.cell_region)
    Bm = np.linalg.norm(field.B_cells, axis=1)
    out = []
    for rid, r in sorted(problem.regions.items()):
        if not isinstance(r.material, SteelMaterial):
            continue
        sel = reg == rid
        if not sel.any():
            continue
        v, b = field.volumes[sel], Bm[sel]
        out.append(Saturation3D(name=r.name, volume=float(v.sum()), B_max=float(b.max()),
                                fraction_above=float(v[b > float(threshold)].sum() / v.sum()), threshold=float(threshold)))
    return out
