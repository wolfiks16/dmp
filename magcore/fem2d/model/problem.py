from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.fem2d.mesh import TriangleMesh
from magcore.fem2d.model.materials import (
    Air,
    LinearMaterial,
    MagnetMaterial,
    SteelMaterial,
)
from magcore.fem2d.newton import solve_nonlinear_2d_newton
from magcore.fem2d.nonlinear import Fem2DPicardResult, solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.hybrid.magnet_demag import (
    DemagRiskMap,
    MagnetDemagPolicy,
    compute_demag_risk_map,
)

# ОБЩАЯ регион-объектная модель 2D-магнитостатической задачи. Не знает ни про «мотор», ни
# про «ротор»: произвольная сетка + помеченные регионы + материал на регион + ток по ячейкам
# + ГУ. Разбирается в поячеечную ν и источники и решается общим Picard-ядром. PMSM (и любая
# другая конструкция) — лишь ПОСТАВЩИК такой задачи (см. machines.bridge). Носитель принципа
# универсальности: физика/решатель не привязаны к одной модели.


@dataclass(frozen=True, slots=True)
class Region2D:
    """Помеченная область геометрии с назначенным материалом."""
    region_id: int
    name: str
    material: object   # Air | LinearMaterial | SteelMaterial | MagnetMaterial


@dataclass(frozen=True, slots=True)
class Problem2D:
    """
    Полная постановка общей 2D-задачи: сетка + поячеечные метки регионов + материалы регионов
    (+ ось магнита поячеечно, + ток поячеечно [физ. А/м²], + T, + ГУ). ГУ по умолчанию —
    A_z=0 на границе сетки.
    """
    mesh: TriangleMesh
    cell_region: np.ndarray                # (n_cells,) id региона на ячейку
    regions: dict                          # region_id -> Region2D
    magnet_axis: np.ndarray | None = None  # (n_cells,2) ось·полярность в магните, иначе 0
    j_cells: np.ndarray | None = None      # (n_cells,) физ. плотность тока А/м² (0 вне проводника)
    T: float = 20.0                        # температура [°C] (заданная; S1/S2)
    dirichlet_dofs: object = None          # None -> граница сетки
    dirichlet_values: object = 0.0

    def magnet_regions(self) -> list[Region2D]:
        return [r for r in self.regions.values() if isinstance(r.material, MagnetMaterial)]

    def magnet(self) -> AnisotropicBHTMagnet | None:
        mrs = self.magnet_regions()
        if not mrs:
            return None
        mags = {id(r.material.magnet): r.material.magnet for r in mrs}
        if len(mags) != 1:
            raise NotImplementedError("несколько марок магнита в одной задаче пока не поддержано.")
        return next(iter(mags.values()))

    def magnet_mask(self) -> np.ndarray:
        ids = {r.region_id for r in self.magnet_regions()}
        return np.isin(self.cell_region, list(ids)) if ids else np.zeros(self.mesh.n_cells, bool)

    def validate(self) -> list[str]:
        p: list[str] = []
        nc = self.mesh.n_cells
        reg = np.asarray(self.cell_region)
        if reg.shape != (nc,):
            p.append("cell_region должен иметь форму (n_cells,).")
            return p
        missing = set(np.unique(reg).tolist()) - set(self.regions.keys())
        if missing:
            p.append(f"нет материала для регионов: {sorted(missing)}.")
        if self.j_cells is not None and np.asarray(self.j_cells).shape != (nc,):
            p.append("j_cells должен иметь форму (n_cells,).")
        mrs = self.magnet_regions()
        if mrs:
            if self.magnet_axis is None or np.asarray(self.magnet_axis).shape != (nc, 2):
                p.append("для магнитных регионов нужна magnet_axis формы (n_cells,2).")
            try:
                self.magnet()
            except NotImplementedError as e:
                p.append(str(e))
        return p

    def check(self) -> None:
        issues = self.validate()
        if issues:
            raise ValueError("Некорректная постановка Problem2D:\n  - " + "\n  - ".join(issues))


def _reluctivity(problem: Problem2D):
    """Собрать поячеечный ν_of_B (относительный) + начальное ν из материалов регионов."""
    nc = problem.mesh.n_cells
    reg = np.asarray(problem.cell_region)
    nu_base = np.ones(nc, dtype=float)              # воздух по умолчанию
    steel_groups: list[tuple[np.ndarray, object]] = []
    for rid, region in problem.regions.items():
        cells = np.where(reg == rid)[0]
        if cells.size == 0:
            continue
        mat = region.material
        if isinstance(mat, Air):
            nu_base[cells] = 1.0
        elif isinstance(mat, LinearMaterial):
            nu_base[cells] = 1.0 / mat.mu_r
        elif isinstance(mat, MagnetMaterial):
            nu_base[cells] = 1.0 / mat.magnet.mu_rec
        elif isinstance(mat, SteelMaterial):
            steel_groups.append((cells, mat.curve))
        else:
            raise TypeError(f"неизвестный материал региона {rid}: {type(mat)}")

    def nu_of_B(B_cells: np.ndarray) -> np.ndarray:
        nu = nu_base.copy()
        for cells, curve in steel_groups:
            Bmag = np.hypot(B_cells[cells, 0], B_cells[cells, 1])
            nu[cells] = MU0 * np.array([curve.nu_chord(float(b)) for b in Bmag])
        return nu

    return nu_of_B, nu_of_B(np.zeros((nc, 2), dtype=float))


def _reluctivity_newton(problem: Problem2D):
    """
    Как `_reluctivity`, но для Ньютона: возвращает `nu_and_dnu(B) → (ν, dν/d|B|²)`. Для стали
    dν/d|B|² = (ν_d − ν_chord)/(2|B|²) (относит.; ν_d = μ₀·nu_differential); воздух/линейный/
    магнит — постоянная ν ⇒ dν=0. Это и есть касательный член метода Ньютона.
    """
    nc = problem.mesh.n_cells
    reg = np.asarray(problem.cell_region)
    nu_base = np.ones(nc, dtype=float)
    steel_groups: list[tuple[np.ndarray, object]] = []
    for rid, region in problem.regions.items():
        cells = np.where(reg == rid)[0]
        if cells.size == 0:
            continue
        mat = region.material
        if isinstance(mat, Air):
            nu_base[cells] = 1.0
        elif isinstance(mat, LinearMaterial):
            nu_base[cells] = 1.0 / mat.mu_r
        elif isinstance(mat, MagnetMaterial):
            nu_base[cells] = 1.0 / mat.magnet.mu_rec
        elif isinstance(mat, SteelMaterial):
            steel_groups.append((cells, mat.curve))
        else:
            raise TypeError(f"неизвестный материал региона {rid}: {type(mat)}")

    def nu_and_dnu(B_cells: np.ndarray):
        nu = nu_base.copy()
        dnu = np.zeros(nc, dtype=float)
        for cells, curve in steel_groups:
            Bmag = np.hypot(B_cells[cells, 0], B_cells[cells, 1])
            nu_c = MU0 * np.array([curve.nu_chord(float(b)) for b in Bmag])
            nu_d = MU0 * np.array([curve.nu_differential(float(b)) for b in Bmag])
            nu[cells] = nu_c
            b2 = np.maximum(Bmag ** 2, 1e-12)
            dnu[cells] = np.where(Bmag > 1e-6, (nu_d - nu_c) / (2.0 * b2), 0.0)
        return nu, dnu

    return nu_and_dnu, nu_and_dnu(np.zeros((nc, 2), dtype=float))[0]


@dataclass(frozen=True, slots=True)
class Solution2D:
    problem: Problem2D
    field: Fem2DPicardResult
    risk: DemagRiskMap | None

    @property
    def B_cells(self) -> np.ndarray:
        return self.field.B_cells

    @property
    def converged(self) -> bool:
        return self.field.converged


def solve_problem2d(
    problem: Problem2D,
    *,
    method: str = "newton",
    relaxation: float = 0.1,
    demag_relaxation: float = 0.25,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    track_worst_point: bool = False,
    demag: bool = True,
) -> Solution2D:
    """
    Решить общую 2D-задачу: материалы регионов → поячеечная ν + источники (магнит через
    MagnetDemagPolicy, ток = μ₀·j). `method='newton'` (по умолчанию) — метод Ньютона с
    касательной релуктивностью: квадратичная сходимость, число итераций НЕ зависит от сетки,
    без подбора релаксации (демаг гасится ФИКСИРОВАННОЙ `demag_relaxation`, не зависящей от
    сетки). `method='picard'` — хордовый Пикар с `relaxation` (совместимость/эталон).
    Чистая магнитостатика при заданной T (нагрев — динамический модуль S3).
    """
    problem.check()
    space = LagrangeP1Space2D(problem.mesh)
    nc = problem.mesh.n_cells
    j = None if problem.j_cells is None else MU0 * np.asarray(problem.j_cells, dtype=float)
    magnet = problem.magnet()
    mmask = problem.magnet_mask()

    def _policy(relax):
        if magnet is None or not demag:
            return None
        return MagnetDemagPolicy(magnet, mmask, T=problem.T, n_cells=nc,
                                 axis=problem.magnet_axis, relaxation=relax,
                                 track_worst_point=track_worst_point)

    if method == "newton":
        nu_and_dnu, nu_init = _reluctivity_newton(problem)
        em = solve_nonlinear_2d_newton(
            space, nu_and_dnu, nu_init=nu_init, j_cells=j, magnetization=_policy(demag_relaxation),
            dirichlet_dofs=problem.dirichlet_dofs, dirichlet_values=problem.dirichlet_values,
            max_iter=max_iter, tol=tol,
        )
    elif method == "picard":
        nu_of_B, nu_init = _reluctivity(problem)
        em = solve_nonlinear_2d_picard(
            space, nu_of_B=nu_of_B, nu_init=nu_init, j_cells=j, magnetization=_policy(relaxation),
            dirichlet_dofs=problem.dirichlet_dofs, dirichlet_values=problem.dirichlet_values,
            relaxation=relaxation, max_iter=max_iter, tol=tol,
        )
    else:
        raise ValueError("method должен быть 'newton' | 'picard'.")

    risk = None
    if magnet is not None:
        risk = compute_demag_risk_map(magnet, em, mmask, T=problem.T, axis=problem.magnet_axis)
    return Solution2D(problem=problem, field=em, risk=risk)
