from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from magcore.constants import MU0
from magcore.fem2d.model.materials import Air, LinearMaterial, MagnetMaterial, SteelMaterial
from magcore.fem3d.mesh import TetMesh3D
from magcore.fem3d.problem import Problem3D

# ЛИНЕЙНАЯ 3D-МАГНИТОСТАТИКА НА ПОЛНОМ СКАЛЯРНОМ ПОТЕНЦИАЛЕ (этап 3D-2, план —
# docs/plan_3d_2026-09-11.md).
#
# Без токов ∇×H = 0 ⇒ H = −∇φ. Материалы: воздух B = μ₀H; линейный B = μ₀μ_r H; магнит
# B = μ₀ μ̂ H + B_r e, где μ̂ = μ_∥ e eᵀ + μ_⊥ (I − e eᵀ) — ТЕНЗОР: вдоль оси recoil μ_rec,
# поперёк μ_⊥ (скалярная проницаемость магнита однажды уже дала ложный каскад в 2D, Л-20).
# ∇·B = 0, делённое на μ₀:  ∇·(μ̂ ∇φ) = ∇·M,  M = B_r e / μ₀ [А/м]. Слабая форма:
#     ∫ μ̂ ∇φ·∇v dΩ = ∫ M·∇v dΩ − ∮ v (B·n/μ₀) dS.
# Узловые элементы первого порядка: ∇φ и M постоянны в ячейке ⇒ интегралы точные.
#
# ВНЕШНЯЯ ГРАНИЦА (область обрезана параллелепипедом):
#   'neumann'   — B·n = μ₀ H₀·n (без внешнего поля — поток не выходит: «сверхпроводящая стенка»);
#                 φ определён с точностью до константы и закрепляется в одном узле;
#   'dirichlet' — φ = −H₀·x (без внешнего поля φ = 0: силовые линии перпендикулярны границе,
#                 «стенка бесконечной проницаемости»).
# Первая выталкивает поток, и магнит размагничен сильнее, чем в свободном пространстве; вторая
# притягивает, и слабее. Истина между ними; разрыв убывает как (размер тела / расстояние до
# границы)³ — дипольное поле. Решение с обеими границами даёт ошибку обрезки без эталона.


@dataclass(frozen=True, slots=True)
class ScalarField3D:
    """Решение: потенциал в узлах [А], H и B по ячейкам, материалы по ячейкам, граница."""

    problem: Problem3D
    phi: np.ndarray            # (n_vertices,) [А]
    H_cells: np.ndarray        # (n_cells, 3) [А/м]
    B_cells: np.ndarray        # (n_cells, 3) [Тл]
    mu_cells: np.ndarray       # (n_cells, 3, 3) относительная проницаемость
    M_cells: np.ndarray        # (n_cells, 3) намагниченность B_r e / μ₀ [А/м]
    volumes: np.ndarray        # (n_cells,) [м³]
    bc: str
    applied_field: np.ndarray  # (3,) H₀ [А/м]
    residual: float            # относительная невязка (линейной системы / нелинейной задачи)
    n_iterations: int = 1      # итерации Ньютона (у линейной задачи — одна)
    converged: bool = True
    residual_history: tuple = ()
    risk: object = None        # DemagRiskMap — карта риска размагничивания (нелинейный решатель)
    retention: np.ndarray | None = None    # (n_cells,) сохранённая доля ремнантности r ∈ [0, 1] после этого нагружения — история (вне магнита 1; Л-100)
    coenergy_density: np.ndarray | None = None  # (n_cells,) w' = ∫₀^H B·dH того закона, по которому решено [Дж/м³]

    def average(self, values: np.ndarray, mask) -> np.ndarray:
        """Среднее по объёму поячеечной величины в ячейках `mask`."""
        m = np.asarray(mask)
        w = self.volumes[m]
        return (np.asarray(values)[m] * w[:, None]).sum(axis=0) / w.sum()


def p1_gradients(mesh: TetMesh3D) -> tuple[np.ndarray, np.ndarray]:
    """Градиенты барицентрических функций ячеек (M,4,3) и объёмы (M,)."""
    v = mesh.vertices[mesh.cells]
    J = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], axis=2)  # столбцы — рёбра
    inv = np.linalg.inv(J)                                   # строки — ∇λ₁, ∇λ₂, ∇λ₃
    grads = np.empty((mesh.n_cells, 4, 3))
    grads[:, 1:, :] = inv
    grads[:, 0, :] = -inv.sum(axis=1)
    return grads, np.linalg.det(J) / 6.0


def material_fields(problem: Problem3D) -> tuple[np.ndarray, np.ndarray]:
    """Поячеечный тензор относительной проницаемости (M,3,3) и намагниченность B_r e/μ₀ (M,3)."""
    nc = problem.mesh.n_cells
    reg = np.asarray(problem.cell_region)
    mu = np.tile(np.eye(3), (nc, 1, 1))
    M = np.zeros((nc, 3))
    for rid, region in problem.regions.items():
        cells = np.where(reg == rid)[0]
        if cells.size == 0:
            continue
        mat = region.material
        if isinstance(mat, Air):
            continue
        if isinstance(mat, LinearMaterial):
            mu[cells] = float(mat.mu_r) * np.eye(3)
        elif isinstance(mat, MagnetMaterial):
            e = np.asarray(problem.magnet_axis, dtype=float)[cells]
            ee = e[:, :, None] * e[:, None, :]
            mag = mat.magnet
            mu[cells] = float(mag.mu_perp) * (np.eye(3) - ee) + float(mag.mu_rec) * ee
            M[cells] = (float(mag.Br(problem.T)) / MU0) * e
        elif isinstance(mat, SteelMaterial):
            raise NotImplementedError("нелинейная сталь в 3D — этап 3D-3.")
        else:
            raise TypeError(f"неизвестный материал региона {rid}: {type(mat)}")
    return mu, M


def assemble_scalar_system(mesh: TetMesh3D, mu: np.ndarray, M: np.ndarray,
                           grads: np.ndarray | None = None,
                           vol: np.ndarray | None = None) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Матрица ∫ μ̂ ∇φ·∇v (разреженная, симметричная) и правая часть ∫ M·∇v по узлам.
    В ячейке: K_c = V G μ̂ Gᵀ, f_c = V G M (G — градиенты узловых функций, 4×3).
    """
    if grads is None or vol is None:
        grads, vol = p1_gradients(mesh)
    n = mesh.n_vertices
    Gm = np.einsum("cik,ckl->cil", grads, mu)
    Kc = vol[:, None, None] * np.einsum("cil,cjl->cij", Gm, grads)
    rows = np.repeat(mesh.cells, 4, axis=1).reshape(-1)
    cols = np.tile(mesh.cells, (1, 4)).reshape(-1)
    K = sp.coo_matrix((Kc.reshape(-1), (rows, cols)), shape=(n, n)).tocsr()
    fc = vol[:, None] * np.einsum("cik,ck->ci", grads, M)
    f = np.bincount(mesh.cells.reshape(-1), weights=fc.reshape(-1), minlength=n)
    return K, f


def solve_linear3d(problem: Problem3D, *, bc: str = "neumann", applied_field=None,
                   solver: str = "direct", rtol: float = 1.0e-10) -> ScalarField3D:
    """
    Решить линейную 3D-задачу на полном скалярном потенциале (магниты, воздух, линейные
    материалы; нелинейная сталь — этап 3D-3). `bc` — 'neumann' | 'dirichlet' (см. шапку);
    `applied_field` — однородное внешнее поле H₀ [А/м]; `solver` — 'direct' | 'cg'.
    """
    problem.check()
    if bc not in ("neumann", "dirichlet"):
        raise ValueError("bc должен быть 'neumann' или 'dirichlet'.")
    if solver not in ("direct", "cg"):
        raise ValueError("solver должен быть 'direct' или 'cg'.")
    H0 = np.zeros(3) if applied_field is None else np.asarray(applied_field, dtype=float)
    if H0.shape != (3,) or not np.isfinite(H0).all():
        raise ValueError("applied_field — три конечных числа [А/м].")

    mesh = problem.mesh
    n = mesh.n_vertices
    mu, M = material_fields(problem)
    grads, vol = p1_gradients(mesh)
    K, f = assemble_scalar_system(mesh, mu, M, grads, vol)
    faces, area_vec = mesh.boundary_faces_oriented()
    bnodes = np.unique(faces)
    phi = np.zeros(n)
    if bc == "dirichlet":
        fixed = bnodes
    else:
        flux = area_vec @ H0                                  # H₀·n·S по грани
        f = f - np.bincount(faces.reshape(-1), weights=np.repeat(flux / 3.0, 3), minlength=n)
        fixed = bnodes[:1]                                    # константа потенциала
    phi[fixed] = -mesh.vertices[fixed] @ H0
    free = np.setdiff1d(np.arange(n), fixed)
    K_ff = K[free, :][:, free].tocsc()
    rhs = f[free] - K[free, :][:, fixed] @ phi[fixed]
    if solver == "direct":
        x = spla.spsolve(K_ff, rhs)
    else:
        precond = sp.diags(1.0 / K_ff.diagonal())
        x, info = spla.cg(K_ff, rhs, rtol=rtol, maxiter=20 * free.size, M=precond)
        if info != 0:
            raise RuntimeError(f"сопряжённые градиенты не сошлись (info={info}).")
    phi[free] = x
    residual = float(np.linalg.norm(K_ff @ x - rhs) / max(np.linalg.norm(rhs), 1e-300))

    H = -np.einsum("ci,cik->ck", phi[mesh.cells], grads)
    B = MU0 * (np.einsum("ckl,cl->ck", mu, H) + M)
    wco = 0.5 * MU0 * np.einsum("ck,ckl,cl->c", H, mu, H) + MU0 * np.einsum("ck,ck->c", M, H)
    return ScalarField3D(problem=problem, phi=phi, H_cells=H, B_cells=B, mu_cells=mu, M_cells=M,
                         volumes=vol, bc=bc, applied_field=H0, residual=residual, coenergy_density=wco)


# ----------------------------------------------------------------- значения в точках
def _barycentric(mesh: TetMesh3D, cand: np.ndarray, pts: np.ndarray, tol: float):
    v = mesh.vertices[mesh.cells[cand]]                                   # (N,k,4,3)
    J = np.stack([v[..., 1, :] - v[..., 0, :], v[..., 2, :] - v[..., 0, :],
                  v[..., 3, :] - v[..., 0, :]], axis=-1)                   # (N,k,3,3)
    l123 = np.linalg.solve(J, (pts[:, None, :] - v[..., 0, :])[..., None])[..., 0]
    lam = np.concatenate([1.0 - l123.sum(axis=-1, keepdims=True), l123], axis=-1)
    return lam, lam.min(axis=-1) >= -tol


def locate_points(mesh: TetMesh3D, points, *, k: int = 32,
                  tol: float = 1.0e-9) -> tuple[np.ndarray, np.ndarray]:
    """Ячейка, содержащая каждую точку (N,), и её барицентрические координаты (N,4)."""
    from scipy.spatial import cKDTree

    pts = np.atleast_2d(np.asarray(points, dtype=float))
    k = int(min(k, mesh.n_cells))
    _, cand = cKDTree(mesh.cell_centroids()).query(pts, k=k)
    cand = np.asarray(cand).reshape(pts.shape[0], k)
    lam, ok = _barycentric(mesh, cand, pts, tol)
    hit = ok.any(axis=1)
    first = ok.argmax(axis=1)
    rows = np.arange(pts.shape[0])
    cells = np.where(hit, cand[rows, first], -1)
    bary = lam[rows, first]
    all_cells = np.arange(mesh.n_cells)[None, :]
    for i in np.where(~hit)[0]:                                          # запасной перебор
        lam_i, ok_i = _barycentric(mesh, all_cells, pts[i:i + 1], tol)
        if not ok_i[0].any():
            raise ValueError(f"точка {pts[i]} вне сетки.")
        j = int(ok_i[0].argmax())
        cells[i], bary[i] = j, lam_i[0, j]
    return cells, bary


def evaluate_phi(field: ScalarField3D, points) -> np.ndarray:
    """Потенциал в точках — линейная интерполяция внутри ячейки (N,) [А]."""
    mesh = field.problem.mesh
    cells, bary = locate_points(mesh, points)
    return (bary * field.phi[mesh.cells[cells]]).sum(axis=1)
