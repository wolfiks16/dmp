from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import (
    AdaptiveIntegrationConfig,
    assemble_single_layer_p0p0_full,
)
from magcore.bem.spaces import FaceP0Space, VertexP1Space
from magcore.mesh.surface_mesh import SurfaceMesh


def surface_curl_hats(tri: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Поверхностные вихри (surface curls) трёх P1-шапочек на плоской грани.

    Для скалярной P1-функции на плоском треугольнике `curl_Γ λ = n × ∇_Γ λ` —
    ПОСТОЯННЫЙ касательный вектор. Возвращает массив (3, 3): строка `i` — это
    `c_i = n × ∇λ_i` для локальной вершины `i` (в порядке вершин грани).

    `∇λ_i` берётся внутренне (in-plane Gram-система), `n` — переданная внешняя
    нормаль грани (как выдаёт CouplingInterface). Знак `c_i` зависит от нормали,
    поэтому ВАЖНО передавать СОГЛАСОВАННЫЕ внешние нормали по всей Γ: в сборке
    W фигурируют перекрёстные члены `c_T·c_{T'}` между разными гранями, и
    глобально несогласованная ориентация их испортила бы (глобальный разворот
    всех нормалей сокращается, локальный — нет).

    Тождество (для проверки знаков): `n × ∇λ_i = (p_{i+1} − p_{i+2}) / (2A)`,
    если `n` совпадает с нормалью правила правой руки порядка (p0,p1,p2).
    """
    tri = np.asarray(tri, dtype=float)
    n = np.asarray(normal, dtype=float)
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    g = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]], dtype=float)
    # P[0]=∇λ_1, P[1]=∇λ_2 (касательные градиенты барицентрик s,t); ∇λ_0=−(∇λ_1+∇λ_2).
    p = np.linalg.solve(g, np.array([e1, e2], dtype=float))  # (2, 3)
    grad = np.empty((3, 3), dtype=float)
    grad[0] = -(p[0] + p[1])
    grad[1] = p[0]
    grad[2] = p[1]
    return np.cross(n, grad)  # (3, 3): строка i = n × ∇λ_i


def assemble_hypersingular_maue(
    surface_mesh: SurfaceMesh,
    outward_normals: np.ndarray,
    phi_space: VertexP1Space,
    flux_space: FaceP0Space,
    config: AdaptiveIntegrationConfig | None = None,
    s_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """
    Гиперсингулярный оператор W (P1×P1 по вершинам Γ) через тождество Мауэ,
    фаза B (docs/math/coupling.md §4, §6).

    Тождество Мауэ для Лапласа (regularisation гиперсингулярного оператора через
    однослойное ядро на поверхностных вихрях):

        ⟨W ψ, φ⟩_Γ = ∮_Γ ∮_Γ G(x,y) · curl_Γψ(y) · curl_Γφ(x) ds_y ds_x,
        G(x,y) = 1/(4π|x−y|).

    Для P1 `ψ=Σ_n ψ_n λ_n` на плоских гранях `curl_Γ λ_n = c_T^n` постоянен на
    каждой грани T (ненулевой только для 3 вершин грани). Поэтому

        W[m, n] = Σ_T Σ_{T'} (c_T^m · c_{T'}^n) · S[T, T'],
        S[T, T'] = ∮_T ∮_{T'} G(x,y) ds_y ds_x   (P0×P0 однослойная матрица).

    Реализация: `W = Rᵀ S R`, где S — уже собранный и верифицированный
    однослойный оператор (`assemble_single_layer_p0p0_full`), R — отображение
    «вершинные P1-степени свободы → постоянные поверхностные вихри по граням».
    Новых сингулярных интегралов НЕ вводит ⇒ автоматически наследует точную
    сингулярную квадратуру однослойного (Заутер–Шваб, фоновая оптимизация).

    Структурные свойства (проверяются тестами Кальдерона/Costabel):
      • симметрия `W = Wᵀ` (S симметрична);
      • `W · 1 = 0` точно (машинно): `Σ_{n∈T'} c_{T'}^n = n×∇(Σλ) = 0` ⇒ константы
        в ядре — дискретное тождество совместности для гиперсингулярного оператора;
      • положительная полуопределённость `W ⪰ 0` с ядром = константы (S SPD ⇒ RᵀSR ⪰ 0).

    Возвращает плотную (phi_space.ndofs × phi_space.ndofs).
    """
    normals = np.asarray(outward_normals, dtype=float)
    if normals.shape != (surface_mesh.n_faces, 3):
        raise ValueError("outward_normals must have shape (n_faces, 3).")

    # S — однослойный P0×P0 (= V_BB). Можно передать готовый, чтобы не пересобирать
    # (S_ext уже собирает V) — экономит самую дорогую часть.
    if s_matrix is None:
        s_matrix, _ = assemble_single_layer_p0p0_full(surface_mesh, flux_space.dof_to_face, config)
    else:
        s_matrix = np.asarray(s_matrix, dtype=float)

    face_indices = flux_space.dof_to_face  # отсортированный кортеж граней в порядке dof
    nf = flux_space.ndofs
    nv = phi_space.ndofs

    curls = np.zeros((nf, 3, 3), dtype=float)
    vdofs = np.zeros((nf, 3), dtype=int)
    for fi, face in enumerate(face_indices):
        tri = surface_mesh.face_vertices(face)
        curls[fi] = surface_curl_hats(tri, normals[face])
        vdofs[fi] = [phi_space.vertex_to_dof[int(v)] for v in surface_mesh.faces[face]]

    # W = Σ_{d=0}^{2} G_dᵀ S G_d — векторизованная форма того же контракта `RᵀSR`:
    # G_d[f, m] = d-я компонента поверхностного вихря шапочки вершины m на грани f
    # (0, если m∉f). Скалярное произведение c_f^a·c_g^b = Σ_d C[f,a,d]·C[g,b,d]
    # разделяет вклад граней ⇒ 3 произведения (BLAS) вместо O(nf²) python-цикла.
    fidx = np.repeat(np.arange(nf), 3)
    vidx = vdofs.reshape(-1)
    cvals = curls.reshape(nf * 3, 3)
    g = np.zeros((nf, nv, 3), dtype=float)
    g[fidx, vidx, :] = cvals  # вершины грани различны ⇒ коллизий присваивания нет

    w = np.zeros((nv, nv), dtype=float)
    for d in range(3):
        gd = g[:, :, d]
        w += gd.T @ (s_matrix @ gd)
    return w
