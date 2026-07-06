from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from magcore.bem.element_integrals import triangle_area

"""
Полусейминалитическая сингулярная квадратура Заутера–Шваба для галёркинских
интегралов BEM на ПЛОСКИХ треугольных панелях (Sauter & Schwab, "Boundary
Element Methods", Springer 2011, гл. 5).

Регуляризующие преобразования координат отображают 4-куб (η1,η2,η3,ξ)∈[0,1]^4
на пару опорных симплексов T̂×T̂, а якобиан преобразования (степени ξ,η) гасит
сингулярность ядра 1/r (и 1/r³ для двойного слоя на смежных непланарных
панелях). Это даёт O(1) работу на пару и экспоненциальную сходимость по числу
гауссовых узлов — вместо наивного рекурсивного подразбиения.

Формулы преобразований сверены с эталонной реализацией
(krcools/SauterSchwabQuadrature.jl, src/pulled_back_integrands.jl) и проверены
тождеством меры: ∫_{[0,1]^4} J_SS dη dξ = area(T̂)² = 1/4 для КАЖДОГО случая
(CommonFace: 6·1/24; CommonEdge: 1/12+4/24; CommonVertex: 2·1/8).

Опорный симплекс: T̂ = {(u1,u2): u1≥0, u2≥0, u1+u2≤1} (как в element_integrals).
Точка: x = V0 + u1·(V1−V0) + u2·(V2−V0); dS = 2·area · du1 du2.
"""

_FOUR_PI = 4.0 * np.pi

# Случаи касания пар панелей (соответствуют FacePairRelation).
FACE = "face"      # совпадающие панели (single-layer self); double-layer self ≡ 0
EDGE = "edge"      # общее ребро
VERTEX = "vertex"  # общая вершина

_GRID_CACHE: dict[int, tuple] = {}


def gauss_legendre_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Узлы/веса Гаусса–Лежандра порядка n на [0,1]."""
    x, w = leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


def _tensor_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Тензорная сетка (η1,η2,η3,ξ) на [0,1]^4 + комбинированный вес, развёрнутая в (M,)."""
    if n in _GRID_CACHE:
        return _GRID_CACHE[n]
    p, w = gauss_legendre_01(n)
    e1, e2, e3, xi = np.meshgrid(p, p, p, p, indexing="ij")
    wt = np.einsum("i,j,k,l->ijkl", w, w, w, w)
    grid = (e1.ravel(), e2.ravel(), e3.ravel(), xi.ravel(), wt.ravel())
    _GRID_CACHE[n] = grid
    return grid


# --- Регуляризующие преобразования: списки (ux1, ux2, uy1, uy2, jac) ----------
# ux=(ux1,ux2) — опорные координаты точки x (тест-панель), uy — точки y (trial).


def _terms_common_face(e1, e2, e3, xi):
    j = xi**3 * e1**2 * e2
    a = xi - xi * e1
    b = xi - xi * e1 * e2 * e3
    c = xi - xi * e1 * e2
    return [
        (1 - xi, a + xi * e1 * e2, 1 - b, a, j),
        (1 - b, a, 1 - xi, a + xi * e1 * e2, j),
        (1 - xi, xi * e1 * (1 - e2 + e2 * e3), 1 - c, xi * e1 * (1 - e2), j),
        (1 - c, xi * e1 * (1 - e2), 1 - xi, xi * e1 * (1 - e2 + e2 * e3), j),
        (1 - b, xi * e1 * (1 - e2 * e3), 1 - xi, xi * e1 * (1 - e2), j),
        (1 - xi, xi * e1 * (1 - e2), 1 - b, xi * e1 * (1 - e2 * e3), j),
    ]


def _terms_common_edge(e1, e2, e3, xi):
    xe1 = xi * e1
    j1 = xi**3 * e1**2
    j2 = j1 * e2
    return [
        (1 - xi, xe1 * e3, 1 - xi * (1 - e1 * e2), xe1 * (1 - e2), j1),
        (1 - xi, xe1, 1 - xi * (1 - e1 * e2 * e3), xe1 * e2 * (1 - e3), j2),
        (1 - xi * (1 - e1 * e2), xe1 * (1 - e2), 1 - xi, xe1 * e2 * e3, j2),
        (1 - xi * (1 - e1 * e2 * e3), xe1 * e2 * (1 - e3), 1 - xi, xe1, j2),
        (1 - xi * (1 - e1 * e2 * e3), xe1 * (1 - e2 * e3), 1 - xi, xe1 * e2, j2),
    ]


def _terms_common_vertex(e1, e2, e3, xi):
    xe1 = xi * e1
    xe2 = xi * e2
    j = xi**3 * e2
    return [
        (1 - xi, xe1, 1 - xe2, xe2 * e3, j),
        (1 - xe2, xe2 * e3, 1 - xi, xe1, j),
    ]


_TERMS = {FACE: _terms_common_face, EDGE: _terms_common_edge, VERTEX: _terms_common_vertex}


def _accumulate(tri_x, tri_y, case, grid, point_integrand):
    """Σ по 4-кубу: W·jac·integrand(x,y), x∈tri_x, y∈tri_y. Возвращает скаляр или вектор."""
    e1, e2, e3, xi, wt = grid
    v0x, v1x, v2x = tri_x
    v0y, v1y, v2y = tri_y
    ex1, ex2 = v1x - v0x, v2x - v0x
    ey1, ey2 = v1y - v0y, v2y - v0y
    jac_x = float(np.linalg.norm(np.cross(ex1, ex2)))  # = 2·area_x
    jac_y = float(np.linalg.norm(np.cross(ey1, ey2)))  # = 2·area_y

    out = 0.0
    for ux1, ux2, uy1, uy2, jac in _TERMS[case](e1, e2, e3, xi):
        x = v0x[None, :] + ux1[:, None] * ex1[None, :] + ux2[:, None] * ex2[None, :]
        y = v0y[None, :] + uy1[:, None] * ey1[None, :] + uy2[:, None] * ey2[None, :]
        f = point_integrand(x, y, uy1, uy2)
        wj = wt * jac
        if np.ndim(f) == 1:
            out = out + float(np.dot(wj, f))
        else:
            out = out + (wj[:, None] * f).sum(axis=0)
    return out * (jac_x * jac_y)


def single_layer_pair(tri_x: np.ndarray, tri_y: np.ndarray, case: str, n_gauss: int = 6) -> float:
    """∫_{Tx}∫_{Ty} 1/(4π|x−y|) dS_y dS_x (касающаяся пара) через Заутера–Шваба."""
    grid = _tensor_grid(n_gauss)

    def pint(x, y, uy1, uy2):
        r = np.linalg.norm(x - y, axis=1)
        return 1.0 / (_FOUR_PI * r)

    return float(_accumulate(np.asarray(tri_x, float), np.asarray(tri_y, float), case, grid, pint))


def double_layer_pair_p1(
    tri_x: np.ndarray, tri_y: np.ndarray, n_y: np.ndarray, case: str, n_gauss: int = 6
) -> np.ndarray:
    """
    ∫_{Tx}∫_{Ty} [∂G/∂n_y · λ_m(y)] dS_y dS_x для m=0,1,2 (3 вершины trial-панели tri_y
    в её ТЕКУЩЕМ порядке вершин). Ядро ∂G/∂n_y = (x−y)·n_y/(4π|x−y|³).
    """
    grid = _tensor_grid(n_gauss)
    ny = np.asarray(n_y, dtype=float)

    def pint(x, y, uy1, uy2):
        rv = x - y
        r = np.linalg.norm(rv, axis=1)
        kern = (rv @ ny) / (_FOUR_PI * r**3)
        lam = np.stack([1.0 - uy1 - uy2, uy1, uy2], axis=1)  # (M,3) барицентрики trial
        return kern[:, None] * lam

    return np.asarray(_accumulate(np.asarray(tri_x, float), np.asarray(tri_y, float), case, grid, pint))


def constant_pair(tri_x: np.ndarray, tri_y: np.ndarray, case: str, n_gauss: int = 4) -> float:
    """∫_{Tx}∫_{Ty} 1 dS_y dS_x = area_x·area_y (тест меры преобразования)."""
    grid = _tensor_grid(n_gauss)

    def pint(x, y, uy1, uy2):
        return np.ones(x.shape[0], dtype=float)

    return float(_accumulate(np.asarray(tri_x, float), np.asarray(tri_y, float), case, grid, pint))


# --- Переупорядочивание вершин: общая фича → канонические опорные позиции ------
# Конвенция позиций сверяется ЭМПИРИЧЕСКИ (тест экспоненциальной сходимости),
# т.к. зависит от карты опорный симплекс→физика. Значения ниже закреплены
# калибровкой (см. tests/unit/test_bem_singular_quadrature.py).

# Закреплено калибровкой (test_bem_singular_quadrature.py::test_alignment_*):
# выбор позиций даёт наилучшую (экспоненциальную) сходимость по числу гауссовых узлов.
_VERTEX_SHARED_POS = 1            # общая вершина → опорная позиция (1,0) карты
_EDGE_SHARED_POS = (0, 1)         # общее ребро → опорные позиции (0,0)-(1,0); одинаковая ориентация обеих панелей
_EDGE_OPPOSITE_POS = 2            # противоположная вершина → опорная позиция (0,1)


def reorder_common_vertex(tri: np.ndarray, faces_row, shared_global: int):
    """Поставить общую вершину на позицию _VERTEX_SHARED_POS. Возврат: (tri', global_verts')."""
    tri = np.asarray(tri, float)
    g = [int(v) for v in faces_row]
    loc = g.index(int(shared_global))
    others = [i for i in range(3) if i != loc]
    order = [0, 0, 0]
    order[_VERTEX_SHARED_POS] = loc
    rem = [p for p in range(3) if p != _VERTEX_SHARED_POS]
    order[rem[0]] = others[0]
    order[rem[1]] = others[1]
    new_tri = tri[order]
    new_g = [g[k] for k in order]
    return new_tri, new_g


def reorder_common_edge(tri: np.ndarray, faces_row, shared_globals):
    """Поставить общее ребро на позиции _EDGE_SHARED_POS, противоположную вершину — на _EDGE_OPPOSITE_POS."""
    tri = np.asarray(tri, float)
    g = [int(v) for v in faces_row]
    s = [int(v) for v in shared_globals]
    loc_shared = [g.index(v) for v in s]
    loc_opp = [i for i in range(3) if i not in loc_shared][0]
    order = [0, 0, 0]
    order[_EDGE_SHARED_POS[0]] = loc_shared[0]
    order[_EDGE_SHARED_POS[1]] = loc_shared[1]
    order[_EDGE_OPPOSITE_POS] = loc_opp
    new_tri = tri[order]
    new_g = [g[k] for k in order]
    return new_tri, new_g
