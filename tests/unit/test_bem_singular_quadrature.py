from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import (
    AdaptiveIntegrationConfig,
    single_layer_triangle_pair_adaptive,
    single_layer_triangle_self_adaptive,
)
from magcore.bem.element_integrals import triangle_area
from magcore.bem.singular_quadrature import (
    EDGE,
    FACE,
    VERTEX,
    constant_pair,
    double_layer_pair_p1,
    reorder_common_edge,
    reorder_common_vertex,
    single_layer_pair,
)

# Опорные пары: A — в плоскости z=0; B_cv делит только вершину (0,0,0);
# B_ce делит ребро (0,0,0)-(1,0,0) и изгибается из плоскости (непланарно).
A = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
A_FACES = [0, 1, 2]
B_CV = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
B_CV_FACES = [0, 3, 4]
B_CE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
B_CE_FACES = [0, 1, 3]


def test_transform_measure_identity() -> None:
    # ∫∫ 1 dS_x dS_y = area_x · area_y для КАЖДОГО случая ⇒ якобиан преобразования
    # тайлит T̂×T̂ корректно (∫_{[0,1]^4} J_SS = area(T̂)² = 1/4).
    assert np.isclose(constant_pair(A, A, FACE, 4), triangle_area(A) ** 2, atol=1e-12)
    assert np.isclose(constant_pair(A, B_CV, VERTEX, 4), triangle_area(A) * triangle_area(B_CV), atol=1e-12)
    assert np.isclose(constant_pair(A, B_CE, EDGE, 4), triangle_area(A) * triangle_area(B_CE), atol=1e-12)


def test_self_single_layer_converges_exponentially() -> None:
    # CommonFace (self): экспоненциальная сходимость ⇒ значения n=8 и n=12 совпадают.
    v8 = single_layer_pair(A, A, FACE, 8)
    v12 = single_layer_pair(A, A, FACE, 12)
    assert abs(v8 - v12) < 1e-7
    # Согласие с адаптивным эталоном (тот менее точен ~3e-4 — его и заменяем).
    ref = single_layer_triangle_self_adaptive(A, AdaptiveIntegrationConfig(self_max_depth=7, max_depth=7))
    assert abs(v12 - ref) < 1e-3
    assert v12 > 0.0


def test_common_vertex_alignment_and_accuracy() -> None:
    # Закрепляет _VERTEX_SHARED_POS: правильное выравнивание ⇒ быстрая (экспон.)
    # сходимость (|v4−v12| мал); неверная позиция дала бы ~1e-5 (см. калибровку).
    tx, _gx = reorder_common_vertex(A, A_FACES, 0)
    ty, _gy = reorder_common_vertex(B_CV, B_CV_FACES, 0)
    v4 = single_layer_pair(tx, ty, VERTEX, 4)
    v12 = single_layer_pair(tx, ty, VERTEX, 12)
    assert abs(v4 - v12) < 2e-6
    ref = single_layer_triangle_pair_adaptive(A, B_CV, AdaptiveIntegrationConfig(max_depth=8))
    assert abs(v12 - ref) < 1e-4


def test_common_edge_alignment_and_accuracy() -> None:
    # Закрепляет _EDGE_SHARED_POS/_EDGE_OPPOSITE_POS (общее ребро, одинаковая
    # ориентация обеих панелей). Неверное размещение → |v4−v12| ≳ 5e-5.
    tx, _gx = reorder_common_edge(A, A_FACES, (0, 1))
    ty, _gy = reorder_common_edge(B_CE, B_CE_FACES, (0, 1))
    v4 = single_layer_pair(tx, ty, EDGE, 4)
    v12 = single_layer_pair(tx, ty, EDGE, 12)
    assert abs(v4 - v12) < 2e-5
    ref = single_layer_triangle_pair_adaptive(A, B_CE, AdaptiveIntegrationConfig(max_depth=8))
    assert abs(v12 - ref) < 1e-4


def test_double_layer_pair_p1_returns_finite_trial_vector() -> None:
    # Двойнослойное ядро на непланарной касающейся паре: 3-вектор по вершинам trial.
    n_y = np.array([0.0, 1.0, 0.0])  # внешняя нормаль B_CE (в плоскости y=0)
    tx, _gx = reorder_common_edge(A, A_FACES, (0, 1))
    ty, _gy = reorder_common_edge(B_CE, B_CE_FACES, (0, 1))
    vec = double_layer_pair_p1(tx, ty, n_y, EDGE, 6)
    assert vec.shape == (3,)
    assert np.isfinite(vec).all()
