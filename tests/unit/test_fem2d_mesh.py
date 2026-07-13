import numpy as np
import pytest

from magcore.fem2d.mesh import TriangleMesh, p1_gradients, signed_area2, triangle_area
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh


def _unit_tri():
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)


def test_signed_area_positive_for_ccw_and_negative_for_cw():
    ccw = _unit_tri()
    assert signed_area2(ccw) == pytest.approx(1.0)
    assert triangle_area(ccw) == pytest.approx(0.5)
    cw = ccw[[0, 2, 1]]
    assert signed_area2(cw) == pytest.approx(-1.0)


def test_p1_gradient_identities():
    # Несколько произвольных CCW-треугольников.
    tris = [
        _unit_tri(),
        np.array([[0.2, -0.1], [1.3, 0.4], [0.5, 1.1]], dtype=float),
        np.array([[-1.0, -1.0], [2.0, 0.0], [0.0, 2.5]], dtype=float),
    ]
    for p in tris:
        g = p1_gradients(p)                      # (3,2)
        # Разбиение единицы градиентов: Σ_i ∇φ_i = 0.
        assert np.allclose(g.sum(axis=0), 0.0, atol=1e-12)
        # Линейность φ_i: ∇φ_i·(p_j − p_i) = −1 при j≠i, = 0 при j=i.
        for i in range(3):
            for j in range(3):
                dot = float(g[i] @ (p[j] - p[i]))
                assert dot == pytest.approx(-1.0 if j != i else 0.0, abs=1e-12)


def test_structured_rectangle_area_and_counts():
    mesh = build_structured_rectangle_tri_mesh(4, 4, x0=0.0, x1=2.0, y0=0.0, y1=3.0)
    assert mesh.n_vertices == 25
    assert mesh.n_cells == 32                    # 2 треугольника на квадрат, 16 квадратов
    total = sum(mesh.cell_area(c) for c in range(mesh.n_cells))
    assert total == pytest.approx(6.0)           # площадь прямоугольника 2×3


def test_structured_rectangle_boundary():
    n = 5
    mesh = build_structured_rectangle_tri_mesh(n, n)
    bverts = mesh.boundary_vertices()
    assert len(bverts) == 4 * n                  # периметр структурированной сетки
    # Каждая граничная вершина лежит на кромке единичного квадрата.
    for v in bverts:
        x, y = mesh.vertices[v]
        on_edge = (
            np.isclose(x, 0.0) or np.isclose(x, 1.0)
            or np.isclose(y, 0.0) or np.isclose(y, 1.0)
        )
        assert on_edge
    # Внутренние вершины — НЕ на границе.
    interior = set(range(mesh.n_vertices)) - set(bverts)
    assert len(interior) == (n - 1) ** 2


def test_rejects_non_ccw_triangle():
    verts = _unit_tri()
    cw_cell = np.array([[0, 2, 1]], dtype=int)   # по часовой ⇒ отрицательная площадь
    with pytest.raises(ValueError, match="non-positive signed area"):
        TriangleMesh(vertices=verts, cells=cw_cell)


def test_rejects_bad_shapes_and_indices():
    verts = _unit_tri()
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        TriangleMesh(vertices=np.zeros((3, 3)), cells=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match=r"shape \(M, 3\)"):
        TriangleMesh(vertices=verts, cells=np.array([[0, 1, 2, 0]]))
    with pytest.raises(ValueError, match="out-of-range"):
        TriangleMesh(vertices=verts, cells=np.array([[0, 1, 5]]))


def test_rejects_duplicate_triangle():
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    cells = np.array([[0, 1, 2], [2, 0, 1]], dtype=int)  # тот же треугольник
    with pytest.raises(ValueError, match="Duplicate triangle"):
        TriangleMesh(vertices=verts, cells=cells)
