import numpy as np
import scipy.sparse as sp

from magcore.fem2d.assembly import (
    assemble_current_rhs_piecewise,
    assemble_stiffness,
    assemble_stiffness_sparse,
    p1_cell_geometry,
)
from magcore.fem2d.mesh import p1_gradients, triangle_area
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.newton import assemble_newton_tangent
from magcore.fem2d.solver import apply_dirichlet, solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D

# Разрежённая сборка — фикс переполнения памяти (плотная (n,n) на реальной сетке = ГБ).
# Оракул НЕ «работает без ошибки», а ТОЧНОЕ совпадение с плотным эталоном по значениям +
# доказанная разрежённость (≈7 ненулей в строке ⇒ (n,n) не аллоцируется).


def _dense_newton_tangent(space, nu, dnu, a):
    """Плотная касательная старой формулой — эталон для векторизованной разрежённой."""
    mesh = space.mesh
    n = space.ndofs
    T = np.zeros((n, n))
    for c in range(mesh.n_cells):
        v = mesh.cell_vertices(c)
        A = triangle_area(v)
        g = p1_gradients(v)
        idx = list(mesh.cell_vertex_indices(c))
        w = g @ (g.T @ a[idx])
        loc = A * (nu[c] * (g @ g.T) + 2.0 * dnu[c] * np.outer(w, w))
        for i in range(3):
            for jj in range(3):
                T[idx[i], idx[jj]] += loc[i, jj]
    return T


def test_cell_geometry_matches_pointwise():
    mesh = build_structured_rectangle_tri_mesh(5, 5)
    cells, grad, area = p1_cell_geometry(mesh)
    assert np.array_equal(cells, np.asarray(mesh.cells))
    for c in range(mesh.n_cells):
        v = mesh.cell_vertices(c)
        assert np.allclose(grad[c], p1_gradients(v), atol=1e-12)
        assert np.isclose(area[c], triangle_area(v), atol=1e-14)


def test_sparse_stiffness_equals_dense():
    mesh = build_structured_rectangle_tri_mesh(10, 10)
    space = LagrangeP1Space2D(mesh)
    rng = np.random.default_rng(0)
    nu = rng.uniform(0.4, 2.5, mesh.n_cells)
    Kd = assemble_stiffness(space, nu)
    Ks = assemble_stiffness_sparse(space, nu)
    assert sp.issparse(Ks)
    assert np.allclose(Ks.toarray(), Kd, atol=1e-12)
    # P1-разрежённость: сильно меньше плотной (n² ненулей) — ~7·n.
    assert Ks.nnz < 10 * space.ndofs


def test_apply_dirichlet_sparse_equals_dense_solution():
    mesh = build_structured_rectangle_tri_mesh(12, 12)
    space = LagrangeP1Space2D(mesh)
    nu = np.full(mesh.n_cells, 1.3)
    cen = np.array([mesh.cell_centroid(c) for c in range(mesh.n_cells)])
    j = np.where(np.hypot(cen[:, 0] - 0.5, cen[:, 1] - 0.5) < 0.2, 1.0e3, 0.0)
    f = assemble_current_rhs_piecewise(space, j)
    bd = space.boundary_dofs()

    Kd_bc, fd_bc = apply_dirichlet(assemble_stiffness(space, nu), f, bd, 0.0)
    Ks_bc, fs_bc = apply_dirichlet(assemble_stiffness_sparse(space, nu), f, bd, 0.0)
    assert sp.issparse(Ks_bc)
    xd = solve_scalar(Kd_bc, fd_bc)
    xs = solve_scalar(Ks_bc, fs_bc)
    assert np.allclose(xd, xs, atol=1e-10)


def test_newton_tangent_sparse_matches_dense_formula():
    mesh = build_structured_rectangle_tri_mesh(8, 8)
    space = LagrangeP1Space2D(mesh)
    rng = np.random.default_rng(1)
    nu = rng.uniform(0.5, 2.0, mesh.n_cells)
    dnu = rng.uniform(-0.3, 0.3, mesh.n_cells)
    a = rng.standard_normal(space.ndofs)
    Ts = assemble_newton_tangent(space, nu, dnu, a)
    Td = _dense_newton_tangent(space, nu, dnu, a)
    assert sp.issparse(Ts)
    assert np.allclose(Ts.toarray(), Td, atol=1e-12)
    assert np.allclose(Ts.toarray(), Ts.toarray().T, atol=1e-12)  # симметрия
