import math

import numpy as np
import pytest

from magcore.fem2d.assembly import (
    assemble_current_rhs,
    assemble_magnetization_rhs,
    assemble_stiffness,
)
from magcore.fem2d.manufactured import manufactured_sine_current
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.post import h1_seminorm_error, l2_error, reconstruct_B_on_cells
from magcore.fem2d.solver import apply_dirichlet, solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D


def _solve_sine(n: int, nu0: float = 1.3):
    mesh = build_structured_rectangle_tri_mesh(n, n)
    space = LagrangeP1Space2D(mesh)
    mms = manufactured_sine_current(nu0)
    K = assemble_stiffness(space, nu0)
    f = assemble_current_rhs(space, mms.J_fn, quadrature_order=5)
    K_bc, f_bc = apply_dirichlet(K, f, space.boundary_dofs(), 0.0)  # A_z=0 на ∂Ω
    a = solve_scalar(K_bc, f_bc)
    return space, mms, a


def test_sine_solution_matches_exact_pointwise():
    space, mms, a = _solve_sine(24)
    # Центральный узел (0.5,0.5): точное A_z = 1.
    verts = space.mesh.vertices
    center = int(np.argmin(np.sum((verts - np.array([0.5, 0.5])) ** 2, axis=1)))
    assert a[center] == pytest.approx(1.0, abs=2e-3)
    # Однородный Dirichlet соблюдён точно.
    assert np.allclose(a[list(space.boundary_dofs())], 0.0, atol=1e-12)


def test_sine_convergence_orders():
    # L²→2, H¹→1 для скалярного P1 (в отличие от Неделека 1-го рода в 3D: L²→1).
    ns = [8, 16, 32]
    l2s, h1s = [], []
    for n in ns:
        space, mms, a = _solve_sine(n)
        l2s.append(l2_error(space, a, mms.A_exact, quadrature_order=5))
        h1s.append(h1_seminorm_error(space, a, mms.grad_A_exact, quadrature_order=5))

    # Монотонное убывание.
    assert l2s[0] > l2s[1] > l2s[2] > 0.0
    assert h1s[0] > h1s[1] > h1s[2] > 0.0

    # Наблюдаемые порядки p = log2(e_h / e_{h/2}).
    l2_rate = math.log2(l2s[1] / l2s[2])
    h1_rate = math.log2(h1s[1] / h1s[2])
    assert l2_rate == pytest.approx(2.0, abs=0.25)
    assert h1_rate == pytest.approx(1.0, abs=0.2)


def test_magnetization_rhs_equals_stiffness_times_linear_potential():
    # Дискретное тождество (2D-аналог 3D `F_mag = K·a_r`): для ОДНОРОДНОГО ν·B_r
    # источник магнита = K·a_r, где A_z^r = Brx·y − Bry·x порождает B = B_r.
    mesh = build_structured_rectangle_tri_mesh(6, 5)
    space = LagrangeP1Space2D(mesh)
    nu0 = 1.0
    Br = np.array([0.3, -0.7], dtype=float)
    nu_br_cells = np.tile(nu0 * Br, (mesh.n_cells, 1))

    f_mag = assemble_magnetization_rhs(space, nu_br_cells)
    K = assemble_stiffness(space, nu0)

    verts = mesh.vertices
    a_r = Br[0] * verts[:, 1] - Br[1] * verts[:, 0]      # узловой A_z^r
    assert np.allclose(f_mag, K @ a_r, atol=1e-12)

    # A_z^r действительно порождает B = B_r (проверка реконструкции B).
    B = reconstruct_B_on_cells(space, a_r)
    assert np.allclose(B, Br[None, :], atol=1e-12)
