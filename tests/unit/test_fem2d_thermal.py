import math

import numpy as np
import pytest

from magcore.fem2d.manufactured import manufactured_sine_current
from magcore.fem2d.mesh_generators import (
    build_disk_tri_mesh,
    build_structured_rectangle_tri_mesh,
)
from magcore.fem2d.post import l2_error
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import solve_thermal


def test_thermal_dirichlet_mms_convergence():
    # Тепловая задача −div(k∇T)=q структурно = магнитостатика (k↔ν); та же MMS: L²→2.
    k = 1.3
    ns = [8, 16, 32]
    errs = []
    for n in ns:
        mesh = build_structured_rectangle_tri_mesh(n, n)
        space = LagrangeP1Space2D(mesh)
        mms = manufactured_sine_current(k)          # q = 2π²k·sinπx·sinπy, T_exact = sin·sin
        T = solve_thermal(
            space, k, source=mms.J_fn,
            dirichlet_dofs=space.boundary_dofs(), dirichlet_values=0.0,
        )
        errs.append(l2_error(space, T, mms.A_exact))
    assert errs[0] > errs[1] > errs[2] > 0.0
    assert math.log2(errs[1] / errs[2]) == pytest.approx(2.0, abs=0.25)


def _radial_exact(q, R, k, h, T_amb):
    def T(x):
        r2 = x[0] ** 2 + x[1] ** 2
        return T_amb + q * R / (2.0 * h) + q * (R * R - r2) / (4.0 * k)
    return T


def test_thermal_radial_convection_matches_analytic():
    # Диск с равномерным тепловыделением q и конвекцией (Robin) h на кромке:
    # T(r) = T_amb + qR/(2h) + q(R²−r²)/(4k). Проверяет источник + Robin вместе.
    R, q, k, h, T_amb = 1.0, 1.0, 1.3, 2.0, 10.0
    exact = _radial_exact(q, R, k, h, T_amb)

    disk = build_disk_tri_mesh(R, 24, 96)
    space = LagrangeP1Space2D(disk.mesh)
    T = solve_thermal(space, k, source=np.full(disk.mesh.n_cells, q), h=h, T_amb=T_amb)

    # Центр (r=0) и кромка (r=R) против аналитики.
    assert T[disk.center_node] == pytest.approx(exact(np.zeros(2)), abs=1e-3)
    T_edge = T[disk.boundary_nodes]
    assert np.allclose(T_edge, T_amb + q * R / (2.0 * h), atol=5e-3)
    # Глобально мал L²; тепло НЕ убегает (максимум в центре, минимум на кромке).
    assert l2_error(space, T, exact) < 2e-3
    assert T[disk.center_node] > T_edge.mean()


def test_thermal_radial_convergence_order():
    R, q, k, h, T_amb = 1.0, 1.0, 1.3, 2.0, 10.0
    exact = _radial_exact(q, R, k, h, T_amb)
    errs = []
    for nr in (12, 24, 36):
        disk = build_disk_tri_mesh(R, nr, 4 * nr)
        space = LagrangeP1Space2D(disk.mesh)
        T = solve_thermal(space, k, source=np.full(disk.mesh.n_cells, q), h=h, T_amb=T_amb)
        errs.append(l2_error(space, T, exact))
    assert errs[0] > errs[1] > errs[2] > 0.0    # монотонная сходимость к аналитике
