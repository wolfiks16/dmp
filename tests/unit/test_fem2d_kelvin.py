import numpy as np
import pytest

from magcore.fem2d.kelvin import solve_kelvin_magnetostatic
from magcore.fem2d.mesh_generators import build_disk_tri_mesh


def _magnetized_cylinder(nr, nt, *, a_disk=3.0, R_m=1.0, Br=1.0):
    """Концентрич. намагниченный цилиндр (B_r вдоль x) в диске; R_m выровнен на кольцо."""
    disk = build_disk_tri_mesh(a_disk, nr, nt)
    mesh = disk.mesh
    nu_br = np.zeros((mesh.n_cells, 2), dtype=float)
    magnet_cells = []
    for c in range(mesh.n_cells):
        if np.linalg.norm(mesh.cell_centroid(c)) < R_m:
            nu_br[c] = [Br, 0.0]
            magnet_cells.append(c)
    res = solve_kelvin_magnetostatic(disk, nu_real=np.ones(mesh.n_cells), magnetization=nu_br)
    H = res.B_cells[magnet_cells] - np.array([Br, 0.0])   # ν=1 ⇒ H = B − B_r
    return res, np.asarray(magnet_cells), H


def test_kelvin_magnetized_cylinder_demag_factor():
    # 2D demag-фактор цилиндра = 1/2 ⇒ H_in = −M/2 (аналог сферы −M/3), поле ОДНОРОДНО.
    # Первая физическая проверка всей связки FEM(магнит)↔Kelvin(точная внешность).
    _, _, H = _magnetized_cylinder(24, 96)   # R_m = ring 8 (точно)
    assert H[:, 0].mean() == pytest.approx(-0.5, abs=3e-3)   # H_x → −M/2
    assert abs(H[:, 1].mean()) < 1e-6                        # поперечное ≈ 0
    assert H[:, 0].std() < 1e-6                              # однородно внутри
    # B_in = M/2 (= B_r + H_in).
    Bx = (H[:, 0] + 1.0).mean()
    assert Bx == pytest.approx(0.5, abs=3e-3)


def test_kelvin_demag_convergence_to_half():
    errs = []
    for nr in (12, 24, 36):                  # кратно 3 ⇒ R_m=1.0 попадает на кольцо
        _, _, H = _magnetized_cylinder(nr, 4 * nr)
        errs.append(abs(H[:, 0].mean() + 0.5))
    assert errs[0] > errs[1] > errs[2] > 0.0                # монотонная сходимость к −0.5
    assert errs[2] < 1e-3


def test_kelvin_zero_source_gives_zero_field():
    # Без источников открытая задача имеет тривиальное решение (нет паразитных мод).
    disk = build_disk_tri_mesh(2.0, 10, 40)
    res = solve_kelvin_magnetostatic(disk, nu_real=np.ones(disk.mesh.n_cells))
    assert np.allclose(res.a, 0.0, atol=1e-12)
    assert np.allclose(res.B_cells, 0.0, atol=1e-12)
