from types import SimpleNamespace

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_model import n42sh_magnet
from magcore.fem2d.kelvin import solve_nonlinear_kelvin_2d_picard
from magcore.fem2d.mesh_generators import build_disk_tri_mesh
from magcore.hybrid.magnet_demag import MagnetDemagPolicy, compute_demag_risk_map


# --------------------------------------------------------------------------------------
# (A) ДЕТЕРМИНИРОВАННО: логика risk-map/политики демага в 2D на ЗАДАННОМ поле.
# 2D-порт 3D-теста (test_hybrid_magnet_knee::test_risk_map_logic_on_prescribed_field):
# та же скалярная B(H,T)-машина магнита, ось проекции — плоскостная (1,0). Частичная
# картина по построению (одна ячейка безопасна, две за коленом). Независимый оракул.
# --------------------------------------------------------------------------------------
def test_demag_risk_map_logic_2d_prescribed_field():
    magnet = n42sh_magnet(easy_axis=[1.0, 0.0, 0.0])   # 3D-ось для конструктора магнита
    T = 80.0
    knee = magnet.knee_field(T)                         # < 0 [А/м]

    h_phys = np.array([0.5, 1.05, 1.15]) * knee         # безопасно, чуть и глубже за коленом
    n = h_phys.size
    H_cells = np.zeros((n, 2))
    H_cells[:, 0] = MU0 * h_phys                        # H вдоль плоскостной оси x, мост H_solver=μ₀·H_phys
    result = SimpleNamespace(H_cells=H_cells)

    rmap = compute_demag_risk_map(
        magnet, result, np.ones(n, dtype=bool), T, axis=(1.0, 0.0)
    )

    # Мост единиц восстанавливает физические А/м.
    assert np.allclose(rmap.H_par, h_phys)
    # Маржа: первая безопасна, две за коленом (всё глубже).
    assert rmap.margin[0] > 0.0
    assert rmap.margin[2] < rmap.margin[1] < 0.0
    assert np.array_equal(rmap.demagnetized, np.array([False, True, True]))
    # Потеря: 0 выше колена, растёт с глубиной за коленом.
    assert abs(rmap.loss[0]) < 1e-9
    assert rmap.loss[2] > rmap.loss[1] > 0.0
    assert rmap.Br_eff[2] < rmap.Br_eff[1] < rmap.Br_nominal


# --------------------------------------------------------------------------------------
# (B) END-TO-END: магнит с состоянием (MagnetDemagPolicy) в ОТКРЫТОЙ 2D-области (Kelvin)
# через ОБЩЕЕ ядро run_picard_fixed_point. NdFeB при комнатной T — физически БЕЗОПАСЕН
# (саморазмагничивание цилиндра −M/2 ≈ −5e5 А/м много выше колена −1.35e6) ⇒ 0 потерь.
# --------------------------------------------------------------------------------------
def test_kelvin_magnet_safe_at_room_temperature_2d():
    a_disk, R_m = 3.0, 1.0
    disk = build_disk_tri_mesh(a_disk, 12, 48)         # R_m=1.0 = кольцо 4 (точно)
    mesh = disk.mesh
    nc = mesh.n_cells

    magnet = n42sh_magnet(easy_axis=[1.0, 0.0, 0.0])
    nu_mag = 1.0 / magnet.mu_rec
    mask = np.array(
        [np.linalg.norm(mesh.cell_centroid(c)) < R_m for c in range(nc)], dtype=bool
    )
    nu_cells = np.where(mask, nu_mag, 1.0)             # магнит=recoil, воздух=1

    policy = MagnetDemagPolicy(magnet, mask, T=20.0, n_cells=nc, axis=(1.0, 0.0))
    res = solve_nonlinear_kelvin_2d_picard(
        disk, nu_of_B=lambda B: nu_cells.copy(), nu_init=nu_cells,
        magnetization=policy, tol=1e-6, max_iter=40,
    )
    assert res.converged
    assert res.n_iterations <= 4                       # выше колена B_r_eff const ⇒ быстро

    rmap = compute_demag_risk_map(magnet, res, mask, T=20.0, axis=(1.0, 0.0))
    assert rmap.n_demagnetized == 0
    assert np.all(rmap.margin > 0.0)
    assert np.allclose(rmap.Br_eff, rmap.Br_nominal)
    assert abs(rmap.total_loss) < 1e-9

    # Саморазмагничивание физично: поле в магните ВСТРЕЧНОЕ (H_par < 0) и умеренное.
    assert np.all(rmap.H_par < 0.0)
    assert rmap.H_par.min() > magnet.knee_field(20.0)  # выше колена (не за ним)
