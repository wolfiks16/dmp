import numpy as np

from magcore.fem2d.losses import (
    CU_ALPHA,
    CU_RHO0,
    CU_T0,
    copper_loss_density,
    copper_loss_sensitivity,
    copper_resistivity,
)
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh

# S3, инкремент 2: плотность потерь меди. ОРАКУЛЫ (физика, не «работает»):
# (1) ρ(T) линейна с верными точками; (2) интеграл q по проводнику × длина = I²·R(T)
#     (джоулево тепло = мощность резистора) при T=20 и T=120 — проверяет и формулу, и
#     T-зависимость; (3) чувствительность dq/dT совпадает с конечной разностью.


def test_resistivity_endpoints():
    assert abs(copper_resistivity(CU_T0) - CU_RHO0) < 1e-20
    assert abs(copper_resistivity(CU_T0 + 100.0) - CU_RHO0 * (1 + 100 * CU_ALPHA)) < 1e-20


def test_integrated_loss_equals_i2r():
    # Проводник (прямоугольная область) с ОДНОРОДНОЙ плотностью тока J. Ток I=J·A_cross,
    # сопротивление R(T)=ρ(T)·L/A_cross ⇒ мощность I²R = ∫q dA · L. Проверяем на 2 T.
    mesh = build_structured_rectangle_tri_mesh(8, 8)
    A = float(sum(mesh.cell_area(c) for c in range(mesh.n_cells)))   # сечение проводника [м²]
    L = 0.03                                                          # осевая длина [м]
    J = 5.0e6                                                         # А/м²
    I = J * A
    j_cells = np.full(mesh.n_cells, J)
    for T in (20.0, 120.0):
        q = copper_loss_density(j_cells, np.full(mesh.n_cells, T))
        areas = np.array([mesh.cell_area(c) for c in range(mesh.n_cells)])
        P_field = float((q * areas).sum()) * L                       # ∫q dV
        R = copper_resistivity(T) * L / A
        P_i2r = I * I * R
        assert abs(P_field - P_i2r) / P_i2r < 1e-10


def test_sensitivity_matches_finite_difference():
    J = np.array([0.0, 3.0e6, 6.0e6])
    T, dT = 80.0, 1.0
    fd = (copper_loss_density(J, np.full(3, T + dT)) - copper_loss_density(J, np.full(3, T))) / dT
    s = copper_loss_sensitivity(J)
    assert np.allclose(fd, s, rtol=1e-9)
    assert s[0] == 0.0 and s[2] == 4.0 * s[1]        # ∝ J², и ноль вне проводника
