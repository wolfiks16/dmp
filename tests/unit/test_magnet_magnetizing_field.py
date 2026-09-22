import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_model import n42sh_magnet
from magcore.hybrid.magnet_demag import MagnetDemagPolicy, compute_demag_risk_map

# Л-92 (2026-09-14): главная кривая магнита за краями таблицы [−H_cJ, 0] обрезалась. В
# намагничивающем поле (H∥ > 0) это давало ложную необратимую «потерю» μ0·μ_rec·H — 0,14 Тл
# при +100 кА/м — в статическом решателе и в карте риска (там берётся текущее поле, а не
# наихудшее из истории). Физика: при H∥ > 0 магнит идёт по линии возврата выше B_r и ничего не
# теряет; левее −H_cJ собственная намагниченность уже ноль, наклон кривой — μ0.

T_LIST = (20.0, 150.0)


@pytest.mark.parametrize("T", T_LIST)
def test_magnetizing_field_causes_no_irreversible_loss(T):
    m = n42sh_magnet((0.0, 0.0, 1.0))
    H = np.array([1.0e3, 1.0e5, 3.0e5])
    assert np.allclose(m.irreversible_loss(H, T), 0.0, atol=1e-12)
    assert np.allclose(m.effective_Br(H, T), m.Br(T), rtol=1e-12)
    assert np.isclose(float(m.B_major_parallel(0.0, T)), m.Br(T))            # непрерывна в нуле
    assert np.allclose(m.B_major_slope(H, T), MU0 * m.mu_rec)                  # линия возврата
    assert np.isclose(float(m.B_major_slope(-1.0, T)), MU0 * m.mu_rec, rtol=1e-6)   # и слева от нуля


@pytest.mark.parametrize("T", T_LIST)
def test_beyond_intrinsic_coercivity_loss_is_total_as_in_coupled_core(T):
    # Как в ядре К6′ (coupled_transient._curve_eval): левее −H_cJ — консервативно полная потеря,
    # линия возврата из начала координат, B_r,eff = 0, наклон μ0·μ_rec.
    m = n42sh_magnet((0.0, 0.0, 1.0))
    h_cj = -m.Hcj(T)
    assert np.isclose(float(m.B_major_parallel(h_cj, T)), MU0 * h_cj, rtol=1e-6)   # J = 0 в −H_cJ
    for dh in (1.0e3, 1.0e5):
        h = h_cj - dh
        assert np.isclose(float(m.B_major_parallel(h, T)), MU0 * m.mu_rec * h, rtol=1e-12)
        assert abs(float(m.effective_Br(h, T))) < 1e-12
        assert np.isclose(float(m.irreversible_loss(h, T)), m.Br(T))
        assert np.isclose(float(m.B_major_slope(h, T)), MU0 * m.mu_rec)


def test_slope_matches_finite_difference_of_the_curve():
    m = n42sh_magnet((0.0, 0.0, 1.0))
    T = 150.0
    H = np.linspace(-0.95 * m.Hcj(T), -1.0e3, 40)
    dh = 1.0
    fd = (np.asarray(m.B_major_parallel(H + dh, T)) - np.asarray(m.B_major_parallel(H - dh, T))) / (2 * dh)
    # ломаная: разность совпадает с наклоном отрезка везде, кроме узлов таблицы
    close = np.isclose(np.asarray(m.B_major_slope(H, T)), fd, rtol=1e-6)
    assert close.mean() > 0.9


def test_policy_and_risk_map_report_no_loss_in_magnetizing_field():
    m = n42sh_magnet((0.0, 0.0, 1.0))
    n = 4
    mask = np.ones(n, dtype=bool)
    H_cells = np.zeros((n, 3))
    H_cells[:, 2] = MU0 * np.array([-1.0e5, 0.0, 1.0e5, 3.0e5])               # в единицах решателя μ0·H
    policy = MagnetDemagPolicy(m, mask, T=20.0, n_cells=n)
    src = policy(np.zeros((n, 3)), H_cells, None)
    assert np.allclose(src[:, 2], m.Br(20.0) / m.mu_rec)                       # ν·B_r без потерь
    risk = compute_demag_risk_map(m, type("R", (), {"H_cells": H_cells})(), mask, 20.0)
    assert np.allclose(risk.loss, 0.0, atol=1e-12) and risk.n_demagnetized == 0
