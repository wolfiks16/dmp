from types import SimpleNamespace

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_model import n42sh_magnet
from magcore.fem2d.coupled_transient import IrreversibleMagnetState
from magcore.hybrid.magnet_demag import compute_demag_risk_map

# Необратимая память магнита — доля сохранённой ремнантности r (Л-100, постановка (S2) в
# docs/math/coupled_problem.md). Оракулы:
#  (1) края закона: выше колена и в подмагничивающем поле r_now = 1 ровно, ниже −H_cJ — 0 ровно, в самой
#      точке −H_cJ линия возврата из (−H_cJ, −μ₀H_cJ) даёт μ₀·H_cJ·(μ_rec − 1)/B_r; при одной температуре
#      r_now·B_r = B_r,eff(H) — та же физика, что «потеря по полю»;
#  (2) поле переключения обращает r_now: r = 1 → колено, r ниже порога таблицы → −H_cJ;
#  (3) копии одной физики совпадают: r_now и главная кривая модели магнита = ядро К6′ (Л-92, Л-100);
#  (4) карта риска с историей переносит потерянную долю на другую температуру; память «по худшему
#      полю» при тех же числах потерю бы потеряла.

TEMPS = (20.0, 100.0, 150.0)


def test_retention_now_edges_and_monotone():
    mag = n42sh_magnet((1.0, 0.0, 0.0))
    for T in TEMPS:
        knee, hcj, br = mag.knee_field(T), mag.Hcj(T), mag.Br(T)
        assert np.all(mag.retention_now(np.array([3.0e5, 1.0, 0.0, 0.5 * knee, knee, knee - 1.0e-3]), T) == 1.0)
        assert np.all(mag.retention_now(np.array([-1.001 * hcj, -2.0 * hcj]), T) == 0.0)
        # первый узел таблицы лежит на прямой J = 0 (B = μ₀H) с точностью 1e-14 ⇒ r до ~1e-13
        assert float(mag.retention_now(-hcj, T)) == pytest.approx(MU0 * hcj * (mag.mu_rec - 1.0) / br, rel=1e-12)
        h = np.linspace(-hcj, knee, 400)
        r = mag.retention_now(h, T)
        assert np.all(np.diff(r) >= 0.0) and r[0] < 1.0 and r[-1] == 1.0
        # при одной температуре — та же величина, что эффективная ремнантность «по полю»; различие только
        # в привязке к единице в полосе 1e-9 под коленом
        assert np.allclose(r * br, mag.effective_Br(h, T), rtol=0.0, atol=1.0e-9 * br)


def test_switch_field_inverts_retention_now():
    mag = n42sh_magnet((1.0, 0.0, 0.0))
    for T in TEMPS:
        knee, hcj = mag.knee_field(T), mag.Hcj(T)
        r_floor = float(mag.retention_now(-hcj, T))
        assert float(mag.switch_field(1.0, T)) == pytest.approx(knee, rel=1e-12)
        assert float(mag.switch_field(0.0, T)) == pytest.approx(-hcj, rel=1e-12)
        assert float(mag.switch_field(0.5 * r_floor, T)) == pytest.approx(-hcj, rel=1e-12)
        r = np.linspace(r_floor, 1.0 - 1.0e-6, 200)          # вне полосы привязки к единице
        hs = mag.switch_field(r, T)
        assert np.all(np.diff(hs) > 0.0)
        assert np.allclose(mag.retention_now(hs, T), r, rtol=0.0, atol=1e-12)   # округление double, запас ×1000


def test_retention_law_matches_the_k6_core():
    # В ядре К6′ (`IrreversibleMagnetState`) r_now и главная кривая считаются своим кодом внутри итерации;
    # здесь — что это та же величина, что в модели магнита, по которой работают 3D и карта риска.
    # Температуры — на узлах сетки температур ядра (шаг 0,5 °C); поля сдвинуты с узлов таблицы — в самом
    # узле наклон двух реализаций берётся с разных сторон. Приватные `_curve_eval` и `_pending` — это
    # и есть проверяемые места ядра.
    mag = n42sh_magnet((1.0, 0.0, 0.0))
    n = 97
    for T in TEMPS:
        h = np.linspace(-1.2 * mag.Hcj(T), 3.0e5, n) + 0.123
        state = IrreversibleMagnetState(mag, np.ones(n, dtype=bool), n, axis=(1.0, 0.0))
        state.set_temperature(np.full(n, T))
        b_maj, slope, br_nom, beyond = state._curve_eval(h)
        assert np.allclose(b_maj, mag.B_major_parallel(h, T), rtol=0.0, atol=1e-13)
        assert np.allclose(slope, mag.B_major_slope(h, T), rtol=1e-12, atol=0.0)
        assert np.all(br_nom == mag.Br(T)) and np.array_equal(beyond, h < -mag.Hcj(T))
        state(np.zeros((n, 2)), MU0 * np.column_stack([h, np.zeros(n)]), np.ones(n))   # поле решателя в Тл
        assert np.allclose(state._pending, mag.retention_now(h, T), rtol=0.0, atol=1e-13)


def test_risk_map_keeps_the_loss_after_cooling():
    mag = n42sh_magnet((1.0, 0.0, 0.0))
    T_hot, T_cold = 150.0, 20.0
    h_dmg = 0.5 * (mag.knee_field(T_hot) - mag.Hcj(T_hot))            # за коленом, выше −H_cJ при 150 °C
    mask = np.ones(3, dtype=bool)
    loaded = SimpleNamespace(H_cells=MU0 * np.array([[h_dmg, 0.0, 0.0], [-1.0e5, 0.0, 0.0], [2.0e5, 0.0, 0.0]]))
    hot = compute_demag_risk_map(mag, loaded, mask, T_hot)
    r = hot.retention
    assert 0.0 < r[0] < 1.0 and r[1] == 1.0 and r[2] == 1.0 and hot.n_damaged == 1     # предпосылки
    cold = compute_demag_risk_map(mag, SimpleNamespace(H_cells=np.zeros((3, 3))), mask, T_cold, retention=r)
    assert np.array_equal(cold.retention, r) and cold.n_damaged == 1 and not cold.beyond_hcj.any()
    assert np.allclose(cold.loss, mag.Br(T_cold) * (1.0 - r), rtol=0.0, atol=1e-15)
    assert np.allclose(cold.Br_eff, r * mag.Br(T_cold), rtol=0.0, atol=1e-15)
    # то, что делала память «по худшему полю»: при 20 °C то же поле выше колена — потеря исчезла бы
    assert float(mag.irreversible_loss(h_dmg, T_cold)) == pytest.approx(0.0, abs=1e-12)
    # история без потери не создаёт её, а ниже −H_cJ — полная потеря с флагом
    fresh = compute_demag_risk_map(mag, SimpleNamespace(H_cells=np.zeros((3, 3))), mask, T_cold, retention=np.ones(3))
    assert fresh.n_damaged == 0 and np.all(fresh.loss == 0.0)
    deep = compute_demag_risk_map(mag, SimpleNamespace(H_cells=MU0 * np.array([[-2.0 * mag.Hcj(T_hot), 0.0, 0.0]] * 3)),
                                  mask, T_hot)
    assert deep.beyond_hcj.all() and np.all(deep.loss == mag.Br(T_hot))
