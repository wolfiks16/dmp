import dataclasses

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.fem2d.magnet_law import MagnetLaw2D

# Рабочая ветвь закона магнита — ОДНА реализация на 2D и 3D (Л-92, Л-100): главная кривая, когда идёт
# новая необратимая потеря, иначе линия возврата с сохранённой долей r. В 3D неизвестное — поле
# (нужен прямой закон B(H)), в планарной 2D — индукция (нужно обращение H(B)); здесь проверяется, что
# это ровно одна и та же кривая, и что наклон берётся у той же ветви (Л-21, Л-93).

MAGNETS = [n42sh_magnet((1.0, 0.0, 0.0)), sm2co17_magnet((1.0, 0.0, 0.0))]


@pytest.mark.parametrize("magnet", MAGNETS, ids=lambda m: m.material_id)
@pytest.mark.parametrize("T", [20.0, 150.0])
@pytest.mark.parametrize("r", [1.0, 0.9, 0.5, 0.0])
def test_inverse_law_returns_the_same_point_and_the_same_slope(magnet, T, r):
    """H → B → H возвращает исходное поле, а наклон совпадает: обращение точное, а не приближённое."""
    H = np.linspace(-1.25 * magnet.Hcj(T), 3.0e5, 2001)
    b, s = magnet.branch_parallel(H, T, r)
    h_back, s_back = magnet.branch_parallel_inverse(b, T, r)
    assert np.abs(h_back - H).max() < 1.0e-6                      # А/м при полях в сотни кА/м
    assert np.allclose(s_back, s, rtol=0.0, atol=0.0)
    assert np.all(np.diff(b) > -1e-15)                            # закон монотонен ⇒ обращение однозначно


@pytest.mark.parametrize("magnet", MAGNETS, ids=lambda m: m.material_id)
@pytest.mark.parametrize("r", [1.0, 0.7])
def test_branches_meet_at_the_switch_field(magnet, r):
    """В точке переключения главная кривая и линия возврата дают одно B — закон непрерывен."""
    T = 140.0
    h_star = float(magnet.switch_field(r, T))
    b_major = float(magnet.B_major_parallel(h_star, T))
    b_recoil = r * magnet.Br(T) + MU0 * magnet.mu_rec * h_star
    assert b_major == pytest.approx(b_recoil, abs=1e-9)
    left, _ = magnet.branch_parallel(h_star - 1.0e3, T, r)        # слева — главная кривая
    right, _ = magnet.branch_parallel(h_star + 1.0e3, T, r)       # справа — линия возврата
    assert float(left) < b_major < float(right)


@pytest.mark.parametrize("magnet", MAGNETS, ids=lambda m: m.material_id)
def test_new_magnet_follows_the_major_curve_past_the_knee_and_the_recoil_line_above_it(magnet):
    T = 150.0
    knee = magnet.knee_field(T)
    past = knee - 50.0e3
    above = knee + 50.0e3
    b_past, s_past = magnet.branch_parallel(past, T, 1.0)
    assert float(b_past) == pytest.approx(float(magnet.B_major_parallel(past, T)), abs=1e-12)
    assert float(s_past) > MU0 * magnet.mu_rec                    # за коленом кривая круче линии возврата
    b_above, s_above = magnet.branch_parallel(above, T, 1.0)
    assert float(b_above) == pytest.approx(magnet.Br(T) + MU0 * magnet.mu_rec * above, abs=1e-12)
    assert float(s_above) == pytest.approx(MU0 * magnet.mu_rec, rel=1e-12)


@pytest.mark.parametrize("magnet", MAGNETS, ids=lambda m: m.material_id)
@pytest.mark.parametrize("r", [1.0, 0.6, 0.2])
def test_remanence_at_zero_field_is_the_retained_share(magnet, r):
    """Снятое поле: B(0) = r·B_r(T) — потерянная доля не возвращается (постановка (S2), Л-100)."""
    T = 120.0
    b0, _ = magnet.branch_parallel(0.0, T, r)
    assert float(b0) == pytest.approx(r * magnet.Br(T), rel=1e-12)


@pytest.mark.parametrize("magnet", MAGNETS, ids=lambda m: m.material_id)
def test_slope_matches_the_numeric_derivative_of_the_same_branch(magnet):
    """Наклон — производная ТОЙ ЖЕ ветви (согласованная линеаризация): сверка конечной разностью."""
    T = 150.0
    H = np.linspace(magnet.knee_field(T) - 80.0e3, magnet.knee_field(T) - 20.0e3, 37)
    _, s = magnet.branch_parallel(H, T, 1.0)
    d = 1.0                                                        # А/м — внутри отрезка таблицы
    b_plus, _ = magnet.branch_parallel(H + d, T, 1.0)
    b_minus, _ = magnet.branch_parallel(H - d, T, 1.0)
    assert np.allclose((b_plus - b_minus) / (2 * d), s, rtol=1e-6, atol=0.0)


def test_planar_law_gives_axial_branch_and_constant_perpendicular_permeability():
    """MagnetLaw2D: вдоль оси — ветвь закона, поперёк — μ⊥ материала (касательная одноосная)."""
    # μ⊥ — физическая (NdFeB ≈ 1,17, см. ablation_nu_tensor.py), не равная μ_rec: иначе подмена μ⊥ на μ_rec
    # поперёк оси тестом не различалась бы (по умолчанию у марок μ⊥ = μ_rec).
    magnet = dataclasses.replace(n42sh_magnet((1.0, 0.0, 0.0)), mu_perp=1.17)
    assert abs(magnet.mu_perp - magnet.mu_rec) > 0.03
    T = 150.0
    law = MagnetLaw2D(magnet, np.array([True, False]), 2, T=T, axis=(0.0, 1.0))
    b_par = 0.2                                                    # Тл, глубоко за коленом
    b_perp = 0.05
    H, D = law(np.array([[b_perp, b_par], [0.0, 0.0]]))
    h_exp, s_exp = (float(x[0]) for x in magnet.branch_parallel_inverse(np.array([b_par]), T, np.ones(1)))
    assert H[0, 1] == pytest.approx(MU0 * h_exp, rel=1e-12)          # вдоль оси — ветвь
    assert H[0, 0] == pytest.approx(b_perp / magnet.mu_perp, rel=1e-12)     # поперёк — μ⊥
    assert D[0, 1, 1] == pytest.approx(MU0 / s_exp, rel=1e-12)
    assert D[0, 0, 0] == pytest.approx(1.0 / magnet.mu_perp, rel=1e-12)
    assert D[0, 0, 1] == pytest.approx(0.0, abs=1e-15)
    assert float(law.retention_now()[0]) < 1.0                             # за коленом доля меньше единицы


@pytest.mark.parametrize("bad", [
    {"axis": (0.0, 0.0)},                       # нулевая ось
    {"axis": (1.0, 0.0, 0.0)},                  # три числа в планарной задаче
    {"retention": np.array([1.4, 1.0])},        # доля вне [0, 1]
])
def test_planar_law_refuses_bad_input(bad):
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    kwargs = {"T": 20.0, "axis": (0.0, 1.0)}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        MagnetLaw2D(magnet, np.array([True, False]), 2, **kwargs)


def test_planar_law_refuses_mask_of_wrong_length():
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        MagnetLaw2D(magnet, np.array([True, False, True]), 2, T=20.0, axis=(0.0, 1.0))


# ------------------------------------------------ поячеечная температура (транзиент К6′)

def _cells_through_both_branches(magnet, n=9):
    """Поле, доли и температуры ячеек: за коленом, на возврате, ниже −H_cJ, подмагничивание; T — от 20 до 175 °C."""
    T = np.array([20.0, 60.0, 120.0, 150.0, 150.0, 162.7, 175.3, 175.3, 101.1])
    r = np.array([1.0, 0.9, 1.0, 0.7, 1.0, 0.5, 0.3, 1.0, 0.95])
    B = np.random.default_rng(1).uniform(-0.7, 1.4, (n, 2))
    return T, r, B


def test_planar_law_with_cell_temperatures_is_the_one_temperature_law_cell_by_cell():
    """Поячеечная T: каждая ячейка подчиняется ровно тому закону, что весь магнит при её температуре (бит в бит)."""
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    T, r, B = _cells_through_both_branches(magnet)
    n = T.size
    law = MagnetLaw2D(magnet, np.ones(n, dtype=bool), n, T=T, axis=(0.0, 1.0), retention=r)
    H, D = law(B)
    r_now, beyond = law.retention_now(), law.beyond_hcj()
    for c in range(n):
        one = MagnetLaw2D(magnet, np.array([True]), 1, T=float(T[c]), axis=(0.0, 1.0), retention=r[c:c + 1])
        Hc, Dc = one(B[c:c + 1])
        assert np.array_equal(H[c], Hc[0]) and np.array_equal(D[c], Dc[0])
        assert r_now[c] == one.retention_now()[0] and beyond[c] == one.beyond_hcj()[0]
    # предпосылки: обе ветви, новая потеря и ячейка ниже −H_cJ в выборке есть
    assert np.any(r_now < r) and np.any(r_now == r) and beyond.any()


def test_uniform_cell_temperature_is_the_scalar_temperature_bit_for_bit():
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    T, r, B = _cells_through_both_branches(magnet)
    n = T.size
    kw = dict(axis=(0.0, 1.0), retention=r)
    scalar = MagnetLaw2D(magnet, np.ones(n, dtype=bool), n, T=150.0, **kw)
    cells = MagnetLaw2D(magnet, np.ones(n, dtype=bool), n, T=np.full(n, 150.0), **kw)
    for a, b in zip(scalar(B), cells(B)):
        assert np.array_equal(a, b)
    assert np.array_equal(scalar.retention_now(), cells.retention_now())


def test_temperature_grid_takes_the_nearest_node_and_bounds_the_error():
    """
    T_bin: кривая берётся в ближайшем узле сетки температур — как в ядре К6′. На линии возврата H(B) = (B − B_r(T))/
    (μ₀μ_rec) линейна по B_r, поэтому ошибка закона от сетки — ровно |dB_r/dT|·|T − T_узла| ≤ |dB_r/dT|·T_bin/2.
    """
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    mask = np.array([True])
    B = np.array([[0.0, 1.0]])                                    # линия возврата целого магнита
    T_bin = 0.5
    for T in (150.2, 150.3, 150.25 + 1e-9):
        node = round(T / T_bin) * T_bin
        binned = MagnetLaw2D(magnet, mask, 1, T=np.array([T]), axis=(0.0, 1.0), T_bin=T_bin)
        at_node = MagnetLaw2D(magnet, mask, 1, T=node, axis=(0.0, 1.0))
        assert np.array_equal(binned(B)[0], at_node(B)[0])
        exact = MagnetLaw2D(magnet, mask, 1, T=T, axis=(0.0, 1.0))
        dBr = magnet.mu_rec * abs(binned(B)[0][0, 1] - exact(B)[0][0, 1])   # H решателя = (B − B_r)/μ_rec
        dBr_dT = magnet.alpha_Br / 100.0 * magnet.Br0
        assert dBr == pytest.approx(dBr_dT * abs(T - node), rel=1e-6)
        assert dBr <= dBr_dT * T_bin / 2 * (1 + 1e-9)


def test_cell_temperature_and_grid_are_validated():
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    mask = np.array([True, False])
    for bad in ({"T": np.array([20.0, 30.0, 40.0])}, {"T": np.array([np.nan, 20.0])},
                {"T": 20.0, "T_bin": 0.0}, {"T": 20.0, "T_bin": -0.5}):
        with pytest.raises(ValueError):
            MagnetLaw2D(magnet, mask, 2, axis=(0.0, 1.0), **bad)
def test_cell_without_a_magnetization_direction_behaves_as_linear_recoil():
    """Поячеечная ось бывает не определена (радиальное намагничивание в центре): там нет намагниченности,
    тело ведёт себя как линейное с μ_rec — как и в прежней схеме, без исключения."""
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    axis = np.array([[0.0, 1.0], [0.0, 0.0]])                  # вторая ячейка — ось не определена
    law = MagnetLaw2D(magnet, np.array([True, True]), 2, T=20.0, axis=axis)
    B = np.array([[0.0, 0.5], [0.3, 0.4]])
    H, D = law(B)
    assert np.allclose(H[1], B[1] / magnet.mu_rec, rtol=1e-12)
    assert np.allclose(D[1], np.eye(2) / magnet.mu_rec, rtol=1e-12)
    assert H[0, 1] != pytest.approx(B[0, 1] / magnet.mu_rec)   # у ячейки с осью — закон магнита
    # Намагниченности у такой ячейки нет — и терять нечего: доля r прежняя, «ниже −H_cJ» нет. При 150 °C
    # формальное H∥ ячейки (−H_cB при B∥ = 0) лежит за коленом, и без этой оговорки защёлка записала бы потерю.
    hot = MagnetLaw2D(magnet, np.array([True, True]), 2, T=150.0, axis=axis, retention=np.array([1.0, 0.8]))
    hot(B)
    assert float(magnet.retention_now(hot.H_par[1], 150.0)) < 1.0      # предпосылка: было бы «повреждение»
    assert hot.retention_now()[1] == 0.8 and not hot.beyond_hcj()[1]
