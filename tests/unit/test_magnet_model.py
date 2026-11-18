from __future__ import annotations

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_model import (
    AnisotropicBHTMagnet,
    excel_reference_magnet,
    magnet_from_datasheet,
    n42sh_magnet,
)


EZ = np.array([0.0, 0.0, 1.0])


def test_easy_axis_normalized_and_validation() -> None:
    m = magnet_from_datasheet("m", "m", np.array([0.0, 0.0, 2.0]), 1.2, 9.0e5, 1.0e6, 1.8e6)
    assert m.easy_axis == pytest.approx(EZ)
    with pytest.raises(ValueError):  # Hk >= Hcj
        magnet_from_datasheet("x", "x", EZ, 1.2, 9.0e5, 1.8e6, 1.0e6)
    with pytest.raises(ValueError):  # zero axis
        magnet_from_datasheet("x", "x", np.zeros(3), 1.2, 9.0e5, 1.0e6, 1.8e6)
    with pytest.raises(ValueError):  # Br <= 0
        AnisotropicBHTMagnet("x", "x", EZ, -1.0, 9.0e5, 1.0e6, 1.8e6, mu_perp=1.05)


@pytest.mark.parametrize("factory", ["n42sh_magnet", "sm2co17_magnet", "n35_magnet", "ks25dts240_magnet",
                                     "excel_reference_magnet"])
def test_magnet_temperature_sign_convention(factory) -> None:
    # Конвенция знака (docstring AnisotropicBHTMagnet): в коде коэффициенты — МОДУЛИ падения в %/°C, минус в
    # формуле; в постановке (coupled_problem.md §3, nonlinear_materials.md §4) — знаковые α, β [1/K] < 0. Проверка
    # по физике, а не по записи: у каждого реального пресета B_r, H_cB, H_k, H_cJ строго УБЫВАЮТ с ростом T, и
    # значения совпадают с формой документа B_r0·[1 + α(T − T0)] при α = −alpha_Br/100 (β = −gamma_Hc/100).
    import magcore.domain.magnet_model as mm

    m = getattr(mm, factory)(EZ)
    assert m.alpha_Br > 0.0 and m.gamma_Hc > 0.0
    alpha_doc, beta_doc = -m.alpha_Br / 100.0, -m.gamma_Hc / 100.0
    T = np.linspace(m.T0, m.temperature_limit() - 1.0, 9)
    Br = np.array([m.Br(t) for t in T])
    Hcj = np.array([m.Hcj(t) for t in T])
    assert np.all(np.diff(Br) < 0.0) and np.all(np.diff(Hcj) < 0.0)
    assert np.all(np.diff([m.Hcb(t) for t in T]) < 0.0) and np.all(np.diff([m.Hk(t) for t in T]) < 0.0)
    assert Br == pytest.approx(m.Br0 * (1.0 + alpha_doc * (T - m.T0)), rel=1e-12)
    assert Hcj == pytest.approx(m.Hcj0 * (1.0 + beta_doc * (T - m.T0)), rel=1e-12)


def test_negative_temperature_coefficient_is_rejected() -> None:
    # Значение даташита «−0,12 %/°C», введённое со знаком, дало бы РОСТ B_r с нагревом — молча неверная физика.
    for kw in (dict(alpha_Br=-0.12), dict(gamma_Hc=-0.6)):
        with pytest.raises(ValueError, match="МОДУЛЕМ"):
            magnet_from_datasheet("x", "x", EZ, 1.2, 9.0e5, 1.0e6, 1.8e6, **kw)
    m0 = magnet_from_datasheet("x", "x", EZ, 1.2, 9.0e5, 1.0e6, 1.8e6, alpha_Br=0.0, gamma_Hc=0.0)
    assert m0.Br(150.0) == m0.Br0 and m0.temperature_limit() == float("inf")   # ноль — без T-зависимости


def test_temperature_validity_has_three_levels() -> None:
    # Предел модели (линейная H_cJ(T) → 0) — не граница данных: у N42SH это ≈ 201,8 °C при данных марки SH обычно
    # до ~150 °C. С границей данных статус трёхуровневый; без неё (по умолчанию) — прежнее поведение.
    import dataclasses

    from magcore.domain.magnet_model import VALIDITY_DATA, VALIDITY_EXTRAPOLATION, VALIDITY_OUT_OF_MODEL

    old = n42sh_magnet(EZ)
    limit = old.temperature_limit()
    assert limit == pytest.approx(20.0 + 100.0 / 0.55)
    assert old.T_data_max is None and old.data_temperature_limit() == limit
    assert old.temperature_validity(180.0) == VALIDITY_DATA                  # граница не задана — как было
    assert old.temperature_validity(limit) == VALIDITY_OUT_OF_MODEL
    m = dataclasses.replace(old, T_data_max=150.0)
    assert m.law_key() == old.law_key()                                      # закон тот же — это метаданные
    assert m.data_temperature_limit() == 150.0
    assert m.temperature_validity(150.0) == VALIDITY_DATA
    assert m.temperature_validity(150.5) == VALIDITY_EXTRAPOLATION
    assert m.temperature_validity(201.0) == VALIDITY_EXTRAPOLATION
    assert m.temperature_validity(limit + 0.1) == VALIDITY_OUT_OF_MODEL
    assert m.temperature_validity(float("nan")) == VALIDITY_OUT_OF_MODEL
    assert m.Br(180.0) == old.Br(180.0)                                      # на сам расчёт не влияет
    for bad in (10.0, 20.0, limit, float("inf")):                            # ниже T0, на пределе модели и выше
        with pytest.raises(ValueError, match="T_data_max"):
            dataclasses.replace(old, T_data_max=bad)


def test_mu_rec_derived_from_Br_Hcb() -> None:
    m = excel_reference_magnet(EZ)
    assert m.mu_rec == pytest.approx(m.Br0 / (MU0 * m.Hcb0))
    assert 1.0 < m.mu_rec < 1.2  # NdFeB recoil


def test_remanence_temperature_matches_spreadsheet() -> None:
    m = excel_reference_magnet(EZ)
    assert m.Br(m.T0) == pytest.approx(m.Br0)
    # b(180) = 1 - 0.12*160/100 = 0.808 ; Br(180) ~ 8980 Гс = 0.8980 Тл (как в шаблоне)
    assert m.Br(180.0) == pytest.approx(0.808 * m.Br0, rel=1e-9)
    assert m.Br(180.0) == pytest.approx(8980.0 * 1e-4, rel=2e-3)


def test_knee_scales_with_coercivity_coefficient() -> None:
    m = excel_reference_magnet(EZ)
    assert m.knee_field(m.T0) == pytest.approx(-m.Hk0)
    # h(180) = 1 - 0.465*160/100 = 0.256
    assert m.knee_field(180.0) == pytest.approx(-0.256 * m.Hk0, rel=1e-9)
    assert m.knee_field(180.0) > m.knee_field(m.T0)  # ближе к нулю


def test_recoil_slope_preserved_across_temperature() -> None:
    # КЛЮЧЕВОЙ фикс: наклон recoil-прямой = mu0*mu_rec при ЛЮБОЙ T
    # (равномерное масштабирование кривой исказило бы его в b/h раз).
    m = excel_reference_magnet(EZ)
    expected = MU0 * m.mu_rec

    def recoil_slope(T):
        c = m.curve_at(T)
        return (c.B_of_H(-1.0e5) - c.B_of_H(-2.0e5)) / (1.0e5)  # оба выше колена

    assert recoil_slope(m.T0) == pytest.approx(expected, rel=1e-3)
    assert recoil_slope(180.0) == pytest.approx(expected, rel=1e-3)
    assert recoil_slope(180.0) == pytest.approx(recoil_slope(m.T0), rel=1e-3)


def test_nu_tensor_eigenstructure_anisotropic() -> None:
    m = magnet_from_datasheet("a", "a", EZ, 1.2, 9.0e5, 1.0e6, 1.8e6, mu_perp=1.30)
    N = m.nu_tensor()
    assert np.allclose(N, N.T)
    assert N @ m.easy_axis == pytest.approx(m.nu_parallel * m.easy_axis)
    w = np.array([1.0, 0.0, 0.0])
    assert N @ w == pytest.approx(m.nu_perp * w)
    eig = np.sort(np.linalg.eigvalsh(N))
    assert eig == pytest.approx(np.sort([m.nu_parallel, m.nu_perp, m.nu_perp]))


def test_isotropic_tensor_when_mu_perp_equals_mu_rec() -> None:
    m = excel_reference_magnet(EZ)  # mu_perp == mu_rec по умолчанию
    assert np.allclose(m.nu_tensor(), m.nu_parallel * np.eye(3))


def test_no_loss_above_knee() -> None:
    m = excel_reference_magnet(EZ)
    H_above = -0.5e6  # выше колена T0 (-1.24e6)
    assert float(m.irreversible_loss(H_above, m.T0)) == pytest.approx(0.0, abs=1e-6)
    assert float(m.effective_Br(H_above, m.T0)) == pytest.approx(m.Br(m.T0), abs=1e-6)


def test_partial_loss_below_knee_is_positive_and_bounded() -> None:
    m = excel_reference_magnet(EZ)
    T = 180.0
    # -0.45e6 лежит между коленом(180)=-0.318e6 и -HcJ(180)=-0.579e6 => частичная потеря
    H_min = -0.45e6
    Br_eff = float(m.effective_Br(H_min, T))
    assert 0.0 < Br_eff < float(m.Br(T))
    assert float(m.irreversible_loss(H_min, T)) > 0.0


def test_temperature_turns_safe_point_into_demag() -> None:
    m = excel_reference_magnet(EZ)
    H_min = -0.45e6
    assert float(m.irreversible_loss(H_min, m.T0)) == pytest.approx(0.0, abs=1e-6)  # T0 безопасно
    assert float(m.irreversible_loss(H_min, 180.0)) > 0.0                            # горячо — потеря


def test_effective_Br_matches_generic_formula() -> None:
    m = excel_reference_magnet(EZ)
    for (H, T) in [(-0.3e6, 20.0), (-0.45e6, 180.0), (-0.1e6, 120.0)]:
        lhs = float(m.effective_Br(H, T))
        rhs = float(m.B_major_parallel(H, T)) - MU0 * m.mu_rec * H
        assert lhs == pytest.approx(rhs, rel=1e-9)


def test_parallel_field_round_trip() -> None:
    m = excel_reference_magnet(EZ)
    T, H, H_min = 20.0, -3.0e5, 0.0  # без истории => Br_eff = Br(T0)
    B_par = float(m.effective_Br(H_min, T)) + MU0 * m.mu_rec * H
    B_vec = B_par * m.easy_axis
    assert float(m.parallel_field(B_vec, T, H_min)) == pytest.approx(H, rel=1e-6)


def test_risk_margin_sign() -> None:
    m = excel_reference_magnet(EZ)
    T = 180.0
    hk = float(m.knee_field(T))
    assert float(m.risk_margin(hk + 5.0e4, T)) > 0.0
    assert float(m.risk_margin(hk - 5.0e4, T)) < 0.0


def test_vectorized_over_cells() -> None:
    m = excel_reference_magnet(EZ)
    T = 180.0
    H_min = np.array([-0.1e6, -0.40e6, -0.50e6])  # выше колена / частично / глубже
    loss = m.irreversible_loss(H_min, T)
    assert loss.shape == (3,)
    assert np.all(np.diff(loss) >= 0.0)  # глубже => не меньше
    Br_eff = m.effective_Br(H_min, T)
    assert np.all(Br_eff <= m.Br(T) + 1e-9)


def test_remanence_vector_direction() -> None:
    m = excel_reference_magnet(EZ)
    v = m.effective_remanence_vector(0.0, m.T0)
    assert v == pytest.approx(m.Br0 * EZ)


def test_n42sh_factory_constructs() -> None:
    m = n42sh_magnet(EZ)
    assert 1.0 < m.mu_rec < 1.2
    assert m.Hk0 < m.Hcj0
    assert m.Br(m.T0) == pytest.approx(m.Br0)
