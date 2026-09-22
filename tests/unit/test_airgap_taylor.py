import math

from magcore.fem2d.airgap import (
    K_AIR_DEFAULT,
    airgap_k_eff,
    airgap_nusselt,
    taylor_critical_speed,
    taylor_number,
)

# P-B2: k_eff(n) зазора по Тейлору–Куэтту (Howey 2012). Оракулы:
#  статика → кондукция (k_eff=k_air); монотонность по оборотам; сверка с Howey при 7000;
#  k_eff ≪ прежнего ad-hoc 0.5; порог вихрей ~1700-1900 об/мин; непрерывность корреляций.

R_M, GAP = 0.0225, 0.001      # представительный зазор outrunner (r_m, g)


def test_static_recovers_air_conduction():
    # n=0: Ta=0 → Nu=2 → k_eff=k_air (чистая кондукция).
    assert abs(airgap_k_eff(0.0, R_M, GAP) - K_AIR_DEFAULT) < 1e-12


def test_monotonic_increase_with_speed():
    ke = [airgap_k_eff(n, R_M, GAP) for n in (2000, 4000, 7000, 10000)]
    assert all(b > a for a, b in zip(ke, ke[1:]))
    assert airgap_k_eff(7000, R_M, GAP) > airgap_k_eff(0.0, R_M, GAP)


def test_matches_howey_at_7000():
    Ta = taylor_number(7000, R_M, GAP)
    assert 150.0 < Ta < 185.0                      # ≈166 (P-B0)
    assert 4.0 < airgap_nusselt(Ta) < 5.5          # ≈4.7
    assert 0.05 < airgap_k_eff(7000, R_M, GAP) < 0.10   # ≈0.065–0.071


def test_far_below_adhoc_half():
    # Ключевая поправка B-full: реальный k_eff ≪ прежнего 0.5 ⇒ зазор изолирует магнит сильнее.
    assert airgap_k_eff(7000, R_M, GAP) < 0.2
    assert airgap_k_eff(0.0, R_M, GAP) < 0.2


def test_critical_speed_onset():
    n_cr = taylor_critical_speed(R_M, GAP)
    assert 1500.0 < n_cr < 2100.0                  # ≈1800 об/мин
    assert abs(airgap_nusselt(taylor_number(n_cr * 0.9, R_M, GAP)) - 2.0) < 1e-9  # ниже — кондукция
    assert airgap_nusselt(taylor_number(n_cr * 1.5, R_M, GAP)) > 2.0              # выше — вихри


def test_nusselt_continuous_at_regime_boundaries():
    F_g = 1.04
    for x_b in (1697.0, 1.0e4):
        Ta_b = F_g * math.sqrt(x_b)                # (Ta/F_g)² = x_b
        lo = airgap_nusselt(Ta_b * (1 - 1e-5), F_g)
        hi = airgap_nusselt(Ta_b * (1 + 1e-5), F_g)
        assert abs(lo - hi) < 0.06
