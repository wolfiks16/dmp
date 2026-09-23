import numpy as np
import pytest

pytest.importorskip("gmsh")

from magcore.domain.magnet_model import n42sh_magnet  # noqa: E402
from magcore.domain.steel_curves import m270_35a_bh_curve  # noqa: E402
from magcore.fem2d.machines import (  # noqa: E402
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
    evaluate_demag_impact,
    solve_machine_static,
    star_of_slots_layout,
)

# МАГНИТ ЗА КОЛЕНОМ В ДВИГАТЕЛЕ (этап 2 к Л-107). Статический расчёт машины, расчёт падения ЭДС после
# размагничивания и магнитотепловой расчёт считали колено внешним циклом по источнику магнита с
# релаксацией вне касательной. Теперь все они решают общую задачу 2D, где магнит — закон ветви в
# касательной Ньютона. Двигатель: outrunner 12N14P, N42SH, M270, d-ток 30 А (γ = π, размагничивает),
# 40 витков/паз, сетка 4 мм. При 160 °C прежняя схема с ω = 0,1 не сходилась за 300 итераций.
# Точного решения у двигателя нет, поэтому эталон — та же прежняя схема, доведённая до невязки 1e-10
# там, где она сходится (140 °C при ω = 0,1; 160 °C при ω = 0,03 — граница устойчивости цикла зависит
# от ω, Л-93). Это независимая реализация закона магнита (источник ν·B_r,eff вместо обращения ветви).

IPK, GAMMA_D, NPS = 30.0, np.pi, 40.0


@pytest.fixture(scope="module")
def machine():
    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.004))
    layout = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    return g, n42sh_magnet(easy_axis=(1, 0, 0)), m270_35a_bh_curve(), layout


def _solve(machine, T, **kw):
    g, magnet, steel, layout = machine
    return solve_machine_static(g, magnet, steel, T=T, layout=layout, i_peak=IPK, gamma_elec=GAMMA_D,
                                turns_per_slot=NPS, **kw)


def test_same_field_as_the_previous_scheme_where_that_one_converges(machine):
    """140 °C (часть магнита за коленом): оба метода доведены до невязки 1e-10 — одно и то же поле."""
    new = _solve(machine, 140.0, tol=1e-10, max_iter=60)
    old = _solve(machine, 140.0, method="picard", relaxation=0.1, tol=1e-10, max_iter=2000)
    assert new.converged and old.converged
    assert new.n_iterations <= 12 < old.n_iterations                 # измерено: 9 против 194
    mag = machine[0].region == int(Region.MAGNET)
    d = np.abs(new.B_cells - old.B_cells)
    assert d[mag].max() < 1e-9 and d.max() < 5e-8                    # измерено 4e-11 и 7e-9 Тл
    assert new.risk.n_demagnetized == old.risk.n_demagnetized > 0
    assert new.risk.worst_margin == pytest.approx(old.risk.worst_margin, abs=1e-3)   # А/м


def test_past_the_knee_converges_and_matches_the_previous_scheme_with_small_relaxation(machine):
    """160 °C: с ω = 0,1 прежняя схема не сходилась; с ω = 0,03 сходится — к решению Ньютона."""
    new = _solve(machine, 160.0, tol=1e-10, max_iter=60)
    assert new.converged and new.n_iterations <= 12                  # измерено: 8
    ref = _solve(machine, 160.0, method="picard", relaxation=0.03, tol=1e-10, max_iter=4000)
    assert ref.converged                                             # измерено: 610 итераций
    mag = machine[0].region == int(Region.MAGNET)
    d = np.abs(new.B_cells - ref.B_cells)
    assert d[mag].max() < 1e-8 and d.max() < 1e-7                    # измерено 2e-10 и 1,5e-8 Тл
    assert new.risk.n_demagnetized == ref.risk.n_demagnetized
    assert new.risk.n_demagnetized > mag.sum() // 2                  # больше половины ячеек за коленом


def test_motor_solution_lies_on_the_branch_law_in_every_magnet_cell(machine):
    """Индукция вдоль радиальной оси каждой ячейки магнита лежит на ветви, по которой строилась касательная."""
    g, magnet, _, _ = machine
    r = _solve(machine, 160.0, max_iter=60)
    idx = r.risk.cell_indices
    b_par = np.einsum("ij,ij->i", r.B_cells[idx], g.magnet_easy_axis[idx])
    b_law, _ = magnet.branch_parallel(r.risk.H_par, 160.0, np.ones(idx.size))
    assert np.abs(b_law - b_par).max() < 1e-12


def test_track_worst_point_is_refused_for_the_motor_in_newton(machine):
    """История по итерациям — только в прежней схеме; в Ньютоне — явная ошибка, а не молчаливый пропуск."""
    with pytest.raises(ValueError, match="retention"):
        _solve(machine, 140.0, track_worst_point=True)


def test_demag_impact_is_unchanged_where_the_previous_scheme_converged(machine):
    """Падение ЭДС после размагничивания при 140 °C: Ньютон и прежняя схема — одно и то же."""
    g, magnet, steel, layout = machine
    kw = dict(i_peak=IPK, gamma_elec=GAMMA_D, turns_per_slot=NPS, T=140.0)
    new = evaluate_demag_impact(g, magnet, steel, layout, **kw)
    old = evaluate_demag_impact(g, magnet, steel, layout, method="picard", **kw)
    assert new.load_converged and old.load_converged
    assert new.flux_linkage_drop_frac > 0.0
    assert new.flux_linkage_drop_frac == pytest.approx(old.flux_linkage_drop_frac, abs=1e-6)   # измерено 8e-8
    assert new.aggregate.mean_loss_frac == pytest.approx(old.aggregate.mean_loss_frac, abs=1e-6)
    assert new.risk.n_demagnetized == old.risk.n_demagnetized


def test_demag_impact_past_the_knee_converges_and_grows_with_temperature(machine):
    """При 160 °C нагрузка теперь сходится (прежде — нет), и потеря больше, чем при 140 °C."""
    g, magnet, steel, layout = machine
    kw = dict(i_peak=IPK, gamma_elec=GAMMA_D, turns_per_slot=NPS)
    hot = evaluate_demag_impact(g, magnet, steel, layout, T=160.0, **kw)
    warm = evaluate_demag_impact(g, magnet, steel, layout, T=140.0, **kw)
    assert hot.load_converged
    assert hot.flux_linkage_drop_frac > warm.flux_linkage_drop_frac > 0.0
    assert hot.aggregate.demag_area_fraction > warm.aggregate.demag_area_fraction
    assert hot.aggregate.max_loss_frac >= hot.aggregate.mean_loss_frac > 0.0
