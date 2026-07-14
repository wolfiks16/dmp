import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.hybrid.magnet_demag import MagnetDemagPolicy

# P4: связанный статический решатель на реальной геометрии машины (магнит-демаг + сталь +
# ток обмотки). Оракулы: сходимость, физичность зазорного поля, число полюсов (доминирующая
# угловая гармоника = p), монотонность демаг-маржи по току и по T, начало необратимого демага.

MESH = 0.004      # грубая сетка (плотный решатель) — компромисс скорость/физика
IPK, GAMMA_D, NPS = 30.0, np.pi, 40.0   # d-осевой демаг-ток: i_peak, γ, витков/паз


# --- быстрый тест обобщения политики демага на поячеечную ось (без gmsh) ---

def test_demag_policy_percell_axis_matches_single_axis():
    magnet = n42sh_magnet(easy_axis=(1, 0, 0))
    nc = 6
    mask = np.array([False, True, True, False, True, False])
    H = np.array([[0.1, -0.2], [-0.3, 0.4], [0.5, 0.1], [0.0, 0.0], [-0.6, 0.2], [0.2, 0.2]])
    nu = np.full(nc, 1.0 / magnet.mu_rec)

    single = MagnetDemagPolicy(magnet, mask, T=25.0, n_cells=nc, axis=(1.0, 0.0))
    axes = np.zeros((nc, 2)); axes[mask] = (1.0, 0.0)
    percell = MagnetDemagPolicy(magnet, mask, T=25.0, n_cells=nc, axis=axes)
    # Поячеечная ось = та же (1,0) везде ⇒ идентичный источник (обратная совместимость).
    assert np.allclose(single(H, H, nu), percell(H, H, nu))

    # Другая поячеечная ось (0,1) ⇒ иной источник (ось реально используется построчно).
    axes2 = np.zeros((nc, 2)); axes2[mask] = (0.0, 1.0)
    percell2 = MagnetDemagPolicy(magnet, mask, T=25.0, n_cells=nc, axis=axes2)
    assert not np.allclose(single(H, H, nu), percell2(H, H, nu))


def test_demag_policy_percell_axis_shape_check():
    magnet = n42sh_magnet(easy_axis=(1, 0, 0))
    with pytest.raises(ValueError):
        MagnetDemagPolicy(magnet, np.array([True, False]), T=20.0, n_cells=2,
                          axis=np.zeros((3, 2)))   # (n_cells,dim) с неверным n_cells


# --- интеграционные тесты на сетке машины (gmsh) ---

@pytest.fixture(scope="module")
def machine():
    pytest.importorskip("gmsh")
    from magcore.domain.steel_curves import m270_35a_bh_curve
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        star_of_slots_layout,
    )
    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=MESH))
    magnet = n42sh_magnet(easy_axis=(1, 0, 0))
    steel = m270_35a_bh_curve()
    layout = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    return g, magnet, steel, layout


@pytest.fixture(scope="module")
def s1_magnet_only(machine):
    from magcore.fem2d.machines import solve_machine_static
    g, magnet, steel, _ = machine
    return g, solve_machine_static(g, magnet, steel, T=20.0, max_iter=300)


@pytest.fixture(scope="module")
def s1_current(machine):
    from magcore.fem2d.machines import solve_machine_static
    g, magnet, steel, layout = machine
    return solve_machine_static(g, magnet, steel, T=20.0, layout=layout,
                                i_peak=IPK, gamma_elec=GAMMA_D, turns_per_slot=NPS, max_iter=300)


@pytest.fixture(scope="module")
def s3_hot(machine):
    from magcore.fem2d.machines import solve_machine_static
    g, magnet, steel, layout = machine
    return solve_machine_static(g, magnet, steel, T=140.0, layout=layout,
                                i_peak=IPK, gamma_elec=GAMMA_D, turns_per_slot=NPS, max_iter=300)


def test_reluctivity_convention(machine):
    # Относительная ν при B=0: воздух=1, магнит=1/μ_rec, сталь высокопроницаема (ν мало).
    from magcore.fem2d.machines import Region
    from magcore.fem2d.machines.static_solver import machine_reluctivity
    g, magnet, steel, _ = machine
    _, nu0, magnet_mask, steel_mask = machine_reluctivity(g, magnet, steel)
    air = g.region == int(Region.AIR_GAP)
    assert np.allclose(nu0[air], 1.0)
    assert np.allclose(nu0[magnet_mask], 1.0 / magnet.mu_rec)
    assert np.all(nu0[steel_mask] < 1.0e-2)              # μ_r стали ~ тысячи ⇒ ν_rel мало
    assert np.all(nu0[steel_mask] > 0.0)


def test_s1_magnet_only_converges_and_physical(s1_magnet_only):
    from magcore.fem2d.machines import Region
    g, r = s1_magnet_only
    assert r.converged
    gap = g.region == int(Region.AIR_GAP)
    Bmag = np.hypot(r.B_cells[gap, 0], r.B_cells[gap, 1])
    assert 0.3 < Bmag.mean() < 1.2                       # физичное зазорное поле PMSM
    assert Bmag.max() < 2.5
    # Здоровый магнит под собственным полем при 20 °C — без необратимой потери.
    assert r.risk.n_demagnetized == 0
    assert r.risk.total_loss == 0.0


def test_s1_air_gap_pole_harmonic(s1_magnet_only):
    # Доминирующая угловая гармоника радиального зазорного поля = p (пары полюсов).
    from magcore.fem2d.machines import Region
    g, r = s1_magnet_only
    p = g.params.n_poles // 2
    gap = np.where(g.region == int(Region.AIR_GAP))[0]
    cen = np.array([g.mesh.cell_centroid(int(c)) for c in gap])
    th = np.arctan2(cen[:, 1], cen[:, 0]) % (2.0 * np.pi)
    rhat = np.stack([np.cos(th), np.sin(th)], axis=1)
    Br = np.einsum("ij,ij->i", r.B_cells[gap], rhat)
    nb = 72
    bins = (th / (2.0 * np.pi) * nb).astype(int)
    prof = np.array([Br[bins == b].mean() if np.any(bins == b) else 0.0 for b in range(nb)])
    spec = np.abs(np.fft.rfft(prof))
    assert int(np.argmax(spec[1:]) + 1) == p


def test_demag_current_reduces_margin(s1_magnet_only, s1_current):
    # d-осевой ток якоря понижает маржу к колену (реакция якоря размагничивает).
    _, r0 = s1_magnet_only
    assert s1_current.converged
    assert s1_current.risk.worst_margin < r0.risk.worst_margin
    assert s1_current.gamma_elec is not None


def test_temperature_pushes_past_knee(s1_current, s3_hot):
    # Нагрев при том же токе: маржа падает монотонно; горячий заходит за колено (необратимо).
    assert s3_hot.converged
    assert s3_hot.risk.worst_margin < s1_current.risk.worst_margin
    assert s1_current.risk.n_demagnetized == 0           # холодный цел
    assert s3_hot.risk.n_demagnetized > 0                # горячий частично за коленом
    assert s3_hot.risk.total_loss > 0.0


def test_worst_case_gamma_sweep(machine):
    from magcore.fem2d.machines import worst_case_gamma_sweep
    g, magnet, steel, layout = machine
    best, gammas, margins = worst_case_gamma_sweep(
        g, magnet, steel, layout, i_peak=IPK, turns_per_slot=NPS, T=20.0, n_angles=4,
    )
    assert margins.shape == (4,)
    # Возвращается именно ХУДШИЙ (минимальная маржа) угол.
    assert np.isclose(best.risk.worst_margin, margins.min())
