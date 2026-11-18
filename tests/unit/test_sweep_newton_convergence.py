import math

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.exceptions import ConvergenceError
from magcore.fem2d.machines.catalog import outrunner_pmsm_uav
from magcore.fem2d.machines.iron_loss import stator_iron_loss, stator_iron_loss_density
from magcore.fem2d.machines.library import solve_machine
from magcore.fem2d.machines.magnet_loss import rotor_frame_B_series, rotor_side_cells
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, build_outrunner_spm_pmsm
from magcore.fem2d.machines.rotor_sweep import _solve_at_angle, require_converged
from magcore.fem2d.machines.winding import star_of_slots_layout
from magcore.fem2d.machines.excitation import winding_current_density

# П. 9 плана 2026-11-18: свипы потерь (статор, ротор) и библиотечная магнитостатика (`solve_machine`, путь
# сверки с ДП25) переведены с хордового Пикара без проверки сходимости на Ньютона по стали; флаг сходимости
# теперь читается, и несошедшееся приближение не уходит дальше как «волна B».
# ОРАКУЛЫ: (1) предел Пикара — решение Ньютона: ужесточение допуска Пикара в 10³ раз уменьшает расхождение
# пропорционально (сошёлся бы он к другой точке — расхождение упёрлось бы в разницу); (2) Ньютон сходится по
# НЕВЯЗКЕ за единицы итераций; (3) НЕГАТИВНЫЕ: свип с несошедшимся положением — ConvergenceError, а не число.

PARAMS = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.005)
MAGNET = n42sh_magnet((1.0, 0.0, 0.0))
STEEL = m270_35a_bh_curve()


def _loaded_current(geo, i_peak=60.0):
    lay = star_of_slots_layout(PARAMS.n_slots, PARAMS.n_poles)
    return winding_current_density(geo, lay, i_peak=i_peak, gamma_elec=math.pi / 2, turns_per_slot=20.0)


def test_newton_at_angle_is_the_limit_of_picard():
    # Эталон-Пикар — с ω = 0,1: под нагрузкой при ω = 0,2 (так шли свипы сверки IM-8008) он здесь не сходится и
    # за 3000 итераций (предельный цикл, расхождение с решением по B до 17 %) — оракулом служить не может.
    geo = build_outrunner_spm_pmsm(PARAMS)
    jz = _loaded_current(geo, i_peak=40.0)
    kw = dict(T=20.0, retention=None, j_cells=jz)
    new = _solve_at_angle(geo, MAGNET, STEEL, method="newton", **kw)
    hist = np.asarray(new.rel_change_history)
    assert new.converged and new.n_iterations <= 12 and hist[-1] <= 1e-9 * hist[0]
    errs = []
    for tol in (1e-5, 1e-8):
        pic = _solve_at_angle(geo, MAGNET, STEEL, method="picard", relaxation=0.1, max_iter=5000, tol=tol, **kw)
        assert pic.converged
        errs.append(np.linalg.norm(new.B_cells - pic.B_cells) / np.linalg.norm(new.B_cells))
    assert errs[1] < 1e-2 * errs[0], errs
    with pytest.raises(ValueError, match="method"):
        _solve_at_angle(geo, MAGNET, STEEL, method="что-то", **kw)


def test_require_converged_names_the_failed_positions():
    require_converged([True, True], [0.0, 0.1], "x")                    # всё сошлось — молча
    with pytest.raises(ConvergenceError, match=r"1 из 3 .*5\.73"):
        require_converged([True, False, True], [0.0, 0.1, 0.2], "свип")


def test_loss_sweeps_refuse_unconverged_fields():
    # Бюджет в две итерации эталону-Пикару: поле не решено ни на одном положении — раньше карта потерь из такого
    # приближения уходила в тепловой источник молча (флаг не читался), теперь — отказ.
    kw = dict(speed_rpm=7000.0, i_peak=40.0, gamma_elec=math.pi / 2, turns_per_slot=20.0, T=20.0, n_positions=4,
              method="picard", relaxation=0.1, max_iter=2)
    with pytest.raises(ConvergenceError, match="статора"):
        stator_iron_loss_density(PARAMS, MAGNET, STEEL, **kw)
    with pytest.raises(ConvergenceError, match="статора"):
        stator_iron_loss(PARAMS, MAGNET, STEEL, **kw)
    geo0 = build_outrunner_spm_pmsm(PARAMS)
    idx = np.concatenate(rotor_side_cells(geo0))
    with pytest.raises(ConvergenceError, match="ротора"):
        rotor_frame_B_series(PARAMS, MAGNET, STEEL, speed_rpm=7000.0, cell_idx=idx, T=20.0,
                             mech_span=2.0 * math.pi / PARAMS.n_slots, n_positions=3,
                             method="picard", relaxation=0.1, max_iter=2)


def test_loss_sweep_by_newton_matches_tight_picard():
    # Тот же свип статора Ньютоном (по умолчанию) и туго сжатым эталоном-Пикаром: потери совпадают (интеграл
    # квадратичен по волне B — расхождение порядка допуска Пикара), все положения сошлись.
    kw = dict(speed_rpm=7000.0, i_peak=40.0, gamma_elec=math.pi / 2, turns_per_slot=20.0, T=20.0, n_positions=4)
    q_n, sw_n = stator_iron_loss_density(PARAMS, MAGNET, STEEL, **kw)
    q_p, sw_p = stator_iron_loss_density(PARAMS, MAGNET, STEEL, method="picard", relaxation=0.1, max_iter=5000, **kw)
    assert sw_n.all_converged and sw_p.all_converged
    # Ошибка остановки Пикара ≈ tol·ρ/(1−ρ): при ω = 0,1 здесь ρ ≈ 0,86 (≈120 итераций на 8 порядков) ⇒ ≈ 6·10⁻⁶
    # по B, у квадратичного по B интеграла — вдвое больше; граница 1e-4 — с запасом ×8.
    assert abs(q_n.sum() / q_p.sum() - 1.0) < 1e-4


def test_solve_machine_is_newton_by_default_and_agrees_with_picard():
    d = outrunner_pmsm_uav(mesh_size=6.0e-3, critical_mesh=False)
    new = solve_machine(d, T=20.0, i_peak=40.0, gamma_elec=math.pi / 2)
    assert new.method == "newton" and new.converged and new.residual <= 1e-9 and new.n_iterations <= 12
    pic = solve_machine(d, T=20.0, i_peak=40.0, gamma_elec=math.pi / 2, method="picard", relaxation=0.1,
                        max_iter=5000, tol=1e-9)
    assert pic.method == "picard" and pic.converged
    assert new.K_e == pytest.approx(pic.K_e, rel=1e-6)
    assert np.linalg.norm(new.field.B_cells - pic.field.B_cells) / np.linalg.norm(new.field.B_cells) < 1e-5
    with pytest.raises(ValueError, match="method"):
        solve_machine(d, method="что-то")
