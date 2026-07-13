import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.fem2d.magneto_thermal import solve_magneto_thermal_demag
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.spaces import LagrangeP1Space2D


# --------------------------------------------------------------------------------------
# (A) МАТЕРИАЛОВЫЙ НОСИТЕЛЬ НОВИЗНЫ: температурная стойкость SmCo vs NdFeB.
# Детерминированно (без подгонки солвера), изолирует T-зависимость колена. При умеренном
# демаг-поле оба безопасны при 20°C; при нагреве NdFeB уходит за колено НАМНОГО раньше,
# т.к. |gamma_Hc| у него 0.55 против 0.20 у SmCo. Прямая связка с темой (SmCo для БПЛА).
# --------------------------------------------------------------------------------------
def test_smco_far_more_temperature_stable_than_ndfeb():
    nd = n42sh_magnet([1.0, 0.0, 0.0])
    sm = sm2co17_magnet([1.0, 0.0, 0.0])
    H_op = -6.0e5   # демагнитизирующее рабочее поле [А/м]

    def onset(m):
        for T in range(20, 401):
            if m.risk_margin(H_op, T) < 0.0:
                return T
        return None

    T_nd, T_sm = onset(nd), onset(sm)
    assert T_nd is not None and T_sm is not None
    # Оба безопасны при комнатной температуре при этом поле.
    assert nd.risk_margin(H_op, 20.0) > 0.0 and sm.risk_margin(H_op, 20.0) > 0.0
    # SmCo держится НАМНОГО горячее (измерено ~248 vs ~122 °C).
    assert T_sm > T_nd + 80

    # При T, где NdFeB уже за коленом, SmCo ещё безопасен.
    T = T_nd + 15
    assert nd.risk_margin(H_op, T) < 0.0
    assert sm.risk_margin(H_op, T) > 0.0

    # Плата за стойкость: у SmCo ниже остаточная индукция (проектный trade-off).
    assert sm.Br(20.0) < nd.Br(20.0)


# --------------------------------------------------------------------------------------
# (B) СВЯЗКА тепло→магнит→демаг (полевая, 2D-T3): тепловыделение (потери) греет магнит,
# рост T ухудшает демаг-запас. Монотонная деградация — без knife-edge подгонки.
# --------------------------------------------------------------------------------------
def _pipeline(magnet, load, *, B0=(-0.25, 0.0)):
    mesh = build_structured_rectangle_tri_mesh(16, 16, x0=-2, x1=2, y0=-2, y1=2)
    space = LagrangeP1Space2D(mesh)
    nc = mesh.n_cells
    mask = np.array(
        [np.linalg.norm(mesh.cell_centroid(c)) < 0.8 for c in range(nc)], dtype=bool
    )
    q = np.where(mask, float(load), 0.0)             # потери в магните
    return solve_magneto_thermal_demag(
        space, magnet, mask, heat_source_cells=q, k_cells=np.full(nc, 1.0),
        h=2.0, T_amb=20.0, applied_B0=B0,
    )


def test_thermal_load_drives_demag_risk():
    magnet = n42sh_magnet([1.0, 0.0, 0.0])
    cold = _pipeline(magnet, 0.0)
    hot = _pipeline(magnet, 120.0)

    assert cold.em_converged and hot.em_converged
    # Тепловая связка: нагрузка греет магнит.
    assert cold.T_magnet == pytest.approx(20.0)       # без потерь — окружающая T
    assert hot.T_magnet > 70.0                        # потери подняли T (измерено ~86 °C)
    # Демаг-риск растёт с температурой (монотонная деградация запаса).
    assert hot.risk.worst_margin < cold.risk.worst_margin
    assert cold.risk.n_demagnetized == 0              # холодный магнит цел
    assert hot.risk.n_demagnetized > 0                # горячий — часть за коленом


def test_pipeline_smco_safe_where_ndfeb_demagnetizes():
    # ЯДРО ДЕМОНСТРАЦИИ ТЕМЫ в самой связке: одинаковая тепловая нагрузка → одинаковая
    # температура магнита; исход размагничивания решает МАТЕРИАЛ. При нагреве до ~86 °C
    # NdFeB уходит за колено, а SmCo (T-стабильный) остаётся цел.
    nd = _pipeline(n42sh_magnet([1.0, 0.0, 0.0]), 120.0)
    sm = _pipeline(sm2co17_magnet([1.0, 0.0, 0.0]), 120.0)

    assert nd.em_converged and sm.em_converged
    assert nd.T_magnet == sm.T_magnet                 # тепловая сторона от материала не зависит
    assert nd.risk.n_demagnetized > 0                 # NdFeB частично за коленом
    assert sm.risk.n_demagnetized == 0                # SmCo безопасен при той же T


def test_magnet_temperature_limit():
    # Предел валидности модели = T0 + 100/max(коэфф.); выше — Br/Hc масштабируется в ≤0.
    nd = n42sh_magnet([1.0, 0.0, 0.0])
    sm = sm2co17_magnet([1.0, 0.0, 0.0])
    assert nd.temperature_limit() == pytest.approx(20.0 + 100.0 / 0.55, abs=1e-6)
    assert sm.temperature_limit() == pytest.approx(20.0 + 100.0 / 0.20, abs=1e-6)
    assert sm.temperature_limit() > nd.temperature_limit()   # SmCo валиден до бóльших T


def test_overheat_raises_clear_error():
    # Перегрев за предел модели ⇒ понятное MagnetOverheatedError (не криптичный отказ в curve_at).
    from magcore.fem2d.magneto_thermal import (
        MagnetOverheatedError,
        solve_magneto_thermal_demag,
    )

    magnet = n42sh_magnet([1.0, 0.0, 0.0])
    mesh = build_structured_rectangle_tri_mesh(12, 12, x0=-10, x1=10, y0=-10, y1=10)
    space = LagrangeP1Space2D(mesh)
    nc = mesh.n_cells
    mask = np.array([np.linalg.norm(mesh.cell_centroid(c)) < 5.0 for c in range(nc)], dtype=bool)
    q = np.where(mask, 50.0, 0.0)                    # огромная нагрузка на большой магнит
    with pytest.raises(MagnetOverheatedError) as ei:
        solve_magneto_thermal_demag(
            space, magnet, mask, heat_source_cells=q, k_cells=np.full(nc, 1.0),
            h=1.0, T_amb=20.0, applied_B0=(0.0, 0.0),
        )
    assert ei.value.T_magnet > ei.value.limit
    assert ei.value.T_field is not None             # температурное поле доступно для показа
