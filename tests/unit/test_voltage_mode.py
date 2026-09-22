from magcore.fem2d.losses import copper_resistivity
from magcore.fem2d.machines.characteristics import (
    phase_resistance,
    voltage_limited_operating_point,
)
from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    build_outrunner_spm_pmsm,
)

# P-B6: режим напряжения (деградация характеристик при нагреве). Оракулы:
#  структура лумпед-модели (I≥0, насыщение по скорости, монотонность по R и K_e, КПД∈(0,1));
#  R(T) растёт с температурой ∝ ρ_cu(T); нагрев+демаг (R↑, K_e↓) ⇒ момент и КПД падают.


def test_voltage_limited_structure():
    base = voltage_limited_operating_point(
        voltage=22.2, omega_mech=100.0, emf_constant=0.027, resistance=0.05)
    assert base["current"] > 0.0 and base["torque"] > 0.0
    assert 0.0 < base["efficiency"] < 1.0

    # насыщение по скорости: E ≥ U ⇒ ток 0
    sat = voltage_limited_operating_point(
        voltage=22.2, omega_mech=2000.0, emf_constant=0.027, resistance=0.05)
    assert sat["current"] == 0.0 and sat["torque"] == 0.0

    # выше сопротивление (горячее) ⇒ меньше ток и момент
    hotR = voltage_limited_operating_point(
        voltage=22.2, omega_mech=100.0, emf_constant=0.027, resistance=0.10)
    assert hotR["current"] < base["current"]
    assert hotR["torque"] < base["torque"]

    # при E≪U момент ∝ K_e ⇒ демаг (ниже K_e) снижает момент
    loKe = voltage_limited_operating_point(
        voltage=22.2, omega_mech=100.0, emf_constant=0.020, resistance=0.05)
    assert loKe["torque"] < base["torque"]


def test_phase_resistance_rises_with_temperature():
    geo = build_outrunner_spm_pmsm(OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.005))
    R20 = phase_resistance(geo, turns_per_slot=20.0, slot_fill=0.45, T=20.0)
    R120 = phase_resistance(geo, turns_per_slot=20.0, slot_fill=0.45, T=120.0)
    assert R20 > 0.0
    assert R120 > R20
    # рост ровно по ρ_cu(T)
    assert abs(R120 / R20 - copper_resistivity(120.0) / copper_resistivity(20.0)) < 1e-9


def test_heating_degrades_torque_and_efficiency():
    # холодный/номинал против горячего+демаг (R↑ от нагрева меди, K_e↓ от B_r(T)+необр. демага)
    cold = voltage_limited_operating_point(
        voltage=22.2, omega_mech=600.0, emf_constant=0.027, resistance=0.05)
    hot = voltage_limited_operating_point(
        voltage=22.2, omega_mech=600.0, emf_constant=0.024, resistance=0.07)
    assert hot["torque"] < cold["torque"]
    assert hot["efficiency"] < cold["efficiency"]
