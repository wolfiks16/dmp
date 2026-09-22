"""
КАТАЛОГ ГОТОВЫХ МАШИН — конкретные изделия как НАБОРЫ ПАРАМЕТРОВ, а не как код.

Каждая запись = `MachineDefinition` из библиотеки типов (`library.py`). Добавить новую
машину = дописать функцию с размерами и материалами; физика и решатель не трогаются.

Источники всех чисел — `docs/sources_registry.md`.
"""
from __future__ import annotations

from magcore.domain.magnet_model import (
    AnisotropicBHTMagnet,
    ks25dts240_magnet,
    n35_magnet,
    n42sh_magnet,
)
from magcore.domain.steel_curves import (
    m270_35a_cogent_bh_curve,
    steel10_bh_curve,
)
from magcore.fem2d.machines.library import (
    MAGNETS_ON_ROTOR,
    MAGNETS_ON_STATOR,
    CommutatorWinding,
    MachineDefinition,
    MachineMaterials,
    ThreePhaseWinding,
    Topology,
)
from magcore.fem2d.machines.iron_loss import (
    STEEL10_RESISTIVITY,
    SteinmetzCoefficients,
)
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams

# Электропроводность магнитов при 20 °C [См/м] (sources_registry.md §2):
# NdFeB ρ⊥≈1.27 µΩ·м (Ruoho), Sm2Co17 ρ≈0.85 µΩ·м (Eclipse/FEMM).
# ⚠ SmCo ПРОВОДНЕЕ ⇒ у него БОЛЬШЕ вихревых потерь при равной пульсации.
SIGMA_MAGNET_20 = {"N35": 1.0 / 1.27e-6, "N42SH-representative": 1.0 / 1.27e-6,
                   "KS25DTs-240": 1.0 / 0.85e-6}


def _sigma_of(magnet: AnisotropicBHTMagnet) -> float:
    """
    σ по МАРКЕ фактически установленного магнита. Обязательно брать из самого магнита,
    а не из машины: при замене NdFeB→SmCo проводимость РАСТЁТ (0.79→1.18 МС/м) и вихревые
    потери увеличиваются — это реальный минус SmCo, который нельзя потерять при смене марки.
    """
    return SIGMA_MAGNET_20.get(magnet.material_id, 1.0 / 1.27e-6)

EASY_AXIS = (1.0, 0.0, 0.0)


# ------------------------------------------------------------------ ДП25 (щёточный ДПТ)

def dp25_brushed_dc(magnet: AnisotropicBHTMagnet | None = None,
                    mesh_size: float = 0.25e-3) -> MachineDefinition:
    """
    **ДП25-16-7-24** — щёточный ДПТ с ПМ, 16 Вт, 24 В, 4 полюса, 13 пазов, Ø25 мм.
    ТУ КМИЖ.524212.006. Эталон валидации: паспортное **K_e = 0.02946 В·с/рад**
    (R якоря 3.9±0.4 Ом; расчёт из обмоточных данных даёт 3.77 Ом, −3.5 %).

    Геометрия — из STEP «3D Сборка 1.stp» (слои сходятся точно в Ø25).
    Магниты **сплошные** (N_seg=1). Магниты на СТАТОРЕ, вращается якорь.
    """
    _mag_dp25 = magnet or n35_magnet(EASY_AXIS)      # марка изделия по умолчанию — N35
    params = OutrunnerPMSMParams(
        n_slots=13, n_poles=4,
        R_bore=2.15e-3, h_stator_yoke=1.85e-3, h_tooth=4.532e-3,
        air_gap=0.268e-3, h_magnet=1.90e-3, h_rotor_yoke=1.80e-3,
        tooth_width_frac=0.70,          # оценка по площади меди (1.53 мм²/паз, k_зап 0.35–0.45)
        magnet_embrace=0.726,           # охват 65.3° из 90° (STEP)
        axial_length=16.0e-3,           # длина пакета (STEP)
        mesh_size=mesh_size,
        # ⚠ зазор 0.268 мм ТОНЬШЕ глобального элемента — без измельчения поле в нём
        # не разрешается и итерация не сходится (см. solver_spec.md §5)
        mesh_size_by_region={"air_gap": 0.08e-3, "magnet": 0.15e-3, "tooth": 0.18e-3},
    )
    return MachineDefinition(
        name="ДП25-16-7-24",
        params=params,
        materials=MachineMaterials(
            magnet=_mag_dp25,
            steel_armature=m270_35a_cogent_bh_curve(),   # якорь — электротехническая
            steel_yoke=steel10_bh_curve(),               # корпус — Сталь 10
            steinmetz_armature=SteinmetzCoefficients.m270_35a_cogent(),
            steinmetz_yoke=SteinmetzCoefficients.steel10_laminated(0.5e-3),
            yoke_solid=True,                             # корпус МАССИВНЫЙ (не шихтованный)
            yoke_resistivity=STEEL10_RESISTIVITY,
            magnet_sigma_20=_sigma_of(_mag_dp25),        # σ следует за маркой магнита
            magnet_segments=1,                           # магниты СПЛОШНЫЕ (Sergey)
        ),
        # простая ВОЛНОВАЯ: 13 пазов × 60 проводников = 780; a=1; скос пакета 11.5°;
        # провод ПНЭТ-имид Ø0.18 мм, средняя длина витка 57 мм (чертёж «Параметры обмотки якоря»)
        winding=CommutatorWinding(conductors_total=780, parallel_path_pairs=1,
                                  slot_fill=0.45, skew_deg=11.5,
                                  wire_diameter=0.18e-3, mean_turn_length=57.0e-3),
        topology=Topology(magnets_on=MAGNETS_ON_STATOR, outer_part_rotates=False),
    )


DP25_NAMEPLATE_KE = 0.02946      # В·с/рад — цель валидации (вывод из ТУ, самосогласован)
DP25_NAMEPLATE_R = 3.9           # Ом ± 0.4


# ------------------------------------------------------------------ outrunner PMSM (БПЛА)

def outrunner_pmsm_uav(magnet: AnisotropicBHTMagnet | None = None,
                       steel: str = "steel10",
                       mesh_size: float = 3.0e-3) -> MachineDefinition:
    """
    Представительный **outrunner SPM PMSM** для БПЛА: 12 пазов / 14 полюсов, Ø58×30 мм,
    kV 350, 6S (22.2 В), номинал 40 А, пик 2 кВт, ХХ ≈ 7000 об/мин (f ≈ 817 Гц).

    `steel`: 'steel10' (магнитопровод изделия) | 'm270' (электротехническая, эталон сравнения).
    Магниты на РОТОРЕ (вращаются) ⇒ вихревые потери считаются в системе ротора.
    """
    _mag_pm = magnet or n42sh_magnet(EASY_AXIS)
    curve = steel10_bh_curve() if steel == "steel10" else m270_35a_cogent_bh_curve()
    params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=mesh_size)
    return MachineDefinition(
        name="outrunner PMSM 12/14 (БПЛА)",
        params=params,
        materials=MachineMaterials(
            magnet=_mag_pm,
            steel_armature=curve, steel_yoke=curve,
            steinmetz_armature=(SteinmetzCoefficients.steel10_laminated(0.5e-3)
                                if steel == "steel10" else
                                SteinmetzCoefficients.m270_35a_cogent()),
            yoke_solid=(steel == "steel10"),             # у изделия ярмо ротора массивное
            yoke_resistivity=(STEEL10_RESISTIVITY if steel == "steel10" else None),
            magnet_sigma_20=_sigma_of(_mag_pm),          # σ следует за маркой магнита
            magnet_segments=1,
        ),
        winding=ThreePhaseWinding(turns_per_slot=20.0, slot_fill=0.45),
        topology=Topology(magnets_on=MAGNETS_ON_ROTOR, outer_part_rotates=True),
    )


# ------------------------------------------------------------------ доступные варианты

MAGNETS = {
    "N35": n35_magnet,                    # стандартный NdFeB (ДП25)
    "N42SH": n42sh_magnet,                # высокотемпературный NdFeB
    "KS25DTs-240": ks25dts240_magnet,     # Sm2Co17 ГОСТ 21559-76, верх диапазона
}

MACHINES = {
    "dp25": dp25_brushed_dc,
    "outrunner_pmsm_uav": outrunner_pmsm_uav,
}


def build(machine: str, magnet: str | None = None, **kwargs) -> MachineDefinition:
    """Собрать машину по имени из каталога: build('dp25', magnet='KS25DTs-240')."""
    if machine not in MACHINES:
        raise ValueError("неизвестная машина %r; доступны: %s" % (machine, sorted(MACHINES)))
    if magnet is not None:
        if magnet not in MAGNETS:
            raise ValueError("неизвестный магнит %r; доступны: %s" % (magnet, sorted(MAGNETS)))
        kwargs["magnet"] = MAGNETS[magnet](EASY_AXIS)
    return MACHINES[machine](**kwargs)
