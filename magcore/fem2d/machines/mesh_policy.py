"""
ПОЛИТИКА СЕТКИ: магнит и зазор — всегда мельчайшая сетка (решение Sergey: «пусть час-два, но точно»).

Вынесено из сценария статьи (`docs/papers/experiments/scenario_paper1.critical_mesh`) в пакет, чтобы политика
работала и в боевых путях (каталог машин, публикуемые сборки), а не жила в одном скрипте: на грубой сетке
величины размагничивания расходились до ×11 (падение K_e 3,66 % против 0,32 %, аудит 2026-11-16; Л-30).

Уровень выбран ПО СХОДИМОСТИ (fig_risk_map.py, outrunner 12/14, 130 °C / 40 А):
    равномерная 4 мм (172 яч.)   → падение ЭДС-пост. 6,29 %, повреждённый объём 54 %
    магнит+зазор 1 мм  (874)     → 4,55 %, 35 %
    магнит+зазор 0,5 мм (3041)   → 3,24 %, 27 %          <- принято
    зазор 0,25 мм       (5073)   → 3,25 %, 27 %  (зазор уже сошёлся при 0,5 мм)
    магнит 0,25 мм     (10842)   → 3,74 %, 29 %  (+0,5 п.п. при 6-кратной цене)
ЗАЗОР — от его ширины, не абсолютным числом: не меньше `MESH_AIR_GAP_PER_GAP` элементов поперёк (минимум
`verification.check_airgap_resolution`); на 2,3 элементах коэффициент проницания занижался (P_c 2,14 против
сошедшихся 2,73, Л-28).

Грубая сетка допустима только осознанно — в быстрых smoke-тестах структуры и в подборе ТЕПЛОВОГО режима
(тепловая часть по сетке сходится раньше); выводы о размагничивании — только на этой политике.
"""
from __future__ import annotations

from dataclasses import replace

from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams

MESH_MAGNET = 0.5e-3          # м — размер элемента в магните (уровень по сходимости, см. шапку)
MESH_AIR_GAP_PER_GAP = 3.0    # элементов поперёк зазора (минимум по verification.py)


def critical_mesh(params: OutrunnerPMSMParams) -> dict:
    """Размеры элементов по регионам для боевых расчётов: магнит и зазор — мельчайшая сетка [м]."""
    return {"magnet": MESH_MAGNET, "air_gap": float(params.air_gap) / MESH_AIR_GAP_PER_GAP}


def with_critical_mesh(params: OutrunnerPMSMParams) -> OutrunnerPMSMParams:
    """
    Те же параметры машины с политикой сетки: в магните и зазоре — не крупнее `critical_mesh`. Уже заданное
    более мелкое помельчение региона сохраняется (min), прочие регионы не трогаются.
    """
    sizes = dict(params.mesh_size_by_region or {})
    for name, size in critical_mesh(params).items():
        sizes[name] = min(float(sizes.get(name, size)), size)
    return replace(params, mesh_size_by_region=sizes)
