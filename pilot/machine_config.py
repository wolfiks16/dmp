"""
Разбор конфига машины (TOML) → `MachineProblem` (P7). Пользователь задаёт ВСЮ постановку
статической задачи текстовым файлом БЕЗ правки Python: геометрия сечения, материалы
(магнит как ДАННЫЕ + сталь), обмотка, режим (сценарий/T/ток). Интерфейс — чистая обёртка
над согласованной моделью задачи; сам расчёт и валидация — в `magcore`.
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines import (
    MachineProblem,
    OutrunnerPMSMParams,
    Scenario,
    build_outrunner_spm_pmsm,
    star_of_slots_layout,
)
from pilot.config import build_magnet   # переиспользуем: ndfeb|smco|custom (материал как данные)

_GEOM_KEYS = (
    "n_slots", "n_poles", "R_bore", "h_stator_yoke", "h_tooth", "air_gap",
    "h_magnet", "h_rotor_yoke", "tooth_width_frac", "magnet_embrace",
    "axial_length", "mesh_size",
)


@dataclass(frozen=True)
class MachineRun:
    problem: MachineProblem
    material_name: str
    assess_impact: bool
    output_dir: Path


def build_steel(cfg: dict):
    """Сталь из [steel]: пока пресет 'm270' (замена данных тривиальна — SteelBHCurve)."""
    mat = str(cfg.get("material", "m270")).lower()
    if mat in ("m270", "m270-35a", "m270_35a"):
        return m270_35a_bh_curve()
    raise ValueError(f"неизвестная сталь '{mat}' (пока поддержан пресет m270).")


def load_machine_run(path: str | Path) -> MachineRun:
    """Прочитать TOML → собрать геометрию (gmsh), материалы, обмотку, режим → MachineProblem."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    geom = {k: raw.get("geometry", {})[k] for k in _GEOM_KEYS if k in raw.get("geometry", {})}
    params = OutrunnerPMSMParams(**geom)
    geometry = build_outrunner_spm_pmsm(params)

    magnet, material_name = build_magnet(raw.get("magnet", {}))
    steel = build_steel(raw.get("steel", {}))
    layout = star_of_slots_layout(params.n_slots, params.n_poles)

    wind = raw.get("winding", {})
    reg = raw.get("regime", {})
    sol = raw.get("solver", {})
    out = raw.get("output", {})

    # T передаётся КАК ЗАДАНО (по умолчанию 20): S1 с T≠20 — противоречие, его ловит
    # validate() (не глушим молча). Для S1 просто не указывайте T.
    scenario = Scenario(str(reg.get("scenario", "S1")).upper())

    problem = MachineProblem(
        geometry=geometry, magnet=magnet, steel=steel, layout=layout,
        scenario=scenario, T=float(reg.get("T", 20.0)),
        i_peak=float(reg.get("i_peak", 0.0)),
        gamma_elec=math.radians(float(reg.get("gamma_deg", 0.0))),
        turns_per_slot=float(wind.get("turns_per_slot", 0.0)),
        relaxation=float(sol.get("relaxation", 0.1)),
        max_iter=int(sol.get("max_iter", 200)),
        tol=float(sol.get("tol", 1.0e-6)),
    )
    return MachineRun(
        problem=problem,
        material_name=material_name,
        assess_impact=bool(reg.get("assess_impact", False)),
        output_dir=Path(out.get("dir", "pilot/output")),
    )
