"""
FastAPI-бэкенд интерфейса: тонкая обёртка над magcore. Отдаёт сцену геометрии и считает
поле по запросу (solve_problem2d через MachineScenario), возвращая |B| по ячейкам + сводку
(момент, энергия, рабочая точка, демаг). Фронтенд (static/index.html) рисует НАСТОЯЩУЮ сетку
и поле. Вся физика — в magcore; сервер лишь связывает вход→решатель→выход.

Запуск:  python -m uvicorn webapp.server:app --port 8017   (из корня репозитория; нужен gmsh)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines import (
    MachineScenario,
    OutrunnerPMSMParams,
    build_outrunner_spm_pmsm,
    star_of_slots_layout,
)
from magcore.fem2d.model import magnetic_energy, problem_to_scene

app = FastAPI(title="MagField web")
_STATE: dict = {}
MESH_SIZE = 0.004


def _base():
    """Построить геометрию + обмотку ОДИН раз (кэш); сталь общая."""
    if "geom" not in _STATE:
        g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=MESH_SIZE))
        lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
        _STATE["geom"] = (g, lay, m270_35a_bh_curve())
    return _STATE["geom"]


def _scenario(material: str) -> MachineScenario:
    g, lay, steel = _base()
    magnet = sm2co17_magnet((1, 0, 0)) if material == "smco" else n42sh_magnet((1, 0, 0))
    return MachineScenario(geometry=g, magnet=magnet, steel=steel, layout=lay)


# Прогрев: собрать геометрию (gmsh) в ГЛАВНОМ потоке при импорте — иначе gmsh.initialize
# падает в воркер-потоке FastAPI («signal only works in main thread»). Решателю gmsh не нужен
# (работает на готовом TriangleMesh), поэтому /api/solve в воркер-потоке безопасен.
_base()


@app.get("/api/scene")
def api_scene() -> dict:
    """Сцена геометрии (реальная сетка + регионы) для стартового вьюпорта."""
    scen = _scenario("ndfeb")
    return problem_to_scene(scen.to_problem())


@app.post("/api/solve")
def api_solve(body: dict = Body(default={})) -> dict:
    """Решить задачу по режиму и вернуть |B| по ячейкам + инженерную сводку."""
    material = str(body.get("material", "ndfeb"))
    T = float(body.get("T", 20.0))
    i_peak = float(body.get("i_peak", 0.0))
    gamma = np.deg2rad(float(body.get("gamma_deg", 0.0)))
    turns = float(body.get("turns", 40.0))
    scen = _scenario(material)
    try:
        sol = scen.solve(T=T, i_peak=i_peak, gamma_elec=gamma, turns_per_slot=turns, max_iter=400)
    except ValueError as e:                       # перегрев магнита и т.п.
        return {"error": str(e)}

    B = sol.field.B_cells
    Bmag = np.hypot(B[:, 0], B[:, 1])
    op = scen.operating_point(sol)
    Bd_mean = float(np.average(op.B_op, weights=op.cell_volume))
    risk = sol.risk
    return {
        "converged": bool(sol.converged),
        "iters": int(sol.field.n_iterations),
        "Bx": np.round(B[:, 0], 4).tolist(),      # компоненты поля по ячейкам (для проб/карт)
        "By": np.round(B[:, 1], 4).tolist(),
        "Bmax": round(float(Bmag.max()), 3),
        "Bmean": round(float(Bmag.mean()), 3),
        "torque": round(float(scen.torque(sol)), 4),
        "energy": round(float(magnetic_energy(sol, axial_length=scen.axial_length)), 4),
        "Bd_mean": round(Bd_mean, 3),
        "Bd_worst": round(float(op.B_op.min()), 3),
        "Hop_worst_kA": round(float(op.worst_H_op() / 1e3), 0),
        "n_demag": int(risk.n_demagnetized),
        "n_mag": int(risk.cell_indices.size),
        "demag_frac": round(float(op.volume_fraction_below(op.knee_field)), 4),
    }


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="ui")
