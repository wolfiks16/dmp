"""
FastAPI-бэкенд интерфейса: тонкая обёртка над magcore. Отдаёт сцену геометрии (по плотности
сетки) и считает поле ФОНОВОЙ задачей (solve_problem2d через MachineScenario), чтобы тонкая
сетка не блокировала UI. Фронтенд опрашивает задачу и рисует настоящую сетку + поле.

⚠ gmsh.initialize ставит обработчик сигналов → падает в воркер-потоке. Поэтому ВСЕ геометрии
строятся при ИМПОРТЕ (главный поток); solve gmsh не трогает (работает на TriangleMesh) и
безопасно уходит в фон-поток.

Запуск:  python -m uvicorn webapp.server:app --port 8017   (из корня репозитория)
"""
from __future__ import annotations

import uuid
from concurrent.futures import Future, ThreadPoolExecutor
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

MESHES = {"coarse": 0.0055, "medium": 0.0038, "fine": 0.0026}   # ключ → mesh_size (м)
# Хордовый Picard по насыщающейся стали: тоньше сетка → нужна меньшая релаксация (иначе
# предельный цикл). Значения подобраны под сходимость (fine@0.05 сходится ~166 итер).
RELAX = {"coarse": 0.10, "medium": 0.07, "fine": 0.05}
MAXIT = {"coarse": 400, "medium": 700, "fine": 1000}
_GEOM: dict = {}
_SCENE: dict = {}
_EXEC = ThreadPoolExecutor(max_workers=1)
_JOBS: dict[str, Future] = {}


def _base(mesh_key: str):
    if mesh_key not in _GEOM:
        g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=MESHES[mesh_key]))
        lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
        _GEOM[mesh_key] = (g, lay, m270_35a_bh_curve())
    return _GEOM[mesh_key]


def _scenario(mesh_key: str, material: str) -> MachineScenario:
    g, lay, steel = _base(mesh_key)
    magnet = sm2co17_magnet((1, 0, 0)) if material == "smco" else n42sh_magnet((1, 0, 0))
    return MachineScenario(geometry=g, magnet=magnet, steel=steel, layout=lay)


def _key(v) -> str:
    return v if v in MESHES else "coarse"


# Прогрев: построить ВСЕ геометрии + сцены в главном потоке при импорте.
for _k in MESHES:
    _g, _lay, _st = _base(_k)
    _scen = MachineScenario(geometry=_g, magnet=n42sh_magnet((1, 0, 0)), steel=_st, layout=_lay)
    _SCENE[_k] = problem_to_scene(_scen.to_problem())


def _do_solve(body: dict) -> dict:
    """Тяжёлый расчёт в фон-потоке (без gmsh). Может бросить ValueError (перегрев магнита)."""
    mesh = _key(str(body.get("mesh", "coarse")))
    scen = _scenario(mesh, str(body.get("material", "ndfeb")))
    sol = scen.solve(
        T=float(body.get("T", 20.0)),
        i_peak=float(body.get("i_peak", 0.0)),
        gamma_elec=np.deg2rad(float(body.get("gamma_deg", 0.0))),
        turns_per_slot=float(body.get("turns", 40.0)),
        relaxation=RELAX[mesh],
        max_iter=MAXIT[mesh],
    )
    B = sol.field.B_cells
    Bmag = np.hypot(B[:, 0], B[:, 1])
    op = scen.operating_point(sol)
    risk = sol.risk
    return {
        "converged": bool(sol.converged),
        "iters": int(sol.field.n_iterations),
        "Bx": np.round(B[:, 0], 4).tolist(),
        "By": np.round(B[:, 1], 4).tolist(),
        "Bmax": round(float(Bmag.max()), 3),
        "Bmean": round(float(Bmag.mean()), 3),
        "torque": round(float(scen.torque(sol)), 4),
        "energy": round(float(magnetic_energy(sol, axial_length=scen.axial_length)), 4),
        "Bd_mean": round(float(np.average(op.B_op, weights=op.cell_volume)), 3),
        "Bd_worst": round(float(op.B_op.min()), 3),
        "Hop_worst_kA": round(float(op.worst_H_op() / 1e3), 0),
        "n_demag": int(risk.n_demagnetized),
        "n_mag": int(risk.cell_indices.size),
        "demag_frac": round(float(op.volume_fraction_below(op.knee_field)), 4),
    }


@app.get("/api/scene")
def api_scene(mesh: str = "coarse") -> dict:
    return _SCENE[_key(mesh)]


@app.post("/api/solve")
def api_solve(body: dict = Body(default={})) -> dict:
    """Поставить расчёт в фон-очередь; вернуть job_id для опроса."""
    jid = uuid.uuid4().hex[:12]
    _JOBS[jid] = _EXEC.submit(_do_solve, dict(body))
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def api_job(jid: str) -> dict:
    fut = _JOBS.get(jid)
    if fut is None:
        return {"status": "unknown"}
    if not fut.done():
        return {"status": "running"}
    try:
        return {"status": "done", "result": fut.result()}
    except Exception as e:  # noqa: BLE001 — перегрев магнита и пр. → в UI
        return {"status": "error", "error": str(e)}


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="ui")
