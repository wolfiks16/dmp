"""
FastAPI-бэкенд интерфейса: тонкая обёртка над magcore. Пользователь САМ задаёт сетку —
характерный размер элемента ПО РЕГИОНУ (посегментно) — и строит её; затем поле считается
ФОНОВОЙ задачей на этой сетке.

Потоки и gmsh:
  · Построение геометрии (gmsh) ставит обработчик сигналов → работает ТОЛЬКО в главном
    потоке. Поэтому /api/mesh — `async def`: тело корутины выполняется на потоке event-loop,
    а он в обычном запуске uvicorn и есть главный поток. Блокировка петли на ~1–2 с при
    явном «Построить сетку» приемлема (локальный однопользовательский инструмент).
  · Решение (solve) — чистый numpy, без gmsh → безопасно уходит в фон-поток, чтобы не
    блокировать UI на тяжёлой тонкой сетке.

Запуск:  python -m uvicorn webapp.server:app --port 8017   (из корня репозитория)
"""
from __future__ import annotations

import hashlib
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
from magcore.fem2d.machines.pmsm_outrunner import REGION_NAMES
from magcore.fem2d.model import magnetic_energy, problem_to_scene

app = FastAPI(title="MagField web")

# Регионы для посегментной сетки: порядок отображения, подпись, размер по умолчанию (мм).
# Сгущаем там, где важна физика (зазор/магниты — градиенты поля, демаг), ярма — грубее.
REGION_UI = [
    ("air_gap", "Зазор", 1.2),
    ("magnet", "Магниты", 1.5),
    ("tooth", "Зубцы", 2.5),
    ("slot", "Пазы", 3.0),
    ("stator_yoke", "Ярмо статора", 4.0),
    ("rotor_yoke", "Ярмо ротора", 4.0),
]
DEFAULT_SIZES_MM = {name: mm for name, _, mm in REGION_UI}

_GEOM: dict = {}          # mesh_id -> (MachineGeometry, layout, steel)
_SCENE: dict = {}         # mesh_id -> сцена (геометрия+регионы для рисования)
_EXEC = ThreadPoolExecutor(max_workers=1)
_JOBS: dict[str, Future] = {}


def _spec_from(body: dict) -> dict[str, float]:
    """Размеры по регионам в МЕТРАХ из тела запроса (мм), с подстановкой умолчаний."""
    raw = dict(body.get("sizes_mm") or {})
    out: dict[str, float] = {}
    for name in REGION_NAMES.values():
        mm = float(raw.get(name, DEFAULT_SIZES_MM[name]))
        if not (mm > 0.0):
            raise ValueError(f"размер сетки региона {name!r} должен быть > 0.")
        out[name] = mm / 1000.0
    return out


def _mesh_id(spec: dict[str, float]) -> str:
    key = "|".join(f"{k}:{spec[k]:.6g}" for k in sorted(spec))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _build_mesh(spec: dict[str, float]) -> str:
    """Построить (или взять из кэша) геометрию под посегментный spec. Возвращает mesh_id.
    ⚠ Вызывать только из главного потока (gmsh)."""
    mid = _mesh_id(spec)
    if mid in _GEOM:
        return mid
    params = OutrunnerPMSMParams(
        mesh_size=max(spec.values()), mesh_size_by_region=dict(spec)
    )
    g = build_outrunner_spm_pmsm(params)
    lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    _GEOM[mid] = (g, lay, m270_35a_bh_curve())
    # Сцена не зависит от марки магнита — строим на дефолтном для геометрии/регионов/осей.
    scen = MachineScenario(geometry=g, magnet=n42sh_magnet((1, 0, 0)), steel=_GEOM[mid][2],
                           layout=lay)
    _SCENE[mid] = problem_to_scene(scen.to_problem())
    return mid


def _region_counts(g) -> dict[str, int]:
    reg = np.asarray(g.region)
    return {name: int(np.count_nonzero(reg == code)) for code, name in REGION_NAMES.items()}


def _mesh_payload(mid: str) -> dict:
    g, _, _ = _GEOM[mid]
    return {
        "mesh_id": mid,
        "scene": _SCENE[mid],
        "n_cells": int(g.mesh.n_cells),
        "region_counts": _region_counts(g),
    }


# Прогрев: построить сетку по умолчанию в главном потоке при импорте (быстрый первый показ).
_DEFAULT_MESH_ID = _build_mesh(_spec_from({}))


def _do_solve(body: dict) -> dict:
    """Тяжёлый расчёт в фон-потоке (без gmsh). Может бросить ValueError (перегрев магнита)."""
    mid = str(body.get("mesh_id", "")) or _DEFAULT_MESH_ID
    if mid not in _GEOM:
        raise ValueError("сетка не найдена — постройте её заново.")
    g, lay, steel = _GEOM[mid]
    material = str(body.get("material", "ndfeb"))
    magnet = sm2co17_magnet((1, 0, 0)) if material == "smco" else n42sh_magnet((1, 0, 0))
    scen = MachineScenario(geometry=g, magnet=magnet, steel=steel, layout=lay)
    # Метод Ньютона (дефолт solve_problem2d): сходится за ~20 итераций НА ЛЮБОЙ плотности,
    # без подбора релаксации под сетку. max_iter с запасом.
    sol = scen.solve(
        T=float(body.get("T", 20.0)),
        i_peak=float(body.get("i_peak", 0.0)),
        gamma_elec=np.deg2rad(float(body.get("gamma_deg", 0.0))),
        turns_per_slot=float(body.get("turns", 40.0)),
        max_iter=60,
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


@app.get("/api/mesh_defaults")
def api_mesh_defaults() -> dict:
    """Список регионов для посегментной сетки: имя, подпись, размер по умолчанию (мм)."""
    return {"regions": [{"name": n, "label": lab, "mm": mm} for n, lab, mm in REGION_UI]}


@app.post("/api/mesh")
async def api_mesh(body: dict = Body(default={})) -> dict:
    """Построить сетку по посегментному spec (gmsh на главном потоке) и вернуть сцену."""
    try:
        spec = _spec_from(dict(body))
    except ValueError as e:
        return {"error": str(e)}
    mid = _build_mesh(spec)   # на потоке event-loop = главный поток (gmsh ОК)
    return _mesh_payload(mid)


@app.post("/api/solve")
def api_solve(body: dict = Body(default={})) -> dict:
    """Поставить расчёт на выбранной сетке в фон-очередь; вернуть job_id для опроса."""
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
