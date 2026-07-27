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
import json
import math
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles

from magcore.domain.magnet_model import magnet_from_datasheet, n42sh_magnet, sm2co17_magnet
from magcore.domain.steel_curves import SteelBHCurve, m270_35a_bh_curve
from magcore.fem2d.machines import (
    MachineScenario,
    OutrunnerPMSMParams,
    build_outrunner_spm_pmsm,
    star_of_slots_layout,
)
from magcore.fem2d.machines.characteristics import machine_characteristics
from magcore.fem2d.machines.iron_loss import (
    SteinmetzCoefficients,
    efficiency,
    electrical_frequency,
    stator_iron_loss,
)
from magcore.fem2d.machines.pmsm_outrunner import REGION_NAMES
from magcore.fem2d.machines.rotor_sweep import (
    RotorDamage,
    build_rotor_geometries,
    electrical_period_angles,
)
from magcore.fem2d.machines.spoke_pmsm import (
    MagnetShape,
    SpokeMotorParams,
    build_spoke_pmsm,
)
from magcore.fem2d.machines.thermal_scenario import (
    copper_loss_watts,
    run_machine_thermal_demag,
)
from magcore.fem2d.model import (
    Air,
    GeoObject,
    MagnetMaterial,
    SteelMaterial,
    auto_domain,
    build_object_problem,
    magnetic_energy,
    operating_point,
    problem_to_scene,
    solve_problem2d,
)

app = FastAPI(title="MagField web")


@app.middleware("http")
async def _no_cache(request, call_next):
    """Не кэшировать HTML/UI: правки интерфейса сразу видны при перезагрузке (без Ctrl+Shift+R)."""
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp

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

# Параметры геометрии outrunner PMSM, задаваемые пользователем: имя поля OutrunnerPMSMParams,
# подпись, значение по умолчанию, вид (int — счёт; mm — длина в мм→м; frac — доля 0..1).
GEOM_UI = [
    ("n_slots", "Пазов (зубцов)", 12, "int"),
    ("n_poles", "Полюсов (магнитов)", 14, "int"),
    ("R_bore", "Радиус расточки", 10.0, "mm"),
    ("h_stator_yoke", "Ярмо статора", 4.0, "mm"),
    ("h_tooth", "Зубец (длина)", 8.0, "mm"),
    ("air_gap", "Зазор", 1.0, "mm"),
    ("h_magnet", "Магнит (толщина)", 3.0, "mm"),
    ("h_rotor_yoke", "Ярмо ротора", 3.0, "mm"),
    ("tooth_width_frac", "Доля зубца в шаге", 0.5, "frac"),
    ("magnet_embrace", "Охват полюса магнитом", 0.83, "frac"),
    ("axial_length", "Осевая длина", 30.0, "mm"),
]

# СПИЦЕВОЙ двигатель (реальная схема заказчика): параметры в ДИАМЕТРАХ (мм), с выбором формы
# магнита. Поле SpokeMotorParams, подпись как в чертеже, значение по умолч., вид.
SPOKE_UI = [
    ("n_teeth", "Лучи (зубцы)", 12, "int"),
    ("n_poles", "Магниты (полюса)", 14, "int"),
    ("D_shell_out_mm", "Внешний ⌀ обечайки", 50.5, "mm"),
    ("D_shell_in_mm", "Внутренний ⌀ обечайки", 45.4, "mm"),
    ("D_magnet_in_mm", "Внутренний ⌀ магнита", 41.26, "mm"),
    ("D_tooth_out_mm", "Внешний ⌀ лучей (верх топорика)", 40.86, "mm"),
    ("D_shoe_in_mm", "Внутренний ⌀ луча (низ топорика)", 38.5, "mm"),
    ("D_base_out_mm", "Внешний ⌀ основания луча", 19.5, "mm"),
    ("D_base_in_mm", "Внутренний ⌀ основания (расточка)", 17.0, "mm"),
    ("tooth_stem_mm", "Толщина луча (стержня)", 2.5, "mm"),
    ("shoe_width_mm", "Ширина топорика", 8.55, "mm"),
    ("stack_length_mm", "Длина пакета (осевая)", 20.0, "mm"),
    ("magnet_shape", "Форма магнита", "truncated_sector", "shape"),
    ("sector_angle_deg", "Угол сектора", 25.5, "deg"),
    ("truncated_width_mm", "Ширина усечённого сектора", 8.0, "mm"),
    ("prism_width_mm", "Ширина призмы", 8.0, "mm"),
    ("prism_thickness_mm", "Толщина призмы", 1.5, "mm"),
]
SPOKE_SHAPES = [
    ("sector", "Сектор"),
    ("truncated_sector", "Усечённый сектор"),
    ("prism", "Призма"),
]

_GEOM: dict = {}          # mesh_id -> (MachineGeometry, layout, steel)
_SCENE: dict = {}         # mesh_id -> сцена (геометрия+регионы для рисования)
_OBJ: dict = {}           # model_id -> Problem2D (объектная произвольная модель)
_CORES = os.cpu_count() or 4

try:                                                # ограничение BLAS-потоков на расчёт
    from threadpoolctl import threadpool_limits as _tpl

    def _limit_threads(n: int):
        return _tpl(limits=max(1, int(n)))
except Exception:                                   # noqa: BLE001 — нет threadpoolctl
    import contextlib

    def _limit_threads(n: int):
        return contextlib.nullcontext()


class _JobManager:
    """Фоновые расчёты с НАСТРАИВАЕМЫМ числом параллельных задач и очередью.

    Пул потоков большой, но реальную параллельность гейтит ``max_parallel``: лишние
    задачи ждут в очереди (их конфиг уже сохранён — запустятся, даже если пользователь
    сменил экран). На каждый расчёт ограничиваем BLAS-потоки (``cores // max_parallel``),
    чтобы N параллельных расчётов не пересыщали процессор."""

    def __init__(self, max_parallel: int = 1):
        self.max_parallel = max(1, int(max_parallel))
        self.jobs: dict[str, dict] = {}
        self.queue: list[str] = []
        self.running: set[str] = set()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=64)

    def submit(self, kind: str, label: str, fn, body: dict) -> str:
        jid = uuid.uuid4().hex[:12]
        with self.lock:
            self.jobs[jid] = {
                "id": jid, "kind": kind, "label": label, "status": "queued",
                "created": time.time(), "started": None, "finished": None,
                "result": None, "error": None, "_fn": fn, "_body": body,
            }
            self.queue.append(jid)
            self._pump()
        return jid

    def _pump(self) -> None:                        # вызывать под self.lock
        while len(self.running) < self.max_parallel and self.queue:
            jid = self.queue.pop(0)
            rec = self.jobs[jid]
            rec["status"] = "running"
            rec["started"] = time.time()
            self.running.add(jid)
            self.pool.submit(self._run, jid)

    def _run(self, jid: str) -> None:
        rec = self.jobs[jid]
        per = max(1, _CORES // max(1, self.max_parallel))
        try:
            with _limit_threads(per):
                rec["result"] = rec["_fn"](rec["_body"])
            rec["status"] = "done"
        except Exception as e:                      # noqa: BLE001 — перегрев/данные → в UI
            rec["error"] = str(e)
            rec["status"] = "error"
        finally:
            rec["finished"] = time.time()
            with self.lock:
                self.running.discard(jid)
                self._pump()

    def set_max_parallel(self, n: int) -> int:
        with self.lock:
            self.max_parallel = max(1, min(int(n), 64))
            self._pump()
            return self.max_parallel

    def status(self, jid: str) -> dict:
        rec = self.jobs.get(jid)
        if rec is None:
            return {"status": "unknown"}
        if rec["status"] == "done":
            return {"status": "done", "result": rec["result"]}
        if rec["status"] == "error":
            return {"status": "error", "error": rec["error"]}
        return {"status": rec["status"]}            # queued | running

    @staticmethod
    def _view(rec: dict) -> dict:
        return {k: rec[k] for k in ("id", "kind", "label", "status",
                                    "created", "started", "finished", "error")}

    def listing(self) -> list[dict]:
        with self.lock:
            recs = sorted(self.jobs.values(), key=lambda r: r["created"], reverse=True)
            return [self._view(r) for r in recs]

    def clear_finished(self) -> None:
        with self.lock:
            for jid in [j for j, r in self.jobs.items()
                        if r["status"] in ("done", "error")]:
                del self.jobs[jid]


_JM = _JobManager(max_parallel=1)


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


def _geom_from(body: dict) -> dict[str, float]:
    """Параметры геометрии из тела запроса (мм→м, доли/счёт как есть), с умолчаниями."""
    raw = dict(body.get("geom") or {})
    out: dict[str, float] = {}
    for name, _, default, kind in GEOM_UI:
        v = raw.get(name, default)
        if kind == "int":
            out[name] = int(v)
        elif kind == "mm":
            out[name] = float(v) / 1000.0
        else:  # frac
            out[name] = float(v)
    return out


def _params_from(body: dict) -> tuple[OutrunnerPMSMParams, str]:
    """Собрать OutrunnerPMSMParams (геометрия + посегментная сетка) + детерминированный mesh_id."""
    geom = _geom_from(body)
    spec = _spec_from(body)
    params = OutrunnerPMSMParams(
        mesh_size=max(spec.values()), mesh_size_by_region=dict(spec), **geom
    )
    key = ("|".join(f"{k}:{geom[k]:.6g}" for k in sorted(geom))
           + "#" + "|".join(f"{k}:{spec[k]:.6g}" for k in sorted(spec)))
    return params, hashlib.sha1(key.encode()).hexdigest()[:12]


def _build_mesh(params: OutrunnerPMSMParams, mid: str) -> str:
    """Построить (или взять из кэша) геометрию+сетку под params. Возвращает mesh_id.
    ⚠ Вызывать только из главного потока (gmsh). Может бросить ValueError (плохая геометрия)."""
    if mid in _GEOM:
        return mid
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


# ---- СПИЦЕВОЙ ДВИГАТЕЛЬ (новый генератор, реальная схема заказчика) ----

def _spoke_params_from(body: dict) -> tuple[SpokeMotorParams, str]:
    """Собрать SpokeMotorParams из тела запроса + детерминированный mesh_id."""
    vals = dict(body.get("params") or {})
    kw: dict = {}
    for name, _lbl, default, kind in SPOKE_UI:
        v = vals.get(name, default)
        if kind == "int":
            kw[name] = int(v)
        elif kind == "shape":
            kw[name] = MagnetShape(str(v))
        else:                                    # mm | deg — числа
            kw[name] = float(v)
    raw = dict(body.get("sizes_mm") or {})
    sizes = {name: float(raw.get(name, DEFAULT_SIZES_MM[name])) for name in REGION_NAMES.values()}
    kw["mesh_size_by_region"] = sizes
    kw["mesh_size_mm"] = min(sizes.values())
    params = SpokeMotorParams(**kw)
    key = "spoke#" + "|".join(f"{k}:{v}" for k, v in sorted(
        (k, (v.value if isinstance(v, MagnetShape) else v)) for k, v in kw.items() if k != "mesh_size_by_region"))
    key += "#" + "|".join(f"{k}:{sizes[k]:.4g}" for k in sorted(sizes))
    return params, "sp" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _build_spoke_mesh(params: SpokeMotorParams, mid: str) -> str:
    """Построить (или взять из кэша) спицевую геометрию. ⚠ gmsh — вызывать из главного потока."""
    if mid in _GEOM:
        return mid
    g = build_spoke_pmsm(params)
    lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    _GEOM[mid] = (g, lay, m270_35a_bh_curve())
    scen = MachineScenario(geometry=g, magnet=n42sh_magnet((1, 0, 0)), steel=_GEOM[mid][2],
                           layout=lay)
    _SCENE[mid] = problem_to_scene(scen.to_problem())
    return mid


# Прогрев: построить модель по умолчанию в главном потоке при импорте (быстрый первый показ).
_DEF_PARAMS, _DEF_MID = _params_from({})
_DEFAULT_MESH_ID = _build_mesh(_DEF_PARAMS, _DEF_MID)


def _do_solve(body: dict) -> dict:
    """Тяжёлый расчёт в фон-потоке (без gmsh). Может бросить ValueError (перегрев магнита)."""
    mid = str(body.get("mesh_id", "")) or _DEFAULT_MESH_ID
    if mid not in _GEOM:
        raise ValueError("сетка не найдена — постройте её заново.")
    g, lay, _ = _GEOM[mid]
    magnet = _magnet_by_id(str(body.get("material", "ndfeb")))
    steel = _steel_by_id(str(body.get("steel", "steel")))
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


def _do_scenario(body: dict) -> dict:
    """
    Сценарий тепловой стойкости магнита (S3) на сечении PMSM — в фон-потоке.

    Тепловая связка магнит↔тепло↔необратимый демаг → характеристики «до/после» → (опц.)
    потери в железе и КПД. Возвращает структуру для панели S3.
    """
    mid = str(body.get("mesh_id", "")) or _DEFAULT_MESH_ID
    if mid not in _GEOM:
        raise ValueError("сетка не найдена — постройте её заново.")
    g, lay, _ = _GEOM[mid]
    params = g.params
    magnet = _magnet_by_id(str(body.get("material", "ndfeb")))
    steel = _steel_by_id(str(body.get("steel", "steel")))
    scen = MachineScenario(geometry=g, magnet=magnet, steel=steel, layout=lay)

    ipk = float(body.get("i_peak", 60.0))
    turns = float(body.get("turns", 20.0))
    gamma = np.deg2rad(float(body.get("gamma_deg", 0.0)))
    T_amb = float(body.get("T_amb", 20.0))
    h = float(body.get("h", 60.0))
    dt = float(body.get("dt", 2.0))
    n_steps = int(body.get("n_steps", 30))
    slot_fill = float(body.get("slot_fill", 0.45))
    dens = float(body.get("magnet_density", 7500.0))
    with_losses = bool(body.get("with_losses", False))
    rpm = float(body.get("speed_rpm", 3000.0))

    res = run_machine_thermal_demag(
        scen, i_peak=ipk, turns_per_slot=turns, gamma_elec=gamma, slot_fill=slot_fill,
        h=h, T_amb=T_amb, dt=dt, n_steps=n_steps,
    )
    ret = res.retention
    pristine = bool(np.all(ret >= 1.0))
    tr = res.transient

    # Характеристики ИСПРАВНОЙ машины (одна позиция ротора корректна для симметричной машины).
    # Ущерб от несимметричного повреждения выражаем ИНВАРИАНТНОЙ к положению ротора оценкой по
    # 1-й гармонике ремнантности: снимок «после» в одной позиции даёт неверный знак (K_t якобы
    # растёт), поэтому его сюда НЕ выносим — «после» по моменту/КПД даёт прогонка (потери).
    healthy = machine_characteristics(
        scen, retention=None, i_peak=ipk, turns_per_slot=turns, gamma_elec=gamma,
        T=T_amb, magnet_density=dens,
    )
    ratio = float(res.fundamental_ratio)

    out = {
        "survived": bool(res.survived),
        "runaway": bool(tr.runaway),
        "cascade": bool(tr.magnet_cascade),
        "stop_reason": tr.stop_reason,
        "T_magnet_max": round(float(res.T_magnet_max), 1),
        "fundamental_ratio": round(ratio, 4),
        "kt_drop_est": round(100.0 * res.torque_constant_drop, 2),
        "n_past_knee": int(tr.n_past_knee[-1]),
        "n_mag": int(ret.size),
        "retention_min": round(float(ret.min()), 4),
        "retention_mean": round(float(ret.mean()), 4),
        "pristine": pristine,
        "magnet_mass_g": round(healthy.magnet_mass * 1000.0, 1),
        "healthy": {"torque": round(abs(healthy.torque), 4),
                    "Kt": round(healthy.torque_constant, 5),
                    "Ke": round(healthy.emf_constant, 5),
                    "lambda_m": round(healthy.flux_linkage, 5),
                    "tpm": round(abs(healthy.torque_per_magnet_mass), 3)},
        "trajectory": {
            "t": np.round(tr.times, 2).tolist(),
            "T_magnet": np.round(np.nan_to_num(tr.T_magnet), 1).tolist(),
            "retention_mean": np.round(tr.retention_mean, 4).tolist(),
            "loss_power": np.round(tr.loss_power, 1).tolist(),
        },
    }

    if with_losses:
        cf = SteinmetzCoefficients.m270_35a()
        p_cu = copper_loss_watts(g, i_peak=ipk, turns_per_slot=turns,
                                 slot_fill=slot_fill, T=T_amb)
        n_pos = int(body.get("n_positions", 12))
        # Сетки положений ротора строятся ОДИН раз (gmsh в фон-потоке — interruptible=False +
        # общий лок делают это безопасным) и переиспользуются прогонами «до» и «после».
        angles = electrical_period_angles(params, n_pos, periods=1)
        geoms = build_rotor_geometries(params, angles)
        ck = dict(speed_rpm=rpm, i_peak=ipk, gamma_elec=gamma, turns_per_slot=turns,
                  T=T_amb, n_positions=n_pos, coeffs=cf, geometries=geoms)
        loss_b, sw_b = stator_iron_loss(params, magnet, steel, **ck)
        if pristine:
            loss_a, sw_a = loss_b, sw_b
        else:
            loss_a, sw_a = stator_iron_loss(params, magnet, steel,
                                            damage=RotorDamage(g, ret), **ck)
        eta_b = efficiency(sw_b.torque_mean, rpm, p_cu, loss_b.total)
        eta_a = efficiency(sw_a.torque_mean, rpm, p_cu, loss_a.total)
        out["losses"] = {
            "speed_rpm": rpm, "freq": round(electrical_frequency(params, rpm), 1),
            "copper_W": round(p_cu, 2), "iron_mass_g": round(loss_b.iron_mass * 1000.0, 1),
            "before": {"iron_W": round(loss_b.total, 3), "hyst_W": round(loss_b.hysteresis, 3),
                       "eddy_W": round(loss_b.eddy, 3), "torque_mean": round(abs(sw_b.torque_mean), 4),
                       "ripple": round(100.0 * sw_b.torque_ripple, 1), "eff": round(100.0 * eta_b, 2)},
            "after": {"iron_W": round(loss_a.total, 3), "hyst_W": round(loss_a.hysteresis, 3),
                      "eddy_W": round(loss_a.eddy, 3), "torque_mean": round(abs(sw_a.torque_mean), 4),
                      "ripple": round(100.0 * sw_a.torque_ripple, 1), "eff": round(100.0 * eta_a, 2)},
        }
    return out


@app.post("/api/machine_scenario")
def api_machine_scenario(body: dict = Body(default={})) -> dict:
    """Поставить сценарий S3 (тепловая стойкость магнита) в фон-очередь; вернуть job_id."""
    label = str(body.get("label") or "Тепловая динамика")
    jid = _JM.submit("scenario", label, _do_scenario, dict(body))
    return {"job_id": jid}


@app.get("/api/mesh_defaults")
def api_mesh_defaults() -> dict:
    """Дефолты для UI: геометрия (поля+вид) и регионы посегментной сетки (размер, мм)."""
    return {
        "geometry": [{"name": n, "label": lab, "value": v, "kind": k}
                     for n, lab, v, k in GEOM_UI],
        "regions": [{"name": n, "label": lab, "mm": mm} for n, lab, mm in REGION_UI],
    }


@app.get("/api/spoke_defaults")
def api_spoke_defaults() -> dict:
    """Дефолты формы двигателя (спицевого): параметры чертежа + формы магнита + регионы сетки."""
    return {
        "params": [{"name": n, "label": lab, "value": v, "kind": k} for n, lab, v, k in SPOKE_UI],
        "shapes": [{"value": v, "label": lab} for v, lab in SPOKE_SHAPES],
        "regions": [{"name": n, "label": lab, "mm": mm} for n, lab, mm in REGION_UI],
    }


@app.post("/api/spoke_mesh")
async def api_spoke_mesh(body: dict = Body(default={})) -> dict:
    """Построить спицевой двигатель (gmsh на главном потоке = поток event-loop) и вернуть сцену."""
    try:
        params, mid = _spoke_params_from(dict(body))
        mid = _build_spoke_mesh(params, mid)
    except Exception as e:  # noqa: BLE001 — плохая геометрия → в UI, не 500
        return {"error": str(e)}
    return _mesh_payload(mid)


@app.post("/api/mesh")
async def api_mesh(body: dict = Body(default={})) -> dict:
    """Построить модель (геометрия+сетка, gmsh на главном потоке) и вернуть сцену."""
    try:
        params, mid = _params_from(dict(body))
        mid = _build_mesh(params, mid)   # на потоке event-loop = главный поток (gmsh ОК)
    except Exception as e:  # noqa: BLE001 — плохая геометрия/сетка → в UI, не 500
        return {"error": str(e)}
    return _mesh_payload(mid)


@app.post("/api/solve")
def api_solve(body: dict = Body(default={})) -> dict:
    """Поставить расчёт на выбранной сетке в фон-очередь; вернуть job_id для опроса."""
    label = str(body.get("label") or "Расчёт")
    jid = _JM.submit("solve", label, _do_solve, dict(body))
    return {"job_id": jid}


@app.get("/api/settings")
def api_settings() -> dict:
    """Настройки менеджера: лимит параллельных расчётов, ядра, рекомендация (по 2 ядра на расчёт)."""
    return {"max_parallel": _JM.max_parallel, "cores": _CORES, "recommended": max(1, _CORES // 2)}


@app.post("/api/settings")
def api_set_settings(body: dict = Body(default={})) -> dict:
    if "max_parallel" in body:
        _JM.set_max_parallel(body["max_parallel"])
    return {"max_parallel": _JM.max_parallel, "cores": _CORES, "recommended": max(1, _CORES // 2)}


@app.get("/api/jobs")
def api_jobs() -> dict:
    """Список всех задач (для индикатора и переподключения после перезагрузки)."""
    return {"jobs": _JM.listing(), "max_parallel": _JM.max_parallel}


@app.post("/api/jobs_clear")
def api_jobs_clear() -> dict:
    """Убрать из списка завершённые/ошибочные задачи."""
    _JM.clear_finished()
    return {"ok": True}


@app.get("/api/jobs/{jid}")
def api_job(jid: str) -> dict:
    return _JM.status(jid)


# ---- ЭТАП 4: ПЕРСИСТЕНТНОСТЬ РЕШЁННЫХ ПРОЕКТОВ (полные поля на диск) ----
# Папка на проект: meta.json (мета+геометрия+скаляры поля) + fields.npz (крупные массивы:
# вершины/ячейки/регион сетки + Bx/By по ячейкам). Так «полные поля» хранятся компактно,
# а фронтенд получает единый JSON. Проекты живут между запусками → «Открыть решённый».
_PROJECTS_DIR = Path(__file__).parent / "projects"
_PROJECTS_DIR.mkdir(exist_ok=True)
_ARR_SCENE = ("vertices", "cells", "region")     # крупные массивы сцены → в .npz
_ARR_FIELD = ("Bx", "By")                        # поле по ячейкам → в .npz


def _project_slug(name: str) -> str:
    """ФС-безопасное имя папки: очищенное имя + короткий хеш (разводит коллизии слагов)."""
    cleaned = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in name.strip())
    cleaned = cleaned.replace(" ", "_")[:60] or "проект"
    return cleaned + "-" + hashlib.md5(name.encode("utf-8")).hexdigest()[:6]


@app.get("/api/projects")
def api_projects_list() -> dict:
    """Список решённых проектов (для «Открыть решённый» и проверки уникальности имён)."""
    out = []
    for d in _PROJECTS_DIR.iterdir():
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — битый файл пропускаем
            continue
        out.append({"slug": d.name, "name": m.get("name", ""), "type": m.get("type", ""),
                    "mode": m.get("mode", ""), "scenario": m.get("scenario", ""),
                    "saved_at": m.get("saved_at", 0)})
    out.sort(key=lambda r: r.get("saved_at", 0), reverse=True)
    return {"projects": out}


@app.post("/api/projects")
def api_projects_save(body: dict = Body(default={})) -> dict:
    """Сохранить решённый проект: крупные массивы → fields.npz, остальное → meta.json."""
    name = str(body.get("name") or "проект")
    slug = _project_slug(name)
    folder = _PROJECTS_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)
    scene = dict(body.get("scene") or {})
    field = dict(body.get("field") or {})
    arrays: dict = {}
    for k in _ARR_SCENE:
        if scene.get(k) is not None:
            arrays["scene_" + k] = np.asarray(scene[k])
    for k in _ARR_FIELD:
        if field.get(k) is not None:
            arrays["field_" + k] = np.asarray(field[k], dtype=float)
    np.savez_compressed(folder / "fields.npz", **arrays)
    meta = json.loads(json.dumps(body, ensure_ascii=False))     # глубокая копия
    for k in _ARR_SCENE:
        meta.get("scene", {}).pop(k, None)
    for k in _ARR_FIELD:
        meta.get("field", {}).pop(k, None)
    meta["saved_at"] = time.time()
    meta["version"] = 1
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {"slug": slug, "ok": True}


@app.get("/api/projects/{slug}")
def api_projects_load(slug: str) -> dict:
    """Загрузить проект: собрать meta.json + fields.npz обратно в единый JSON-бандл."""
    folder = _PROJECTS_DIR / slug
    mp = folder / "meta.json"
    if not mp.exists():
        return {"error": "проект не найден"}
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta.setdefault("scene", {})
    meta.setdefault("field", {})
    npz_path = folder / "fields.npz"
    if npz_path.exists():
        with np.load(npz_path) as npz:
            for k in _ARR_SCENE:
                if ("scene_" + k) in npz:
                    meta["scene"][k] = npz["scene_" + k].tolist()
            for k in _ARR_FIELD:
                if ("field_" + k) in npz:
                    meta["field"][k] = npz["field_" + k].tolist()
    return meta


@app.delete("/api/projects/{slug}")
def api_projects_delete(slug: str) -> dict:
    folder = _PROJECTS_DIR / slug
    if folder.exists() and folder.parent == _PROJECTS_DIR:
        shutil.rmtree(folder, ignore_errors=True)
    return {"ok": True}


# ---- ОБЪЕКТНАЯ ПРОИЗВОЛЬНАЯ ГЕОМЕТРИЯ (свободная модель из примитивов) ----

# ---- БИБЛИОТЕКА МАТЕРИАЛОВ (встроенные + свои измеренные магниты, персист на диск) ----
_MATERIALS_PATH = Path(__file__).parent / "materials.json"
_BUILTIN_MAGNETS = {
    "ndfeb": {"name": "NdFeB N42SH (представит.)"},
    "smco": {"name": "SmCo КС25ДЦ (представит.)"},
}
_BUILTIN_STEELS = {"steel": {"name": "M270-35A (представит.)"}}   # 'steel' = дефолтная сталь


def _load_custom_materials() -> dict:
    """Свои магниты с диска (id -> spec). Данные в СИ: Br [Тл], Hcb/Hk/Hcj [А/м]."""
    if not _MATERIALS_PATH.exists():
        return {}
    try:
        return json.loads(_MATERIALS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — битый файл не должен рушить сервер
        return {}


def _magnet_by_id(mid: str):
    """Модель магнита по id: встроенные марки или свой из библиотеки (magnet_from_datasheet)."""
    if mid == "ndfeb":
        return n42sh_magnet((1, 0, 0))
    if mid == "smco":
        return sm2co17_magnet((1, 0, 0))
    spec = _load_custom_materials().get(mid)
    if spec is not None and spec.get("kind", "magnet") == "magnet":
        return magnet_from_datasheet(
            mid, spec.get("name", mid), (1, 0, 0),
            Br=float(spec["Br"]), Hcb=float(spec["Hcb"]), Hk=float(spec["Hk"]),
            Hcj=float(spec["Hcj"]), alpha_Br=float(spec.get("alpha_Br", 0.12)),
            gamma_Hc=float(spec.get("gamma_Hc", 0.6)), T0=float(spec.get("T0", 20.0)),
        )
    raise ValueError(f"неизвестный магнит {mid!r}.")


def _is_steel(mid: str) -> bool:
    if mid in _BUILTIN_STEELS or mid == "m270":
        return True
    spec = _load_custom_materials().get(mid)
    return spec is not None and spec.get("kind") == "steel"


def _steel_by_id(mid: str) -> SteelBHCurve:
    """Кривая стали по id: встроенная M270 ('steel') или своя таблица B(H) из библиотеки."""
    if mid in _BUILTIN_STEELS or mid == "m270":
        return m270_35a_bh_curve()
    spec = _load_custom_materials().get(mid)
    if spec is not None and spec.get("kind") == "steel":
        return SteelBHCurve(curve_id=mid, name=spec.get("name", mid),
                            H_values=np.asarray(spec["H"], dtype=float),
                            B_values=np.asarray(spec["B"], dtype=float))
    raise ValueError(f"неизвестная сталь {mid!r}.")


def _object_material(m: str):
    if m == "air":
        return Air()
    if _is_steel(m):
        return SteelMaterial(_steel_by_id(m))
    return MagnetMaterial(_magnet_by_id(m))     # ndfeb|smco|свой магнит


def _geo_from(o: dict) -> GeoObject:
    """UI-объект (мм/градусы) → GeoObject (м/радианы)."""
    k = str(o.get("kind"))
    up = dict(o.get("params") or {})
    mm = lambda v: float(v) / 1000.0                                   # noqa: E731
    if k == "rect":
        p = {"cx": mm(up["cx"]), "cy": mm(up["cy"]), "w": mm(up["w"]), "h": mm(up["h"]),
             "angle": math.radians(float(up.get("angle", 0.0)))}
    elif k == "circle":
        p = {"cx": mm(up["cx"]), "cy": mm(up["cy"]), "r": mm(up["r"])}
    elif k == "ring":
        p = {"cx": mm(up["cx"]), "cy": mm(up["cy"]), "r_in": mm(up["r_in"]), "r_out": mm(up["r_out"])}
    elif k == "sector":
        p = {"cx": mm(up["cx"]), "cy": mm(up["cy"]), "r_in": mm(up["r_in"]), "r_out": mm(up["r_out"]),
             "a1": math.radians(float(up["a1"])), "a2": math.radians(float(up["a2"]))}
    elif k == "polygon":
        p = {"points": [(mm(x), mm(y)) for x, y in up["points"]]}
    else:
        raise ValueError(f"неизвестный примитив {k!r}.")
    md = o.get("magnet_dir")
    if isinstance(md, (list, tuple)):
        md = (float(md[0]), float(md[1]))
    ms = o.get("mesh_size_mm")
    return GeoObject(name=str(o.get("name", k)), kind=k, params=p,
                     material=_object_material(str(o.get("material", "air"))),
                     current_density=float(o.get("current", 0.0)) or 0.0,
                     magnet_dir=md, mesh_size=(mm(ms) if ms else None),
                     priority=int(o.get("priority", 1)))


def _build_object_model(body: dict) -> str:
    """Собрать Problem2D из объектов тела запроса; кэшировать; вернуть model_id. ⚠ главный поток."""
    objs = [_geo_from(o) for o in (body.get("objects") or [])]
    if not objs:
        raise ValueError("добавьте хотя бы один объект.")
    defm = float(body.get("default_mesh_mm", 2.0)) / 1000.0
    domm = body.get("domain_mesh_mm")
    # Запас домена критичен: граница A_z=0 близко ⇒ поле занижено (0.4 → −28% на аналитике).
    # Дефолт 4.0; дальнее поле мешится грубо (auto_domain), поэтому запас почти бесплатен.
    dom = auto_domain(objs, material=Air(), margin_frac=float(body.get("margin", 4.0)),
                      mesh_size=(float(domm) / 1000.0 if domm else None))
    prob = build_object_problem(objs, dom, default_mesh_size=defm)
    mid = "o" + hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:11]
    _OBJ[mid] = prob
    return mid


def _do_object_solve(body: dict) -> dict:
    """Решить объектную модель (общий solve_problem2d + общий пост). T задаётся на решении."""
    mid = str(body.get("model_id", ""))
    prob = _OBJ.get(mid)
    if prob is None:
        raise ValueError("модель не найдена — постройте заново.")
    prob = replace(prob, T=float(body.get("T", 20.0)))     # T применяем без пересборки сетки
    sol = solve_problem2d(prob, max_iter=60)
    B = sol.field.B_cells
    Bmag = np.hypot(B[:, 0], B[:, 1])
    out = {
        "converged": bool(sol.converged), "iters": int(sol.field.n_iterations),
        "Bx": np.round(B[:, 0], 4).tolist(), "By": np.round(B[:, 1], 4).tolist(),
        "Bmax": round(float(Bmag.max()), 3), "Bmean": round(float(Bmag.mean()), 3),
        "energy": round(float(magnetic_energy(sol, axial_length=0.03)), 4),
    }
    if prob.magnet() is not None and prob.magnet_mask().any() and sol.risk is not None:
        op = operating_point(sol)
        risk = sol.risk
        out.update({
            "Bd_mean": round(float(np.average(op.B_op, weights=op.cell_volume)), 3),
            "Bd_worst": round(float(op.B_op.min()), 3),
            "n_demag": int(risk.n_demagnetized), "n_mag": int(risk.cell_indices.size),
            "demag_frac": round(float(op.volume_fraction_below(op.knee_field)), 4),
        })
    return out


@app.post("/api/object_model")
async def api_object_model(body: dict = Body(default={})) -> dict:
    """Построить свободную объектную модель (gmsh на главном потоке) и вернуть сцену."""
    try:
        mid = _build_object_model(dict(body))
    except Exception as e:  # noqa: BLE001 — плохая геометрия → в UI
        return {"error": str(e)}
    prob = _OBJ[mid]
    reg = np.asarray(prob.cell_region)
    empty_mag = [r.name for rid, r in prob.regions.items()
                 if isinstance(r.material, MagnetMaterial) and int((reg == rid).sum()) == 0]
    warning = ("магнит без ячеек (перекрыт другим объектом или слишком мелкий): "
               + ", ".join(empty_mag)) if empty_mag else None
    return {"mesh_id": mid, "scene": problem_to_scene(prob),
            "n_cells": int(prob.mesh.n_cells),
            "regions": [r.name for r in prob.regions.values()],
            "has_magnet": bool(prob.magnet() is not None and prob.magnet_mask().any()),
            "warning": warning}


@app.post("/api/object_solve")
def api_object_solve(body: dict = Body(default={})) -> dict:
    label = str(body.get("label") or "Расчёт")
    jid = _JM.submit("object_solve", label, _do_object_solve, dict(body))
    return {"job_id": jid}


@app.get("/api/materials")
def api_materials() -> dict:
    """Материалы для списков: магниты и стали (встроенные + свои)."""
    magnets = [{"id": k, "name": v["name"], "builtin": True} for k, v in _BUILTIN_MAGNETS.items()]
    steels = [{"id": k, "name": v["name"], "builtin": True} for k, v in _BUILTIN_STEELS.items()]
    for mid, spec in _load_custom_materials().items():
        entry = {"id": mid, "name": spec.get("name", mid), "builtin": False}
        (steels if spec.get("kind") == "steel" else magnets).append(entry)
    return {"magnets": magnets, "steels": steels}


def _save_material_spec(mid: str, spec: dict) -> None:
    store = _load_custom_materials()
    store[mid] = spec
    _MATERIALS_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


@app.post("/api/materials")
def api_material_save(body: dict = Body(default={})) -> dict:
    """
    Сохранить свой материал. kind='magnet': name + Br[Тл] + Hcb/Hk/Hcj[кА/м] + α_Br,γ_Hc.
    kind='steel': name + points [[H А/м, B Тл], …] (монотонно от нуля, dH/dB ≤ 1/μ₀).
    Валидация — сборкой доменной модели (magnet_from_datasheet / SteelBHCurve).
    """
    kind = str(body.get("kind", "magnet"))
    try:
        name = str(body.get("name", "")).strip()
        if not name:
            return {"error": "нужно имя материала."}
        if kind == "steel":
            pts = list(body["points"])
            H = np.asarray([float(p[0]) for p in pts], dtype=float)
            B = np.asarray([float(p[1]) for p in pts], dtype=float)
            mid = "cust_" + hashlib.sha1(("steel:" + name).encode()).hexdigest()[:8]
            curve = SteelBHCurve(curve_id=mid, name=name, H_values=H, B_values=B)  # валидирует
            spec = {"kind": "steel", "name": name, "H": H.tolist(), "B": B.tolist()}
            _save_material_spec(mid, spec)
            return {"id": mid, "ok": True, "n_points": int(curve.n_points),
                    "B_max": round(float(curve.B_max), 3)}
        spec = {
            "kind": "magnet", "name": name,
            "Br": float(body["Br"]), "Hcb": float(body["Hcb_kA"]) * 1e3,
            "Hk": float(body["Hk_kA"]) * 1e3, "Hcj": float(body["Hcj_kA"]) * 1e3,
            "alpha_Br": float(body.get("alpha_Br", 0.12)),
            "gamma_Hc": float(body.get("gamma_Hc", 0.6)), "T0": float(body.get("T0", 20.0)),
        }
        mid = "cust_" + hashlib.sha1(name.encode()).hexdigest()[:8]
        mg = magnet_from_datasheet(mid, name, (1, 0, 0), Br=spec["Br"], Hcb=spec["Hcb"],
                                   Hk=spec["Hk"], Hcj=spec["Hcj"], alpha_Br=spec["alpha_Br"],
                                   gamma_Hc=spec["gamma_Hc"], T0=spec["T0"])  # валидирует
    except (KeyError, ValueError, TypeError, IndexError) as e:
        return {"error": f"некорректные параметры: {e}"}
    _save_material_spec(mid, spec)
    return {"id": mid, "ok": True, "mu_rec": round(float(mg.mu_rec), 4),
            "temp_limit": round(float(mg.temperature_limit()), 1)}


@app.post("/api/magnet_curve")
def api_magnet_curve(body: dict = Body(default={})) -> dict:
    """Кривая размагничивания B(H) магнита при T (по методике curve_at) + колено/Br(T)/Hk(T)."""
    try:
        m = _magnet_by_id(str(body.get("material", "ndfeb")))
        T = float(body.get("T", 20.0))
        c = m.curve_at(T)                          # перестраивается по методике при T
        Hk = float(m.Hk(T))
        B_knee = float(c.B_of_H(-Hk))
    except Exception as e:  # noqa: BLE001 — T вне диапазона модели и пр. → в UI
        return {"error": str(e)}
    return {
        "H": np.round(c.H_values, 1).tolist(), "B": np.round(c.B_values, 4).tolist(),
        "Br": round(float(m.Br(T)), 4), "Hcb": round(float(m.Hcb(T)), 1),
        "Hk": round(Hk, 1), "B_knee": round(B_knee, 4),
        "mu_rec": round(float(m.mu_rec), 4), "T_limit": round(float(m.temperature_limit()), 1),
    }


@app.post("/api/materials/delete")
def api_material_delete(body: dict = Body(default={})) -> dict:
    store = _load_custom_materials()
    store.pop(str(body.get("id", "")), None)
    _MATERIALS_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="ui")
