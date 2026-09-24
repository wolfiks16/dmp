"""
FastAPI-бэкенд интерфейса: тонкая обёртка над magcore. Пользователь САМ задаёт сетку —
характерный размер элемента ПО РЕГИОНУ (посегментно) — и строит её; затем поле считается
ФОНОВОЙ задачей на этой сетке.

Потоки и gmsh:
  · Поток event-loop только принимает запросы и отвечает; ничего долгого на нём не выполняется.
    Построение сетки (gmsh) — обычные `def`-обработчики, их FastAPI выполняет в рабочих потоках, так
    что пока строится сетка, сервер отвечает на остальное (ход расчёта, предпросмотр, файлы). Раньше
    сетка строилась на потоке event-loop, и на время построения сервер замирал целиком (Л-102).
  · gmsh — один на процесс (глобальное состояние): любой его сеанс идёт под общим замком
    (magcore.mesh.gmsh_session) — и по запросам интерфейса, и внутри прогонок ротора в фоне.
  · Расчёты — фоновые задачи (_JobManager) с очередью; идущую задачу можно отменить: решатели
    проверяют флаг между итерациями (magcore.cancel) и выходят, задача получает статус «cancelled».

Запуск:  python -m uvicorn webapp.server:app --port 8017   (из корня репозитория)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from magcore.cancel import Cancelled, cancel_scope, check as cancel_check
from magcore.domain import magnet_catalog
from webapp import materials_db
from magcore.domain.magnet_model import (
    ks25dts240_magnet,
    magnet_from_datasheet,
    n35_magnet,
    n42sh_magnet,
    sm2co17_magnet,
)
from magcore.domain.steel_curves import (
    SteelBHCurve,
    m270_35a_cogent_bh_curve,
    steel10_bh_curve,
)
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
    iron_loss_from_probe_waveform,
    stator_iron_loss,
)
from magcore.fem2d.machines.pmsm_outrunner import REGION_NAMES, Region
from magcore.fem2d.machines.catalog import MACHINES as _CATALOG
from magcore.fem2d.machines.library import CommutatorWinding
from magcore.fem2d.verification import (
    VerificationReport,
    check_airgap_resolution,
    check_convergence,
    check_energy_balance,
    check_magnet_model_range,
    check_steel_saturation,
    check_time_step,
)
from magcore.fem2d.machines.rotor_sweep import (
    RotorDamage,
    build_rotor_geometries,
    electrical_period_angles,
    sample_B_at_points,
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

from fastapi import Request  # noqa: E402 — приём файла STEP телом запроса (этап 3D-1б)
from fastapi.responses import FileResponse  # noqa: E402 — 3D-режим (этап 3D-5)
from magcore.fem3d import (  # noqa: E402
    CAD_KIND,
    FLUX_MEASURE_T,
    GeoObject3D,
    auto_domain3d,
    build_object_problem3d,
    coenergy,
    demag_summary,
    field_payload,
    flux_loss,
    flux_through_plane,
    magnetic_force_torque,
    new_magnet_flux,
    restore_saved_field,
    solve_nonlinear3d,
)
from magcore.fem3d.export import write_vtu  # noqa: E402
from magcore.fem3d.objects import preview_geometry, step_bodies  # noqa: E402
from magcore.fem3d.scene import (  # noqa: E402
    arrows_payload,
    axis_segment,
    cell_quantities,
    material_kind,
    pack,
    scene_payload,
    section,
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
    # ⚠ Размер элемента в ЗАЗОРЕ должен быть ≤ трети его толщины (при зазоре 1.0 мм это 0.3 мм).
    # Было 1.2 мм — ЭЛЕМЕНТ ТОЛЩЕ ЗАЗОРА: поле в нём не разрешалось, и все прежние расчёты
    # шли на недоразрешённом зазоре. Ловится проверкой «Разрешение зазора» (verification.py).
    ("air_gap", "Зазор", 0.3),
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
                "_cancel": threading.Event(),       # «Отменить»: решатель выйдет на ближайшей точке отмены
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
            with _limit_threads(per), cancel_scope(rec["_cancel"]):
                rec["result"] = rec["_fn"](rec["_body"])
            rec["status"] = "done"
        except Cancelled:
            rec["status"] = "cancelled"
        except Exception as e:                      # noqa: BLE001 — перегрев/данные → в UI
            rec["error"] = str(e)
            rec["status"] = "error"
        finally:
            rec["finished"] = time.time()
            with self.lock:
                self.running.discard(jid)
                self._pump()

    def cancel(self, jid: str) -> dict:
        """Отменить задачу: из очереди — сразу; идущую — флагом, решатель выйдет между итерациями.
        Готовую или уже отменённую не трогаем (успела закончиться — результат остаётся)."""
        with self.lock:
            rec = self.jobs.get(jid)
            if rec is None:
                return {"status": "unknown"}
            if rec["status"] == "queued":
                self.queue.remove(jid)
                rec["status"] = "cancelled"
                rec["finished"] = time.time()
            elif rec["status"] == "running":
                rec["_cancel"].set()
                rec["status"] = "cancelling"
            return {"status": rec["status"]}

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
                        if r["status"] in ("done", "error", "cancelled")]:
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
    # Адаптивный размер в ЗАЗОРЕ (как для спицевой): элемент не толще gap/5, иначе поле
    # в зазоре не разрешается. Пользовательское значение = ВЕРХНЯЯ граница.
    # ⚠ geom и spec здесь в МЕТРАХ (СИ): сравнение вести в метрах, без множителя 1e3.
    spec = dict(spec)
    spec["air_gap"] = min(spec["air_gap"], float(geom["air_gap"]) / 5.0)
    params = OutrunnerPMSMParams(
        mesh_size=max(spec.values()), mesh_size_by_region=dict(spec), **geom
    )
    key = ("|".join(f"{k}:{geom[k]:.6g}" for k in sorted(geom))
           + "#" + "|".join(f"{k}:{spec[k]:.6g}" for k in sorted(spec)))
    return params, hashlib.sha1(key.encode()).hexdigest()[:12]


def _build_mesh(params: OutrunnerPMSMParams, mid: str) -> str:
    """Построить (или взять из кэша) геометрию+сетку под params. Возвращает mesh_id.
    Может бросить ValueError (плохая геометрия). gmsh — под общим замком (magcore.mesh.gmsh_session)."""
    if mid in _GEOM:
        return mid
    g = build_outrunner_spm_pmsm(params)
    # Трёхфазная раскладка есть только у машин с числом пазов, кратным 3. Коллекторная
    # (щёточный ДПТ, напр. ДП25 с 13 пазами) её НЕ имеет — раскладка None. Для поля
    # открытой цепи (магниты, i=0) и коллекторной K_e = p·Z·Φ/(2πa)·k_скоса она не нужна.
    lay = (star_of_slots_layout(g.params.n_slots, g.params.n_poles)
           if g.params.n_slots % 3 == 0 else None)
    _GEOM[mid] = (g, lay, m270_35a_cogent_bh_curve())
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
    # АДАПТИВНЫЙ РАЗМЕР В ЗАЗОРЕ. Фиксированное число тут не работает в принципе: у разных
    # машин зазор разный (у спицевой по умолчанию 0.20 мм, у outrunner ~1 мм). Если элемент
    # толще трети зазора, поле в нём не разрешается и все результаты искажаются — это ловится
    # проверкой «Разрешение зазора», но лучше не допускать. Пользовательское значение
    # трактуется как ВЕРХНЯЯ граница; фактически применённое возвращается в ответе.
    gap_mm = max((float(kw["D_magnet_in_mm"]) - float(kw["D_tooth_out_mm"])) / 2.0, 1e-6)
    # Делитель 5, а не 3: gmsh трактует размер как ЦЕЛЬ и фактически даёт элементы крупнее
    # (при цели gap/3.5 измерялось лишь 2.7 элемента поперёк). Запас берём с проверкой.
    gap_target = gap_mm / 5.0
    sizes["air_gap"] = min(sizes["air_gap"], gap_target)
    kw["mesh_size_by_region"] = sizes
    kw["mesh_size_mm"] = min(sizes.values())
    params = SpokeMotorParams(**kw)
    key = "spoke#" + "|".join(f"{k}:{v}" for k, v in sorted(
        (k, (v.value if isinstance(v, MagnetShape) else v)) for k, v in kw.items() if k != "mesh_size_by_region"))
    key += "#" + "|".join(f"{k}:{sizes[k]:.4g}" for k in sorted(sizes))
    return params, "sp" + hashlib.sha1(key.encode()).hexdigest()[:10]


def _build_spoke_mesh(params: SpokeMotorParams, mid: str) -> str:
    """Построить (или взять из кэша) спицевую геометрию (gmsh — под общим замком)."""
    if mid in _GEOM:
        return mid
    g = build_spoke_pmsm(params)
    lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    _GEOM[mid] = (g, lay, m270_35a_cogent_bh_curve())
    scen = MachineScenario(geometry=g, magnet=n42sh_magnet((1, 0, 0)), steel=_GEOM[mid][2],
                           layout=lay)
    _SCENE[mid] = problem_to_scene(scen.to_problem())
    return mid


# Прогрев: построить модель по умолчанию при импорте (быстрый первый показ).
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
    # ТИП МАШИНЫ (если выбран пресет): задаёт схему обмотки и, значит, ФОРМУЛУ K_e/K_t.
    # Трёхфазная: K_e = p·λ_m, K_t = 1.5·K_e. Коллекторная: K_e = p·Z·Φ/(2πa)·k_скоса, K_t = K_e.
    # Геометрия при этом берётся из полей интерфейса — тип и геометрия независимы.
    machine_key = str(body.get("machine", "") or "")
    winding = None
    if machine_key and machine_key in _CATALOG:
        winding = _CATALOG[machine_key]().winding
    # Реакцию якоря вводим только через ТРЁХФАЗНУЮ раскладку. У коллекторной машины её нет
    # (lay=None): трёхфазный ток к ней неприменим, поэтому считаем поле ОТКРЫТОЙ ЦЕПИ (i=0).
    # Этого достаточно для валидации по K_e (коллекторная формула = магнитный поток на полюс).
    i_peak = float(body.get("i_peak", 0.0))
    commutator_open_circuit = lay is None and i_peak != 0.0
    if lay is None:
        i_peak = 0.0
    # Метод Ньютона (дефолт solve_problem2d): сходится за ~20 итераций НА ЛЮБОЙ плотности,
    # без подбора релаксации под сетку. max_iter с запасом.
    sol = scen.solve(
        T=float(body.get("T", 20.0)),
        i_peak=i_peak,
        gamma_elec=np.deg2rad(float(body.get("gamma_deg", 0.0))),
        turns_per_slot=float(body.get("turns", 40.0)),
        max_iter=60,
    )
    B = sol.field.B_cells
    Bmag = np.hypot(B[:, 0], B[:, 1])
    op = scen.operating_point(sol)
    risk = sol.risk
    # БЛОК ПРОВЕРОК: результат выдаётся вместе с доказательством, что ему можно верить.
    resid = (sol.field.rel_change_history[-1] if getattr(sol.field, "rel_change_history", None)
             else 0.0)
    report = VerificationReport([
        check_convergence(bool(sol.converged), int(sol.field.n_iterations), float(resid)),
        check_airgap_resolution(g.mesh, g.mask(Region.AIR_GAP), _gap_thickness(g.params)),
        check_steel_saturation([steel]),
    ])
    out_machine = {}
    if winding is not None:
        ke = float(winding.emf_constant(g, sol.field.a))
        kt = float(winding.torque_constant(g, sol.field.a))
        # ⚠ kV трёхфазной машины ≠ 60/(2π·K_e): K_e здесь — амплитуда ФАЗНОЙ ЭДС, а
        #   паспортное kV относят к напряжению ШИНЫ. Прежняя формула завышала kV в π/2
        #   раза (аудит 2026-09-03). Для коллекторной машины формула строга.
        from magcore.fem2d.machines.conventions import kv_from_ke
        kv = (None if ke <= 0 else
              (kv_from_ke(ke) if winding.kind == "three_phase" else 60.0 / (2.0 * math.pi * ke)))
        out_machine = {
            "machine": machine_key, "winding": winding.kind,
            "Ke": round(ke, 5), "Kt": round(kt, 5),
            "kV": None if kv is None else round(kv, 1),
            "kV_convention": ("от шины, шеститактный регулятор"
                              if winding.kind == "three_phase" else "коллекторная"),
        }
        if machine_key == "dp25":                      # сверка с паспортом ТУ
            out_machine["Ke_nameplate"] = 0.02946
            out_machine["Ke_dev_pct"] = round(100.0 * (ke - 0.02946) / 0.02946, 1)
        if winding.kind == "commutator":
            out_machine["note"] = (
                "коллекторная машина: поле открытой цепи (магниты), трёхфазная реакция "
                "якоря не моделируется" + (" — заданный ток проигнорирован"
                                           if commutator_open_circuit else ""))

    return {
        "checks": report.to_dict(),
        **out_machine,
        "converged": bool(sol.converged),
        "iters": int(sol.field.n_iterations),
        "Bx": np.round(B[:, 0], 4).tolist(),
        "By": np.round(B[:, 1], 4).tolist(),
        # A_z в узлах [Вб/м] — для силовых линий: они и есть линии уровня A_z, между соседними линиями
        # одинаковый поток на единицу длины машины (этап 3D-7, то же в 2D).
        "A": np.round(sol.field.a, 10).tolist(),
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
        # Поячеечная карта риска: глобальные индексы ячеек магнита + рабочее поле H_op (кА/м) +
        # колено H_knee(T). Фронтенд красит магнит: H_op у колена ⇒ красный, много выше ⇒ зелёный.
        "demag_cells": op.cell_indices.astype(int).tolist(),
        "demag_hop_kA": np.round(op.H_op / 1e3, 1).tolist(),
        "demag_knee_kA": round(float(op.knee_field) / 1e3, 1),
    }


def _do_torque_sweep(body: dict) -> dict:
    """Прогонка ротора спицевого двигателя: момент(угол) + K_t/K_e/λ_m. Тяжёлый фон-джоб —
    геометрия ПЕРЕСТРАИВАЕТСЯ на каждом положении (конформная сетка ⇒ пульсации физичны).
    Ток едет вместе с ротором: γ_abs = γ + p·α, иначе средний момент вышел бы нулём."""
    params, _ = _spoke_params_from(dict(body))
    magnet = _magnet_by_id(str(body.get("material", "ndfeb")))
    steel = _steel_by_id(str(body.get("steel", "steel")))
    T = float(body.get("T", 20.0))
    i_peak = float(body.get("i_peak", 0.0))
    gamma = math.radians(float(body.get("gamma_deg", 0.0)))
    turns = float(body.get("turns", 40.0))
    n_pos = max(4, min(int(body.get("n_pos", 12)), 48))
    p = params.n_poles // 2
    angles = np.arange(n_pos) * (2.0 * math.pi / p) / n_pos     # один эл. период, равномерно
    lay = star_of_slots_layout(params.n_slots, params.n_poles)
    have_current = i_peak != 0.0 and turns != 0.0
    torque = np.empty(n_pos)
    lam_a = np.full(n_pos, np.nan)
    conv = np.empty(n_pos, dtype=bool)
    for i, a in enumerate(angles):
        cancel_check()                               # отмена расчёта — между положениями ротора
        geo = build_spoke_pmsm(replace(params, rotor_angle=float(a)))
        scen = MachineScenario(geometry=geo, magnet=magnet, steel=steel, layout=lay)
        g_abs = gamma + p * float(a)
        sol = scen.solve(T=T, i_peak=i_peak, gamma_elec=g_abs, turns_per_slot=turns, max_iter=60)
        torque[i] = float(scen.torque(sol))
        conv[i] = bool(sol.converged)
        sol_nl = (scen.solve(T=T, i_peak=0.0, gamma_elec=g_abs, turns_per_slot=turns, max_iter=60)
                  if have_current else sol)                     # х.х. для потокосцепления ПМ
        lam_a[i] = float(scen.phase_flux_linkage(sol_nl, turns_per_slot=turns)[0])
    t_mean = float(np.mean(torque))
    t_span = float(np.max(torque) - np.min(torque))
    lam_m = float(2.0 * np.abs(np.sum(lam_a * np.exp(-1j * p * angles))) / n_pos)  # 1-я гармоника
    return {
        "angles_deg": np.round(np.degrees(angles), 2).tolist(),
        "torque": np.round(torque, 4).tolist(),
        "torque_mean": round(t_mean, 4),
        "torque_span": round(t_span, 4),
        "torque_ripple": (round(t_span / abs(t_mean), 4) if abs(t_mean) > 1e-4 else None),
        "torque_min": round(float(np.min(torque)), 4),
        "torque_max": round(float(np.max(torque)), 4),
        "lam_m": round(lam_m, 6),
        "Ke": round(float(p * lam_m), 5),
        "Kt": round(float(1.5 * p * lam_m), 4),
        "n_pos": n_pos, "p": int(p), "loaded": bool(have_current),
        "all_converged": bool(np.all(conv)),
    }


def _do_loss_sweep(body: dict) -> dict:
    """Разбивка потерь и КПД спицевого двигателя в рабочей точке (ток + скорость) — фон-джоб.

    Прогонка ротора на один эл. период с током: волна B в НЕПОДВИЖНЫХ точках железа статора ⇒
    потери в железе по Штейнмецу (гистерезис по пику + вихревые по фактической dB/dθ). Медь =
    ∫ρ(T)·J² по обмотке. Выход = |M_ср|·ω. η = P_вых/(P_вых+медь+железо)."""
    params, _ = _spoke_params_from(dict(body))
    magnet = _magnet_by_id(str(body.get("material", "ndfeb")))
    steel = _steel_by_id(str(body.get("steel", "steel")))
    T = float(body.get("T", 20.0))
    i_peak = float(body.get("i_peak", 0.0))
    gamma = math.radians(float(body.get("gamma_deg", 0.0)))
    turns = float(body.get("turns", 40.0))
    slot_fill = float(body.get("slot_fill", 0.45))
    rpm = max(1.0, float(body.get("rpm", 3000.0)))
    n_pos = max(6, min(int(body.get("n_pos", 24)), 48))
    p = params.n_poles // 2
    angles = np.arange(n_pos) * (2.0 * math.pi / p) / n_pos
    lay = star_of_slots_layout(params.n_slots, params.n_poles)
    # неподвижные пробы в железе статора (зубцы + ярмо эталонной геометрии)
    geo0 = build_spoke_pmsm(params)
    stator = np.where(np.isin(geo0.region, [int(Region.STATOR_YOKE), int(Region.TOOTH)]))[0]
    pts = np.array([geo0.mesh.cell_centroid(int(c)) for c in stator], dtype=float)
    areas = np.array([geo0.mesh.cell_area(int(c)) for c in stator], dtype=float)
    torque = np.empty(n_pos)
    probe_B = np.empty((n_pos, pts.shape[0], 2))
    conv = np.empty(n_pos, dtype=bool)
    for i, a in enumerate(angles):
        cancel_check()                               # отмена расчёта — между положениями ротора
        geo = build_spoke_pmsm(replace(params, rotor_angle=float(a)))
        scen = MachineScenario(geometry=geo, magnet=magnet, steel=steel, layout=lay)
        sol = scen.solve(T=T, i_peak=i_peak, gamma_elec=gamma + p * float(a),
                         turns_per_slot=turns, max_iter=60)
        torque[i] = float(scen.torque(sol))
        conv[i] = bool(sol.converged)
        probe_B[i] = sample_B_at_points(geo.mesh, sol.field.B_cells, pts)
    t_mean = float(np.mean(torque))
    freq = p * rpm / 60.0
    iron = iron_loss_from_probe_waveform(
        probe_B, areas, freq=freq, axial_length=params.axial_length,
        coeffs=SteinmetzCoefficients.m270_35a_cogent(),
    )
    p_cu = float(copper_loss_watts(geo0, i_peak=i_peak, turns_per_slot=turns,
                                   slot_fill=slot_fill, T=T))
    p_out = abs(t_mean) * (2.0 * math.pi * rpm / 60.0)
    p_fe = float(iron.total)
    p_in = p_out + p_cu + p_fe
    return {
        "rpm": round(rpm, 0), "freq": round(freq, 1), "n_pos": n_pos,
        "torque_mean": round(t_mean, 4),
        "p_out": round(p_out, 2), "p_copper": round(p_cu, 2), "p_iron": round(p_fe, 2),
        "p_iron_hyst": round(float(iron.hysteresis), 2),
        "p_iron_eddy": round(float(iron.eddy), 2),
        "p_loss_total": round(p_cu + p_fe, 2), "p_in": round(p_in, 2),
        "efficiency": round(float(p_out / p_in) if p_in > 0 else 0.0, 4),
        "iron_mass": round(float(iron.iron_mass), 4),
        "all_converged": bool(np.all(conv)),
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
    # Сценарий S3 вводит ТРЁХФАЗНЫЙ ток статора (реакция якоря). У коллекторной машины
    # (lay=None) реакция якоря устроена иначе и этой моделью не описывается — честно
    # отказываемся, а не подставляем неверный трёхфазный ток. K_e ей даёт магнитостатика.
    if lay is None:
        raise ValueError(
            "Тепловой сценарий S3 задаёт трёхфазный ток статора и пока не поддержан для "
            "коллекторной машины (щёточный ДПТ): её реакция якоря устроена иначе. Для такой "
            "машины доступен магнитостатический расчёт поля и постоянной K_e.")
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

    # n_steps = ВЕРХНИЙ предел; steady_tol>0 останавливает по выходу на установившуюся T.
    steady_tol = float(body.get("steady_tol", 0.05))
    res = run_machine_thermal_demag(
        scen, i_peak=ipk, turns_per_slot=turns, gamma_elec=gamma, slot_fill=slot_fill,
        h=h, T_amb=T_amb, dt=dt, n_steps=n_steps,
        steady_tol=(steady_tol if steady_tol > 0 else None),
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

    # БЛОК ПРОВЕРОК связанного расчёта. Разгон/каскад — это РЕЗУЛЬТАТ, а не провал проверки,
    # поэтому в проверки они не попадают (их видно в survived/stop_reason). Проверяем то,
    # что делает числа НЕДОСТОВЕРНЫМИ: сходимость, энергобаланс, шаг по времени, разрешение
    # зазора, пригодность кривых стали и выход магнита за область достоверности его модели.
    em_res = float(tr.em_residual[-1]) if getattr(tr, "em_residual", None) is not None and len(
        tr.em_residual) else 0.0
    em_it = int(tr.em_iterations[-1]) if getattr(tr, "em_iterations", None) is not None and len(
        tr.em_iterations) else 0
    checks = [
        check_convergence(bool(tr.em_converged), em_it, em_res),
        check_energy_balance(tr.stored_energy, tr.loss_power, tr.outflow, dt),
        check_time_step(dt),
        check_airgap_resolution(g.mesh, g.mask(Region.AIR_GAP), _gap_thickness(g.params)),
        check_steel_saturation([steel]),
    ]
    if np.isfinite(res.T_magnet_max):
        checks.append(check_magnet_model_range(float(res.T_magnet_max),
                                               float(magnet.temperature_limit())))
    report = VerificationReport(checks)

    out = {
        "checks": report.to_dict(),
        "survived": bool(res.survived),
        "runaway": bool(tr.runaway),
        "cascade": bool(tr.magnet_cascade),
        "stop_reason": tr.stop_reason,
        "T_magnet_max": round(float(res.T_magnet_max), 1),
        "fundamental_ratio": round(ratio, 4),
        "kt_drop_est": round(100.0 * res.torque_constant_drop, 2),
        "n_past_knee": int(tr.n_past_knee[-1]),
        "past_knee_fraction": round(float(tr.past_knee_fraction[-1]), 4),     # доля площади магнита
        "n_mag": int(ret.size),
        "retention_min": round(float(ret.min()), 4),
        "retention_mean": round(float(tr.retention_mean[-1]), 4),             # с весом площади (Л-104)
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

    # ТЕПЛОВОЕ ПОЛЕ: узловое поле T → поячеечно (в порядке ячеек сетки = порядок BX), чтобы
    # фронтенд отрисовал карту поверх той же геометрии. Кадры анимации — равномерная выборка
    # истории нагрева (≤24 кадра), финальный кадр = установившееся поле.
    cells = g.mesh.cells
    T_hist = np.asarray(tr.T_hist, dtype=float)                 # (n+1, ndofs)
    nfr = T_hist.shape[0]
    step = max(1, math.ceil(nfr / 24))
    sel = list(range(0, nfr, step))
    if sel[-1] != nfr - 1:
        sel.append(nfr - 1)
    frames = [np.round(T_hist[s][cells].mean(axis=1), 1).tolist() for s in sel]
    T_final = np.asarray(frames[-1], dtype=float)
    out["T_cells"] = frames[-1]
    out["temp_frames"] = frames
    out["temp_frame_t"] = [round(float(tr.times[s]), 1) for s in sel]
    out["T_min"] = round(float(T_final.min()), 1)
    out["T_max"] = round(float(T_final.max()), 1)
    out["T_amb"] = round(T_amb, 1)
    out["steady"] = bool("установившийся" in (tr.stop_reason or ""))
    dT_end = (float(tr.T_max[-1]) - float(tr.T_max[-2])) if tr.T_max.size >= 2 else 0.0
    out["dTdt_end"] = round(dT_end / dt, 3) if dt else 0.0

    if with_losses:
        cf = SteinmetzCoefficients.m270_35a_cogent()
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
def api_spoke_mesh(body: dict = Body(default={})) -> dict:
    """Построить спицевой двигатель и вернуть сцену (рабочий поток: сервер не замирает на время сетки)."""
    try:
        params, mid = _spoke_params_from(dict(body))
        mid = _build_spoke_mesh(params, mid)
    except Exception as e:  # noqa: BLE001 — плохая геометрия → в UI, не 500
        return {"error": str(e)}
    out = _mesh_payload(mid)
    g = _GEOM[mid][0]
    gap_mm = (float(params.D_magnet_in_mm) - float(params.D_tooth_out_mm)) / 2.0
    used = float(params.mesh_size_by_region.get("air_gap", 0.0))
    out["airgap"] = {"gap_mm": round(gap_mm, 3), "mesh_mm": round(used, 4),
                     "elements_across": round(gap_mm / used, 1) if used > 0 else None}
    return out


@app.post("/api/mesh")
def api_mesh(body: dict = Body(default={})) -> dict:
    """Построить модель (геометрия+сетка) и вернуть сцену (рабочий поток: сервер не замирает)."""
    try:
        params, mid = _params_from(dict(body))
        mid = _build_mesh(params, mid)
    except Exception as e:  # noqa: BLE001 — плохая геометрия/сетка → в UI, не 500
        return {"error": str(e)}
    out = _mesh_payload(mid)
    gap_mm = float(params.air_gap) * 1e3
    # mesh_size_by_region у OutrunnerPMSMParams — в МЕТРАХ (СИ); в отчёт переводим в мм.
    used_mm = float(params.mesh_size_by_region.get("air_gap", 0.0)) * 1e3
    out["airgap"] = {"gap_mm": round(gap_mm, 3), "mesh_mm": round(used_mm, 4),
                     "elements_across": round(gap_mm / used_mm, 1) if used_mm > 0 else None}
    return out


@app.post("/api/solve")
def api_solve(body: dict = Body(default={})) -> dict:
    """Поставить расчёт на выбранной сетке в фон-очередь; вернуть job_id для опроса."""
    label = str(body.get("label") or "Расчёт")
    jid = _JM.submit("solve", label, _do_solve, dict(body))
    return {"job_id": jid}


@app.post("/api/torque_sweep")
def api_torque_sweep(body: dict = Body(default={})) -> dict:
    """Момент от угла ротора (спицевой двигатель) — тяжёлый фон-джоб через менеджер задач."""
    label = str(body.get("label") or "Момент от угла")
    jid = _JM.submit("torque_sweep", label, _do_torque_sweep, dict(body))
    return {"job_id": jid}


@app.post("/api/loss_sweep")
def api_loss_sweep(body: dict = Body(default={})) -> dict:
    """Разбивка потерь и КПД (спицевой двигатель) — тяжёлый фон-джоб через менеджер задач."""
    label = str(body.get("label") or "Потери и КПД")
    jid = _JM.submit("loss_sweep", label, _do_loss_sweep, dict(body))
    return {"job_id": jid}


# Настройки интерфейса, которые переживают перезапуск сервера и одни для всех браузеров (сейчас — тема
# оформления по умолчанию). Файл пользовательский, как materials.db, в git не входит; нет файла или он
# испорчен — встроенные значения.
_UI_SETTINGS_PATH = Path(__file__).parent / "settings.json"
UI_THEMES = ("white", "grey", "color")          # «Белая», «Серая», «Цветная» — см. static/themes.css
DEFAULT_THEME = "color"


def _ui_settings() -> dict:
    try:
        data = json.loads(_UI_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    theme = data.get("default_theme") if isinstance(data, dict) else None
    return {"default_theme": theme if theme in UI_THEMES else DEFAULT_THEME}


def _save_ui_settings(settings: dict) -> None:
    tmp = _UI_SETTINGS_PATH.with_name(_UI_SETTINGS_PATH.name + ".tmp")
    tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_UI_SETTINGS_PATH)                 # замена целиком: оборванная запись не портит файл


def _settings_view() -> dict:
    return {"max_parallel": _JM.max_parallel, "cores": _CORES, "recommended": max(1, _CORES // 2),
            **_ui_settings(), "themes": list(UI_THEMES)}


@app.get("/api/settings")
def api_settings() -> dict:
    """Настройки: лимит параллельных расчётов, ядра, рекомендация (по 2 ядра на расчёт), тема по умолчанию."""
    return _settings_view()


@app.post("/api/settings")
def api_set_settings(body: dict = Body(default={})) -> dict:
    if "default_theme" in body:
        theme = body["default_theme"]
        if theme not in UI_THEMES:
            return {**_settings_view(), "error": f"неизвестная тема {theme!r}: допустимы {', '.join(UI_THEMES)}."}
        _save_ui_settings({**_ui_settings(), "default_theme": theme})
    if "max_parallel" in body:
        _JM.set_max_parallel(body["max_parallel"])
    return _settings_view()


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


@app.post("/api/jobs/{jid}/cancel")
def api_job_cancel(jid: str) -> dict:
    """Отменить задачу. Ответ — новый статус: cancelled (была в очереди), cancelling (идёт, выйдет между
    итерациями), done/error (уже закончилась — не трогаем), unknown."""
    return _JM.cancel(jid)


# ---- ОБЪЕКТНАЯ ПРОИЗВОЛЬНАЯ ГЕОМЕТРИЯ (свободная модель из примитивов) ----

# ---- БИБЛИОТЕКА МАТЕРИАЛОВ (встроенные + свои измеренные магниты, персист на диск) ----
_MATERIALS_PATH = Path(__file__).parent / "materials.json"
_BUILTIN_MAGNETS = {
    "ndfeb": {"name": "NdFeB N42SH (представит.)", "family": "NdFeB"},
    "n35": {"name": "NdFeB N35 (представит.)", "family": "NdFeB"},
    "smco": {"name": "SmCo КС25ДЦ (представит.)", "family": "SmCo"},
    "ks25dts240": {"name": "SmCo КС25ДЦ-240 (представит.)", "family": "SmCo"},
}
# ⚠ 'steel' ТЕПЕРЬ = кривая по DATASHEET Cogent. Прежняя «представительная» была в ~2.4 раза
# МЯГЧЕ реального листа (700 против 1700 А/м при 1.5 Тл) и занижала насыщение во всех
# расчётах. Идентификатор сохранён (совместимость сохранённых расчётов), данные исправлены.
_BUILTIN_STEELS = {
    "steel": {"name": "M270-35A (datasheet Cogent)"},        # дефолтная электротехническая
    "steel10": {"name": "Сталь 10 (ГОСТ 1050, данные изделия)"},
}


def _load_custom_materials() -> dict:
    """Свои материалы из SQLite (id -> spec). Данные в СИ: Br [Тл], Hcb/Hk/Hcj [А/м].
    Старый materials.json втягивается автоматически при первом обращении."""
    try:
        return materials_db.load_all()
    except Exception:  # noqa: BLE001 — сбой хранилища не должен рушить сервер
        return {}


def _magnet_by_id(mid: str):
    """Модель магнита по id: встроенные марки или свой из библиотеки (magnet_from_datasheet)."""
    if mid == "ndfeb":
        return n42sh_magnet((1, 0, 0))
    if mid == "n35":
        return n35_magnet((1, 0, 0))
    if mid == "smco":
        return sm2co17_magnet((1, 0, 0))
    if mid == "ks25dts240":
        return ks25dts240_magnet((1, 0, 0))
    spec = _load_custom_materials().get(mid)
    if spec is not None and spec.get("kind", "magnet") == "magnet":
        return magnet_from_datasheet(
            mid, spec.get("name", mid), (1, 0, 0),
            Br=float(spec["Br"]), Hcb=float(spec["Hcb"]), Hk=float(spec["Hk"]),
            Hcj=float(spec["Hcj"]), alpha_Br=float(spec.get("alpha_Br", 0.12)),
            gamma_Hc=float(spec.get("gamma_Hc", 0.6)), T0=float(spec.get("T0", 20.0)),
        )
    try:                                   # марка из справочного каталога
        return magnet_catalog.to_magnet(mid, (1, 0, 0))
    except KeyError:
        pass
    raise ValueError(f"неизвестный магнит {mid!r}.")


def _is_steel(mid: str) -> bool:
    if mid in _BUILTIN_STEELS or mid == "m270":
        return True
    spec = _load_custom_materials().get(mid)
    return spec is not None and spec.get("kind") == "steel"


def _steel_by_id(mid: str) -> SteelBHCurve:
    """Кривая стали по id: встроенная M270 ('steel') или своя таблица B(H) из библиотеки."""
    if mid == "steel10":
        return steel10_bh_curve()
    if mid in _BUILTIN_STEELS or mid == "m270":
        return m270_35a_cogent_bh_curve()
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
                     priority=int(o.get("priority", 1)),
                     magnet_angle=math.radians(float(o.get("magnet_angle") or 0.0)))


def _build_object_model(body: dict) -> str:
    """Собрать Problem2D из объектов тела запроса; кэшировать; вернуть model_id (gmsh — под общим замком)."""
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
        "A": np.round(sol.field.a, 10).tolist(),           # узловой A_z — силовые линии (линии уровня A_z)
        "Bmax": round(float(Bmag.max()), 3), "Bmean": round(float(Bmag.mean()), 3),
        "energy": round(float(magnetic_energy(sol, axial_length=0.03)), 4),
    }
    if prob.magnet_mask().any() and sol.risk is not None:
        op = operating_point(sol)
        risk = sol.risk
        # Колено — своей марки у каждой ячейки: в модели могут быть магниты разных марок.
        past = op.H_op < op.knee_field_cells
        reg = np.asarray(prob.cell_region)[op.cell_indices]
        magnets = []
        for rid, r in sorted(prob.regions.items()):
            s = reg == rid
            if not isinstance(r.material, MagnetMaterial) or not s.any():
                continue
            v = op.cell_volume[s]
            magnets.append({"name": r.name,
                            "past_knee": round(float(v[past[s]].sum() / v.sum()), 4),
                            "Bd_mean": round(float(np.average(op.B_op[s], weights=v)), 3),
                            "knee_kA": round(float(op.knee_field_cells[s][0]) / 1e3, 1)})
        out.update({
            "Bd_mean": round(float(np.average(op.B_op, weights=op.cell_volume)), 3),
            "Bd_worst": round(float(op.B_op.min()), 3),
            "n_demag": int(risk.n_demagnetized), "n_mag": int(risk.cell_indices.size),
            "demag_frac": round(float(op.fraction_past_knee()), 4),
            "demag_cells": op.cell_indices.astype(int).tolist(),
            "demag_hop_kA": np.round(op.H_op / 1e3, 1).tolist(),
            "demag_knee_cells_kA": np.round(op.knee_field_cells / 1e3, 1).tolist(),
            # одно колено на модель — только у модели из одной марки (его читают файлы до 24.09.2026)
            "demag_knee_kA": None if op.knee_field is None else round(float(op.knee_field) / 1e3, 1),
            "magnets": magnets,
        })
    return out


@app.post("/api/object_model")
def api_object_model(body: dict = Body(default={})) -> dict:
    """Построить свободную объектную модель и вернуть сцену (рабочий поток: сервер не замирает)."""
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
            "has_magnet": bool(prob.magnet_mask().any()),
            "warning": warning}


@app.post("/api/object_solve")
def api_object_solve(body: dict = Body(default={})) -> dict:
    label = str(body.get("label") or "Расчёт")
    jid = _JM.submit("object_solve", label, _do_object_solve, dict(body))
    return {"job_id": jid}


# ---- 3D: СВОБОДНАЯ ГЕОМЕТРИЯ (этап 3D-5, magcore.fem3d) ----
# Объекты — в мм и градусах, как в 2D; сетка и решение — в СИ. Сетку строит gmsh (без перехвата
# сигналов — годится любой поток), решение — фоновая задача. Кэш — последние модели и их решения.
_OBJ3D: dict = {}          # model_id -> Problem3D
_SOL3D: dict = {}          # model_id -> ScalarField3D (последнее решение модели)
_NEWFLUX3D: dict = {}      # (model_id, граница) -> поток нового магнита при 20 °C: от события не зависит (Л-104)
_OBJ3D_MAX = 4             # моделей в памяти (сетка в полмиллиона ячеек — порядка 0,2 ГБ)
_CACHE3D_LOCK = threading.RLock()   # кэш правят и запросы в рабочих потоках, и фоновые задачи
_KIND3D = {"box": ("lx", "ly", "lz"), "cylinder": ("r", "h"), "tube": ("r_in", "r_out", "h"),
           "tube_sector": ("r_in", "r_out", "h", "a1", "a2"), "sphere": ("r",), "prism": ("h", "points"),
           CAD_KIND: ("file_id", "body")}
_STEP_MAX_BYTES = 64 * 1024 * 1024        # больше — скорее сборка целиком, чем электромагнитная модель


def _step_dir():
    """Папка STEP-файлов сервера (этап 3D-1б): файл лежит под SHA-256 содержимого — одинаковый хранится раз."""
    import tempfile
    from pathlib import Path

    d = Path(tempfile.gettempdir()) / "magfield_step"
    d.mkdir(exist_ok=True)
    return d


def _step_path(file_id) -> str:
    fid = str(file_id or "")
    if len(fid) != 64 or any(c not in "0123456789abcdef" for c in fid):
        raise ValueError("неверный идентификатор файла STEP.")
    path = _step_dir() / f"{fid}.step"
    if not path.is_file():
        raise ValueError("файла STEP нет на сервере — загрузите его заново.")
    return str(path)


def _step_params(name: str, up: dict, mm) -> dict:
    """UI-параметры тела STEP (идентификатор файла, номер тела, отпечаток и ось в мм) → СИ."""
    body = up["body"]
    if isinstance(body, bool) or not isinstance(body, int):
        raise ValueError(f"{name}: номер тела должен быть целым числом.")
    p = {"path": _step_path(up["file_id"]), "body": body}
    if up.get("volume_mm3") is not None:
        p["volume"] = float(up["volume_mm3"]) * 1.0e-9
    if up.get("centroid_mm") is not None:
        p["centroid"] = tuple(mm(v) for v in up["centroid_mm"])
    if up.get("axis_origin_mm") is not None:
        p["axis_origin"] = tuple(mm(v) for v in up["axis_origin_mm"])
    if up.get("axis_dir") is not None:
        p["axis_dir"] = tuple(float(v) for v in up["axis_dir"])
    return p


def _geo3d_from(o: dict) -> GeoObject3D:
    """UI-объект 3D (мм, градусы) → GeoObject3D (м, радианы)."""
    k = str(o.get("kind"))
    name = str(o.get("name") or k)
    if k not in _KIND3D:
        raise ValueError(f"{name}: неизвестный примитив {k!r}.")
    up = dict(o.get("params") or {})
    missing = [key for key in _KIND3D[k] if key not in up]
    if missing:
        raise ValueError(f"{name}: не заданы размеры {missing}.")
    mm = lambda v: float(v) / 1000.0                                   # noqa: E731
    if k == CAD_KIND:
        p = _step_params(name, up, mm)
    else:
        p = {key: mm(up[key]) for key in _KIND3D[k] if key not in ("a1", "a2", "points")}
    if k == "tube_sector":
        p["a1"], p["a2"] = math.radians(float(up["a1"])), math.radians(float(up["a2"]))
    if k == "prism":
        p["points"] = [(mm(x), mm(y)) for x, y in up["points"]]
    md = o.get("magnet_dir") or "axial"
    if isinstance(md, (list, tuple)):
        md = tuple(float(v) for v in md)
    ms = o.get("mesh_size_mm")
    return GeoObject3D(name=name, kind=k, params=p,
                       material=_object_material(str(o.get("material", "air"))),
                       center=tuple(mm(v) for v in (o.get("center") or (0, 0, 0))),
                       rotation=tuple(math.radians(float(v)) for v in (o.get("rotation") or (0, 0, 0))),
                       magnet_dir=md, mesh_size=(mm(ms) if ms else None),
                       priority=int(o.get("priority", 1)),
                       magnet_rotation=tuple(math.radians(float(v))
                                             for v in (o.get("magnet_rotation") or (0, 0, 0))))


def _objects3d(body: dict) -> list[GeoObject3D]:
    objs = [_geo3d_from(o) for o in (body.get("objects") or [])]
    if not objs:
        raise ValueError("добавьте хотя бы один объект.")
    names = [o.name for o in objs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError("имена объектов должны быть разными: " + ", ".join(dup) + ".")
    if "domain" in names:
        raise ValueError("имя «domain» занято фоновой областью.")
    return objs


def _range(vals: np.ndarray) -> list | None:
    v = vals[np.isfinite(vals)]
    return [float(v.min()), float(v.max())] if v.size else None


@app.post("/api/3d/preview")
def api_3d_preview(body: dict = Body(default={})) -> dict:
    """
    Предпросмотр 3D-геометрии: поверхности тел без объёмной сетки (то же построение тел, что у сетки),
    осевая линия каждого тела (мм) — ось, от которой считается осевое и радиальное намагничивание, — и при
    `arrows: true` стрелки намагничивания магнитов (этап 3D-7).
    """
    try:
        objs = _objects3d(dict(body))
        pv = preview_geometry(objs, magnet_arrows=bool(body.get("arrows")))
    except Exception as e:  # noqa: BLE001 — плохая геометрия → в UI
        return {"error": str(e)}
    surfs = pv.surfaces
    pts = [t.reshape(-1, 3) * 1000.0 for t in surfs if t.size]
    allp = np.concatenate(pts) if pts else None
    return {"objects": [{"name": o.name, "material": material_kind(o.material), "n": int(t.shape[0]),
                         "tris": pack(t.reshape(-1, 3) * 1000.0, np.float32),
                         "axis": (axis_segment(o, t) * 1000.0).tolist() if t.size else None,
                         "arrows": arrows_payload(a)}
                        for o, t, a in zip(objs, surfs, pv.arrows)],
            "bbox": None if allp is None else [allp.min(axis=0).tolist(), allp.max(axis=0).tolist()]}


@app.post("/api/3d/step_upload")
async def api_3d_step_upload(request: Request, name: str = "") -> dict:
    """
    Принять STEP-файл (тело запроса — байты файла) и вернуть его тела (мм, мм³). Файл хранится под
    SHA-256 содержимого, повторная загрузка ничего не пишет. Файл без тел — ошибка, и он не сохраняется.
    """
    shown = str(name or "файл")
    path = None
    try:
        data = await request.body()
        if not data:
            raise ValueError("пустой файл.")
        if len(data) > _STEP_MAX_BYTES:
            raise ValueError(f"файл больше {_STEP_MAX_BYTES // 2 ** 20} МБ.")
        fid = hashlib.sha256(data).hexdigest()
        path = _step_dir() / f"{fid}.step"
        written = not path.is_file()
        if written:
            part = path.with_suffix(".part")
            part.write_bytes(data)
            os.replace(part, path)
        try:
            bodies = await run_in_threadpool(step_bodies, path)      # чтение STEP — не на потоке event-loop
        except Exception:
            if written:
                path.unlink(missing_ok=True)
            raise
    except Exception as e:  # noqa: BLE001 — плохой файл → в UI
        msg = str(e)
        if path is not None:
            msg = msg.replace(os.path.abspath(str(path)), shown).replace(str(path), shown)
        return {"error": msg}
    return {"file_id": fid, "name": shown, "size": len(data),
            "bodies": [{"index": b.index, "name": b.name, "volume_mm3": b.volume * 1.0e9,
                        "centroid_mm": [c * 1.0e3 for c in b.centroid],
                        "bbox_mm": [[v * 1.0e3 for v in b.bbox_min], [v * 1.0e3 for v in b.bbox_max]],
                        "n_faces": b.n_faces} for b in bodies]}


@app.post("/api/3d/step_has")
async def api_3d_step_has(body: dict = Body(default={})) -> dict:
    """Каких STEP-файлов (по SHA-256) нет на сервере — их браузер досылает из файла расчёта."""
    missing = []
    for fid in body.get("file_ids") or []:
        try:
            _step_path(fid)
        except ValueError:
            missing.append(str(fid))
    return {"missing": missing}


def _domain3d(body: dict, objs):
    return auto_domain3d(objs, material=Air(), margin_frac=float(body.get("margin", 2.0)))


def _model_id3d(body: dict) -> str:
    return "d" + hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:11]


def _build_model3d(body: dict) -> str:
    objs = _objects3d(body)
    prob = build_object_problem3d(objs, _domain3d(body, objs),
                                  default_mesh_size=float(body.get("default_mesh_mm", 2.0)) / 1000.0,
                                  grading=float(body.get("grading", 2.0)))
    mid = _model_id3d(body)
    _register_model3d(mid, prob)
    return mid


def _register_model3d(mid: str, prob, field=None) -> None:
    """Модель (и её решение, если есть) — в кэш; прежнее решение этой модели и поток нового магнита — прочь."""
    with _CACHE3D_LOCK:
        _OBJ3D.pop(mid, None)
        _OBJ3D[mid] = prob
        _forget_solutions3d(mid)
        if field is not None:
            _SOL3D[mid] = field
        while len(_OBJ3D) > _OBJ3D_MAX:              # вытесняем самую старую модель
            old = next(iter(_OBJ3D))
            _OBJ3D.pop(old)
            _forget_solutions3d(old)


def _forget_solutions3d(mid: str) -> None:
    """Сетка модели новая или модель вытеснена — решение и поток нового магнита больше не годятся."""
    with _CACHE3D_LOCK:
        _SOL3D.pop(mid, None)
        for key in [k for k in _NEWFLUX3D if k[0] == mid]:
            _NEWFLUX3D.pop(key, None)


def _flux_loss3d(mid: str, prob, f, bc: str) -> tuple[dict, float]:
    """
    Вердикт о размагничивании (Л-104): потеря потока каждого магнита и всех вместе после этого
    расчёта — замер при 20 °C без внешнего поля, новый магнит против магнита с сохранённой долей r.
    Без повреждения — ровно 0 без лишних расчётов; поток нового магнита — один раз на модель.
    """
    names = [r.name for r in prob.magnet_regions()]
    r = f.retention
    if r is None or not (r[prob.magnet_mask()] < 1.0).any():
        return {n: 0.0 for n in names}, 0.0
    new = _NEWFLUX3D.get((mid, bc))
    if new is None:
        new = _NEWFLUX3D[(mid, bc)] = new_magnet_flux(prob, bc=bc)
    losses = flux_loss(prob, r, bc=bc, new_flux=new)
    total = 1.0 - sum((1.0 - losses[n]) * new[n] for n in names) / sum(new[n] for n in names)
    return losses, total


@app.post("/api/3d/model")
def api_3d_model(body: dict = Body(default={})) -> dict:
    """Построить 3D-модель: сетка gmsh и сцена (поверхности объектов из той же сетки). Рабочий поток:
    сетка на сотни тысяч ячеек строится десятки секунд, сервер в это время отвечает на остальное."""
    try:
        mid = _build_model3d(dict(body))
        prob = _OBJ3D[mid]
        scene = scene_payload(prob)
    except Exception as e:  # noqa: BLE001 — плохая геометрия → в UI
        return {"error": str(e)}
    return {"model_id": mid, "n_cells": int(prob.mesh.n_cells), "n_vertices": int(prob.mesh.n_vertices),
            "empty": prob.empty_regions(), "scene": scene}


def _do_solve3d(body: dict) -> dict:
    """Решить 3D-модель (сталь с насыщением, магнит с коленом — метод Ньютона) и собрать сводку."""
    mid = str(body.get("model_id", ""))
    prob = _OBJ3D.get(mid)
    if prob is None:
        raise ValueError("модель не найдена — постройте сетку заново.")
    prob = replace(prob, T=float(body.get("T", 20.0)))     # T — без пересборки сетки
    H0 = body.get("applied_field_kA")
    bc = str(body.get("bc", "neumann"))
    # Шаг Ньютона — сопряжёнными градиентами: то же решение (совпадение ~10⁻¹³), на 57 тыс. ячеек
    # в 10 раз быстрее прямого решателя (этап 3D-6, Л-96).
    f = solve_nonlinear3d(prob, bc=bc, solver="cg",
                          applied_field=None if H0 is None else np.asarray(H0, dtype=float) * 1000.0)
    _SOL3D[mid] = f
    q = cell_quantities(f)
    reg = np.asarray(prob.cell_region)
    Bm = q["B"][0]
    objects = []
    for rid, r in sorted(prob.regions.items()):
        sel = reg == rid
        if rid == 0 or not sel.any():
            continue
        v = f.volumes[sel]
        objects.append({"name": r.name, "material": material_kind(r.material), "volume_cm3": float(v.sum() * 1e6),
                        "B_mean": float(np.average(Bm[sel], weights=v)), "B_max": float(Bm[sel].max())})
    demag = []
    loss_total, loss_error = None, None
    if f.risk is not None:
        losses = {}
        try:
            losses, loss_total = _flux_loss3d(mid, prob, f, bc)
        except Exception as e:  # noqa: BLE001 — вердикт не посчитан → в UI, поле решения остаётся
            loss_error = str(e)
        # Запас в одной ячейке и наибольшая потеря ячейки — не вердикт (у краёв к сетке не сходятся,
        # Л-104): в сводку не выносятся, «где» показывает карта.
        for name, s in demag_summary(f).items():
            demag.append({"name": name, "flux_loss": losses.get(name), "past_knee": s.past_knee_fraction,
                          "damaged": s.damaged_fraction, "beyond_hcj": s.beyond_hcj_fraction,
                          "retained": s.retained})
    ranges = {}
    for key, (vals, unit) in q.items():
        rng = _range(vals)
        ranges[key] = None if rng is None else rng + [unit]
    return {"model_id": mid, "converged": bool(f.converged), "iters": int(f.n_iterations),
            "residual": float(f.residual), "T": float(prob.T), "coenergy_J": coenergy(f),
            "objects": objects, "demag": demag, "flux_loss_total": loss_total, "flux_loss_error": loss_error,
            "flux_measure_T": FLUX_MEASURE_T, "ranges": ranges}


@app.post("/api/3d/solve")
def api_3d_solve(body: dict = Body(default={})) -> dict:
    jid = _JM.submit("solve3d", str(body.get("label") or "Расчёт 3D"), _do_solve3d, dict(body))
    return {"job_id": jid}


def _solution3d(body: dict):
    f = _SOL3D.get(str(body.get("model_id", "")))
    if f is None:
        raise ValueError("нет решения для этой модели — рассчитайте её.")
    return f


# ---- решение 3D в файле расчёта (этап 3D-9): сетка и потенциал φ сохраняются, поле — без пересчёта ----
@app.post("/api/3d/solution")
def api_3d_solution(body: dict = Body(default={})) -> dict:
    """Сетка и узловой потенциал последнего решения модели — для файла расчёта (`fem3d.storage`)."""
    try:
        return field_payload(_solution3d(body))
    except Exception as e:  # noqa: BLE001 — нет решения → в UI
        return {"error": str(e)}


def _do_restore3d(body: dict) -> dict:
    """
    Модель и поле из файла расчёта: объекты — те же, что в файле (`model`, как для /api/3d/model), сетка
    и φ — из `field`. Годность проверяется уравнениями текущего кода (`restore_saved_field`): не годится —
    поле не принимается, ответ `stale` с невязками, браузер предлагает пересчёт.
    """
    model = dict(body.get("model") or {})
    payload = body.pop("field", None)                # тело задачи хранится в очереди — большой массив не держим
    objs = _objects3d(model)
    saved = restore_saved_field(payload, objs, _domain3d(model, objs))
    out = {"ok": saved.ok, "residual": saved.residual, "stored_residual": saved.stored_residual}
    if not saved.ok:
        return {**out, "stale": True}
    mid = _model_id3d(model)
    _register_model3d(mid, saved.problem, saved.field)
    prob = saved.problem
    return {**out, "model_id": mid, "n_cells": int(prob.mesh.n_cells), "n_vertices": int(prob.mesh.n_vertices),
            "empty": prob.empty_regions()}


@app.post("/api/3d/restore")
def api_3d_restore(body: dict = Body(default={})) -> dict:
    jid = _JM.submit("restore3d", str(body.get("label") or "Поле из файла 3D"), _do_restore3d, dict(body))
    return {"job_id": jid}


@app.post("/api/3d/scene")
def api_3d_scene(body: dict = Body(default={})) -> dict:
    """Сцена модели из кэша (поверхности тел из её сетки) — после восстановления из файла расчёта."""
    prob = _OBJ3D.get(str(body.get("model_id", "")))
    if prob is None:
        return {"error": "модель не найдена — откройте расчёт заново."}
    return {"model_id": str(body.get("model_id")), "n_cells": int(prob.mesh.n_cells),
            "n_vertices": int(prob.mesh.n_vertices), "empty": prob.empty_regions(), "scene": scene_payload(prob)}


def _plane(body: dict):
    point = np.asarray(body.get("point_mm", (0.0, 0.0, 0.0)), dtype=float) / 1000.0
    normal = np.asarray(body.get("normal", (0.0, 0.0, 1.0)), dtype=float)
    return point, normal


@app.post("/api/3d/quantity")
def api_3d_quantity(body: dict = Body(default={})) -> dict:
    """Величина по ячейкам (float32, base64) — раскраска поверхностей и разреза."""
    try:
        q = cell_quantities(_solution3d(body))
        name = str(body.get("quantity", "B"))
        if name not in q:
            raise ValueError(f"неизвестная величина {name!r}; есть: {sorted(q)}.")
        vals, unit = q[name]
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    rng = _range(vals)
    return {"quantity": name, "unit": unit, "values": pack(vals, np.float32),
            "min": None if rng is None else rng[0], "max": None if rng is None else rng[1]}


@app.post("/api/3d/section")
def api_3d_section(body: dict = Body(default={})) -> dict:
    """Разрез плоскостью (точка в мм, нормаль): треугольники в мм и номер ячейки на треугольник."""
    try:
        prob = _OBJ3D.get(str(body.get("model_id", "")))
        if prob is None:
            raise ValueError("модель не найдена — постройте сетку заново.")
        point, normal = _plane(body)
        tris, cells = section(prob, point, normal, objects=body.get("objects") or None)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"n": int(tris.shape[0]), "tris": pack(tris.reshape(-1, 3) * 1000.0, np.float32),
            "cells": pack(cells, np.uint32),
            "regions": pack(np.asarray(prob.cell_region)[cells], np.uint32)}   # цвет по материалу


@app.post("/api/3d/force")
def api_3d_force(body: dict = Body(default={})) -> dict:
    """Сила [Н] и момент [Н·м] поля на тело (список объектов) — метод виртуальной работы."""
    from magcore.fem3d import force_weight

    try:
        f = _solution3d(body)
        pt = body.get("point_mm")
        bodies = list(body.get("bodies") or [])
        weight = str(body.get("weight", "laplace"))
        if weight == "laplace":            # гармонический вес на больших сетках — итерационным решателем (Л-96)
            weight = force_weight(f.problem, bodies, kind="laplace", solver="cg")
        ft = magnetic_force_torque(f, bodies, point=None if pt is None else np.asarray(pt, dtype=float) / 1000.0,
                                   weight=weight)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"force_N": ft.force.tolist(), "torque_Nm": ft.torque.tolist(),
            "point_mm": (ft.point * 1000.0).tolist(), "weight": ft.weight}


@app.post("/api/3d/flux")
def api_3d_flux(body: dict = Body(default={})) -> dict:
    """Магнитный поток [Вб] через сечение плоскостью в выбранных объектах."""
    try:
        f = _solution3d(body)
        point, normal = _plane(body)
        phi = flux_through_plane(f, point, normal, objects=body.get("objects") or None)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"flux_Wb": phi}


@app.post("/api/3d/field_lines")
def api_3d_field_lines(body: dict = Body(default={})) -> dict:
    """
    Силовые линии поля B (этап 3D-7): точки всех линий подряд (мм), начало каждой линии, |B| в точке (Тл) и
    поток на линию (Вб) — каждая линия несёт одинаковый поток, поэтому где линии гуще, там больше индукция.
    """
    from magcore.fem3d.fieldlines import trace_field_lines

    try:
        f = _solution3d(body)
        fl = trace_field_lines(f, n_lines=int(body.get("n_lines", 200)), objects=body.get("objects") or None)
    except Exception as e:  # noqa: BLE001 — в UI
        return {"error": str(e)}
    return {"n_lines": fl.n_lines, "n_points": int(fl.points.shape[0]),
            "points": pack(fl.points * 1000.0, np.float32), "offsets": pack(fl.offsets, np.uint32),
            "values": pack(fl.values, np.float32), "delta_flux_Wb": fl.delta_flux,
            "stop": pack(fl.stop, np.uint32)}


@app.post("/api/3d/section_lines")
def api_3d_section_lines(body: dict = Body(default={})) -> dict:
    """
    Силовые линии в плоскости разреза (этап 3D-7): линии проекции B на плоскость, расставленные равномерно
    (не по потоку). `out_of_plane` — медианная доля поля, выходящая из плоскости: у плоскости симметрии она
    около нуля, и тогда это настоящие линии поля, иначе картинка — проекция.
    """
    from magcore.fem3d.fieldlines import trace_section_lines

    try:
        f = _solution3d(body)
        point, normal = _plane(body)
        fl = trace_section_lines(f, point, normal, n_lines=int(body.get("n_lines", 60)))
    except Exception as e:  # noqa: BLE001 — в UI
        return {"error": str(e)}
    return {"n_lines": fl.n_lines, "n_points": int(fl.points.shape[0]),
            "points": pack(fl.points * 1000.0, np.float32), "offsets": pack(fl.offsets, np.uint32),
            "values": pack(fl.values, np.float32), "out_of_plane": fl.out_of_plane}


@app.get("/api/3d/export_vtu")
def api_3d_export_vtu(model_id: str, name: str = "model3d"):
    """Решение в файл .vtu для ParaView (скачивание из своего сервера)."""
    import tempfile

    f = _SOL3D.get(model_id)
    if f is None:
        return {"error": "нет решения для этой модели — рассчитайте её."}
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_ ").strip() or "model3d"
    out_dir = Path(tempfile.gettempdir()) / "magfield_exports"
    out_dir.mkdir(exist_ok=True)
    path = write_vtu(out_dir / f"{safe}_{model_id}.vtu", f)
    return FileResponse(path, media_type="application/octet-stream", filename=f"{safe}.vtu")


def _gap_thickness(params) -> float:
    """
    Толщина зазора [м] — НЕЗАВИСИМО от параметризации машины: у outrunner это поле `air_gap`,
    у спицевой она выводится из диаметров. Нужна проверке «Разрешение зазора».
    """
    gap = getattr(params, "air_gap", None)
    if gap is not None:
        return float(gap)
    d_in = getattr(params, "D_magnet_in_mm", None)
    d_out = getattr(params, "D_tooth_out_mm", None)
    if d_in is not None and d_out is not None:
        return max((float(d_in) - float(d_out)) / 2.0, 0.0) * 1e-3
    return 0.0


@app.get("/api/machine_catalog")
def api_machine_catalog() -> dict:
    """
    ГОТОВЫЕ МАШИНЫ (пресеты) из библиотеки типов. Пресет заполняет поля геометрии И задаёт
    ТИП: схему обмотки (трёхфазная / коллекторная), топологию и стали по регионам.
    Формула K_e зависит от типа: трёхфазная K_t=1.5·K_e, коллекторная K_t=K_e.
    """
    out = []
    for key, factory in _CATALOG.items():
        d = factory()
        p = d.params
        w = d.winding
        item = {
            "id": key, "name": d.name, "winding": w.kind,
            "winding_label": ("коллекторная (щёточный ДПТ)" if w.kind == "commutator"
                              else "трёхфазная (PMSM)"),
            "magnets_on": d.topology.magnets_on,
            "geometry": {
                "n_slots": p.n_slots, "n_poles": p.n_poles,
                "R_bore": p.R_bore * 1e3, "h_stator_yoke": p.h_stator_yoke * 1e3,
                "h_tooth": p.h_tooth * 1e3, "air_gap": p.air_gap * 1e3,
                "h_magnet": p.h_magnet * 1e3, "h_rotor_yoke": p.h_rotor_yoke * 1e3,
                "tooth_width_frac": p.tooth_width_frac,
                "magnet_embrace": p.magnet_embrace, "axial_length": p.axial_length * 1e3,
            },
            "mesh": {k: v * 1e3 for k, v in (p.mesh_size_by_region or {}).items()},
            "magnet_id": {"N35": "n35", "N42SH-representative": "ndfeb",
                          "KS25DTs-240": "ks25dts240"}.get(d.materials.magnet.material_id),
            "steel_armature": ("steel10" if "Steel10" in d.materials.steel_armature.curve_id
                               else "steel"),
            "steel_yoke": ("steel10" if "Steel10" in d.materials.yoke_curve.curve_id else "steel"),
        }
        if isinstance(w, CommutatorWinding):
            item["commutator"] = {"Z": w.conductors_total, "a": w.parallel_path_pairs,
                                  "skew_deg": w.skew_deg}
        else:
            item["turns_per_slot"] = w.turns_per_slot
        if key == "dp25":
            item["nameplate"] = {"K_e": 0.02946, "R": 3.9, "note": "ТУ КМИЖ.524212.006"}
        out.append(item)
    return {"machines": out}


@app.get("/api/materials")
def api_materials() -> dict:
    """Материалы для списков: магниты и стали (встроенные + свои).

    Для СВОИХ материалов отдаём и полную спецификацию (`spec`) — фронтенд вшивает её в архив
    расчёта, чтобы тот был самодостаточным (не «поедет», если материал потом изменить/удалить).
    Встроенные (`builtin`) всегда воспроизводимы сервером по id, спека не нужна."""
    custom = _load_custom_materials()
    gone = materials_db.deleted_ids()
    magnets = []
    for k, v in _BUILTIN_MAGNETS.items():          # представительные пресеты — с параметрами,
        if k in gone:                              # чтобы строка свойств не была пустой
            continue
        e = {"id": k, "name": v["name"], "builtin": True, "family": v.get("family", ""),
             "source": "встроенная", "source_title": "встроенная"}
        try:
            mm = _magnet_by_id(k)
            e.update(Br=round(float(mm.Br0), 4), Hcb_kA=round(float(mm.Hcb0) / 1e3, 1),
                     Hcj_kA=round(float(mm.Hcj0) / 1e3, 1), Hk_kA=round(float(mm.Hk0) / 1e3, 1),
                     alpha_Br=float(mm.alpha_Br), gamma_Hc=float(mm.gamma_Hc),
                     BHmax=None, T_max=None, hk_given=True, note="")
        except Exception:  # noqa: BLE001 — пресет без модели просто останется без чисел
            pass
        magnets.append(e)
    # Справочные марки (Arnold, ГОСТ 21559-76, ГОСТ Р 52956-2008) — с семейством и источником,
    # чтобы интерфейс мог фильтровать по NdFeB/SmCo и искать по марке. Скрытые пропускаем,
    # изменённые пользователем отдаём в редакции пользователя (пометка edited).
    for g in magnet_catalog.CATALOG:
        if g.id in gone:
            continue
        ov = custom.get(g.id)
        if ov is not None:
            magnets.append({"id": g.id, "name": ov.get("name", g.grade), "builtin": False,
                            "edited": True, "family": ov.get("family", g.family),
                            "source": g.source, "source_title": g.source_title + " · изменено",
                            "Br": round(float(ov["Br"]), 4),
                            "Hcb_kA": round(float(ov["Hcb"]) / 1e3, 1),
                            "Hcj_kA": round(float(ov["Hcj"]) / 1e3, 1),
                            "Hk_kA": round(float(ov["Hk"]) / 1e3, 1),
                            "BHmax": g.BHmax, "alpha_Br": float(ov.get("alpha_Br", g.alpha_Br)),
                            "gamma_Hc": float(ov.get("gamma_Hc", g.gamma_Hc)),
                            "T_max": ov.get("T_max", g.T_max), "BHmax": ov.get("BHmax", g.BHmax),
                            "hk_given": bool(ov.get("hk_given", True)), "note": ""})
            continue
        magnets.append({"id": g.id, "name": g.grade, "builtin": True, "family": g.family,
                        "source": g.source, "source_title": g.source_title,
                        "Br": round(g.Br, 4), "Hcb_kA": round(g.Hcb / 1e3, 1),
                        "Hcj_kA": round(g.Hcj / 1e3, 1), "Hk_kA": round(g.Hk / 1e3, 1),
                        "BHmax": g.BHmax, "alpha_Br": g.alpha_Br, "gamma_Hc": g.gamma_Hc,
                        "T_max": g.T_max, "hk_given": False, "hk_ratio": g.hk_ratio,
                        "note": g.note})
    steels = [{"id": k, "name": v["name"], "builtin": True, "source_title": "встроенная"}
              for k, v in _BUILTIN_STEELS.items() if k not in gone]
    for mid, spec in custom.items():
        if mid in magnet_catalog._BY_ID:          # уже отдан выше как «изменённая справочная»
            continue
        entry = {"id": mid, "name": spec.get("name", mid), "builtin": False, "spec": spec,
                 "family": spec.get("family", "свой"), "source": "свой",
                 "source_title": "свой материал"}
        if spec.get("kind") != "steel":
            # свои магниты показываем в общей таблице теми же колонками, что и справочные
            entry.update(Br=round(float(spec["Br"]), 4),
                         Hcb_kA=round(float(spec["Hcb"]) / 1e3, 1),
                         Hcj_kA=round(float(spec["Hcj"]) / 1e3, 1),
                         Hk_kA=round(float(spec["Hk"]) / 1e3, 1),
                         alpha_Br=float(spec.get("alpha_Br", 0.12)),
                         gamma_Hc=float(spec.get("gamma_Hc", 0.6)),
                         BHmax=spec.get("BHmax"), T_max=spec.get("T_max"),
                         hk_given=bool(spec.get("hk_given", True)), note="")
        (steels if spec.get("kind") == "steel" else magnets).append(entry)
    return {"magnets": magnets, "steels": steels,
            "families": list(magnet_catalog.FAMILIES),
            "sources": magnet_catalog.SOURCES}


def _save_material_spec(mid: str | None, spec: dict) -> str:
    """Создать (mid=None) или перезаписать материал. Возвращает итоговый id."""
    return materials_db.upsert(spec, mid)


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
            mid = str(body.get("id") or "") or None          # id есть => РЕДАКТИРОВАНИЕ
            curve = SteelBHCurve(curve_id=mid or "new", name=name, H_values=H, B_values=B)
            spec = {"kind": "steel", "name": name, "family": "сталь",
                    "H": H.tolist(), "B": B.tolist()}
            mid = _save_material_spec(mid, spec)
            return {"id": mid, "ok": True, "n_points": int(curve.n_points),
                    "B_max": round(float(curve.B_max), 3)}
        # ОБЯЗАТЕЛЬНЫЕ паспортные величины — без них материал в библиотеку не заводится.
        family = str(body.get("family", "")).strip()
        if family not in magnet_catalog.FAMILIES:
            return {"error": "укажите класс материала: %s." % ", ".join(magnet_catalog.FAMILIES)}
        req = {"Br": "B_r, Тл", "Hcb_kA": "H_cB, кА/м", "Hcj_kA": "H_cJ, кА/м",
               "alpha_Br": "α_Br, %/°C", "gamma_Hc": "γ_Hc, %/°C"}
        def _bad(key: str) -> bool:
            v = body.get(key)
            if v is None or str(v).strip() == "":
                return True
            try:
                return not math.isfinite(float(v))
            except (TypeError, ValueError):
                return True

        miss = [t for k, t in req.items() if _bad(k)]
        if miss:
            return {"error": "обязательные данные не заполнены: " + "; ".join(miss)}
        # H_k НЕОБЯЗАТЕЛЕН, но ВЛИЯЕТ на расчёт демага: если не задан, принимается
        # k*H_cJ по классу материала (magnet_catalog.HK_RATIO) — как для справочных марок.
        hcj = float(body["Hcj_kA"]) * 1e3
        hk_in = body.get("Hk_kA")
        hk_given = hk_in is not None and str(hk_in).strip() != ""
        hk = float(hk_in) * 1e3 if hk_given else \
            magnet_catalog.HK_RATIO.get(family, magnet_catalog.HK_RATIO_DEFAULT) * hcj
        spec = {
            "kind": "magnet", "name": name, "family": family,
            "Br": float(body["Br"]), "Hcb": float(body["Hcb_kA"]) * 1e3,
            "Hk": hk, "hk_given": bool(hk_given), "Hcj": hcj,
            "alpha_Br": float(body["alpha_Br"]),
            "gamma_Hc": float(body["gamma_Hc"]), "T0": float(body.get("T0", 20.0)),
        }
        for k, key in (("BHmax", "BHmax"), ("T_max", "T_max")):      # справочные, на расчёт не влияют
            v = body.get(k)
            if v is not None and str(v).strip() != "":
                spec[key] = float(v)
        mid = str(body.get("id") or "") or None              # id есть => РЕДАКТИРОВАНИЕ
        mg = magnet_from_datasheet(mid or "new", name, (1, 0, 0), Br=spec["Br"], Hcb=spec["Hcb"],
                                   Hk=spec["Hk"], Hcj=spec["Hcj"], alpha_Br=spec["alpha_Br"],
                                   gamma_Hc=spec["gamma_Hc"], T0=spec["T0"])  # валидирует
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as e:
        return {"error": f"некорректные параметры: {e}"}
    mid = _save_material_spec(mid, spec)
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
        # «Hcb» отдаём ФАКТИЧЕСКИЙ — снятый с кривой нуль B(H). Параметр m.Hcb(T) масштабируется
        # по alpha_Br и задаёт лишь наклон mu_rec; когда колено заходит перед H_cB (горячий
        # магнит), кривая ломается раньше и параметр расходится с кривой почти вдвое.
        Hcb_curve = float(c.Hcb_actual())
    except Exception as e:  # noqa: BLE001 — T вне диапазона модели и пр. → в UI
        return {"error": str(e)}
    return {
        "H": np.round(c.H_values, 1).tolist(), "B": np.round(c.B_values, 4).tolist(),
        "Br": round(float(m.Br(T)), 4), "Hcb": round(Hcb_curve, 1),
        "Hcb_line": round(float(m.Hcb(T)), 1),
        "Hk": round(Hk, 1), "B_knee": round(B_knee, 4),
        "mu_rec": round(float(m.mu_rec), 4), "T_limit": round(float(m.temperature_limit()), 1),
    }


@app.get("/api/materials/{mid}")
def api_material_get(mid: str) -> dict:
    """Спецификация одного своего материала — для предзаполнения формы редактирования."""
    spec = materials_db.get(mid)
    if spec is None and mid in magnet_catalog._BY_ID:      # справочная марка — отдаём как есть
        g = magnet_catalog.by_id(mid)
        spec = {"kind": "magnet", "name": g.grade, "family": g.family, "Br": g.Br,
                "Hcb": g.Hcb, "Hk": g.Hk, "Hcj": g.Hcj,
                "alpha_Br": g.alpha_Br, "gamma_Hc": g.gamma_Hc, "T0": 20.0}
    if spec is None:                                       # встроенный пресет
        try:
            m = _magnet_by_id(mid)
            spec = {"kind": "magnet", "name": _BUILTIN_MAGNETS.get(mid, {}).get("name", mid),
                    "family": _BUILTIN_MAGNETS.get(mid, {}).get("family", ""),
                    "Br": m.Br0, "Hcb": m.Hcb0, "Hk": m.Hk0, "Hcj": m.Hcj0,
                    "alpha_Br": m.alpha_Br, "gamma_Hc": m.gamma_Hc, "T0": 20.0}
        except Exception:  # noqa: BLE001
            spec = None
    return {"id": mid, "spec": spec} if spec else {"error": "материал не найден"}


@app.post("/api/materials/delete")
def api_material_delete(body: dict = Body(default={})) -> dict:
    """Убрать материал из библиотеки БЕЗВОЗВРАТНО (для любого источника)."""
    mid = str(body.get("id", ""))
    if mid.startswith("cust_"):
        return {"ok": materials_db.delete(mid), "id": mid}
    materials_db.delete_catalog(mid)
    return {"ok": True, "id": mid}


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="ui")
