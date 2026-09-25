import time

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from magcore.packing import pack, unpack  # noqa: E402
from webapp.server import (  # noqa: E402
    DEFAULT_SIZES_MM,
    _build_object_model,
    _build_spoke_mesh,
    _do_object_solve,
    _do_restore2d,
    _do_solve,
    _spoke_params_from,
    app,
)

# ПОЛЕ 2D ИЗ ФАЙЛА РАСЧЁТА СВЕРЯЕТСЯ С ТЕКУЩЕЙ ВЕРСИЕЙ РЕШАТЕЛЯ (как 3D-9; решение Sergey 2026-09-25).
# Точность проверки — в test_fem2d_storage.py; здесь — что сервер собирает задачу модели из файла так же,
# как при расчёте: свободная геометрия — из объектов, двигатель — из параметров, марок и режима. Та же
# модель — поле годится; изменились данные, по которым задача строится заново (марка, угол тока), — нет.

FLAT = {"name": "магнит", "kind": "rect", "params": {"cx": 0, "cy": 0, "w": 20, "h": 4, "angle": 0},
        "material": "ndfeb", "magnet_dir": [0, 1], "current": 0, "priority": 10}
PLATE = {"name": "пластина", "kind": "rect", "params": {"cx": 0, "cy": 5, "w": 24, "h": 3, "angle": 0},
         "material": "steel", "current": 0, "priority": 10}
MOTOR = {"params": {"D_tooth_out_mm": 40.0}, "sizes_mm": {k: 2.0 for k in DEFAULT_SIZES_MM}}   # грубая: ~15 тыс. ячеек
MODE = {"material": "ndfeb", "steel": "steel", "T": 20.0, "i_peak": 10.0, "gamma_deg": 180.0, "turns": 40.0}


@pytest.fixture(scope="module")
def objects_field():
    model = {"objects": [FLAT, PLATE], "default_mesh_mm": 1.0, "margin": 3.0}
    out = _do_object_solve({"model_id": _build_object_model(model), "T": 20.0})
    assert out["converged"] and out["field2d"]["residual"] <= 1.0e-6
    return model, out["field2d"]


def test_the_same_free_model_passes_and_another_grade_does_not(objects_field):
    model, field = objects_field
    same = _do_restore2d({"model": {"kind": "objects", **model}, "field": field})
    assert same["ok"] and same["residual"] == same["stored_residual"] and "stale" not in same
    grade = dict(FLAT, material="smco")
    other = _do_restore2d({"model": {"kind": "objects", **model, "objects": [grade, PLATE]}, "field": field})
    assert other == {**other, "ok": False, "stale": True} and other["residual"] > 2.0 * other["stored_residual"]


def _run(c, body):
    jid = c.post("/api/2d/restore", json=body).json()["job_id"]
    for _ in range(600):
        st = c.get(f"/api/jobs/{jid}").json()
        if st["status"] not in ("queued", "running"):
            return st
        time.sleep(0.05)
    raise AssertionError("задача не закончилась")


def test_restore_runs_as_a_job_and_bad_input_is_an_error(objects_field):
    model, field = objects_field
    with TestClient(app) as c:
        run = lambda body: _run(c, body)                                   # noqa: E731
        ok = run({"model": {"kind": "objects", **model}, "field": field})
        assert ok["status"] == "done" and ok["result"]["ok"]
        for bad in ({"model": {"kind": "objects", **model}, "field": {**field, "format": "другое"}},
                    {"model": {"kind": "что-то", **model}, "field": field},
                    {"model": {"kind": "objects", "objects": []}, "field": field}):
            st = run(bad)
            assert st["status"] == "error" and st["error"], bad["model"].get("kind")


def test_a_foreign_value_at_the_boundary_is_stale_not_a_server_error(objects_field):
    """A в узле Дирихле не тот — другая задача, невязка бесконечна; в ответе (JSON) это null, а не сбой."""
    model, field = objects_field
    a = unpack(field["a"], "<f8").copy()
    v = unpack(field["vertices"], "<f8").reshape(-1, 2)
    a[int(np.argmax(np.abs(v[:, 0])))] = 1.0e-6                          # узел на внешней границе области
    with TestClient(app) as c:
        st = _run(c, {"model": {"kind": "objects", **model}, "field": {**field, "a": pack(a, "<f8")}})
    assert st["status"] == "done" and st["result"]["residual"] is None
    assert st["result"]["stale"] and not st["result"]["ok"]


@pytest.fixture(scope="module")
def motor_field():
    params, mid = _spoke_params_from(MOTOR)
    _build_spoke_mesh(params, mid)
    out = _do_solve({"mesh_id": mid, **MODE})
    assert out["converged"]
    return out["field2d"]


def test_motor_field_is_checked_with_the_winding_current_rebuilt_from_the_mode(motor_field):
    model = {"kind": "motor", "spoke": MOTOR["params"], "sizes": MOTOR["sizes_mm"], **MODE}
    same = _do_restore2d({"model": model, "field": motor_field})
    assert same["ok"] and same["residual"] == same["stored_residual"]
    turned = _do_restore2d({"model": {**model, "gamma_deg": 90.0}, "field": motor_field})   # другой угол тока
    assert turned["stale"] and not turned["ok"]
