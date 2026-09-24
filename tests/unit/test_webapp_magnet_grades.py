import time

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet  # noqa: E402
from webapp.server import _OBJ, _build_object_model, _do_object_solve, api_object_model, app  # noqa: E402

# НЕСКОЛЬКО МАГНИТОВ В ОДНОЙ МОДЕЛИ — через сервер, как их задаёт интерфейс. Сервер даёт каждому телу свою
# копию материала, поэтому прежняя проверка (по объекту в памяти) отвечала «несколько марок магнита … пока не
# поддержано» уже на двух одинаковых магнитах: в 3D — ошибкой расчёта, в 2D — падением построения модели
# (500). Теперь: одна марка у двух тел — одна марка; разные марки считаются, у каждой ячейки колено своей
# марки, вердикт — по каждому магниту. Режим — плоские магниты при 150 °C: NdFeB за коленом, SmCo цел.
# Точность — в тестах ядра (test_magnet_grades.py): здесь связность, подписи и согласованность чисел.

client = TestClient(app)
T_HOT = 150.0
ND, SM = n42sh_magnet((1.0, 0.0, 0.0)), sm2co17_magnet((1.0, 0.0, 0.0))


def _flat2d(name, cx, material):
    """Плоский магнит 20×4 мм, намагниченный поперёк (+y), — как объект из интерфейса (мм)."""
    return {"name": name, "kind": "rect", "params": {"cx": cx, "cy": 0, "w": 20, "h": 4, "angle": 0},
            "material": material, "magnet_dir": [0, 1], "current": 0, "priority": 10}


def _plate3d(name, cx, material):
    """Пластина 20×20×4 мм, намагниченная поперёк (по z)."""
    return {"name": name, "kind": "box", "params": {"lx": 20, "ly": 20, "lz": 4}, "material": material,
            "magnet_dir": "axial", "center": [cx, 0, 0], "rotation": [0, 0, 0], "priority": 1}


def _post(url, body):
    r = client.post(url, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _wait(jid, timeout=300.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/api/jobs/{jid}").json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.2)
    raise AssertionError("задача не завершилась")


def _knee_by_magnet(mid, d):
    """Колено из ответа по ячейкам каждого магнита: {имя: множество значений, кА/м}."""
    prob = _OBJ[mid]
    names = {rid: r.name for rid, r in prob.regions.items()}
    reg = np.asarray(prob.cell_region)[np.asarray(d["demag_cells"], dtype=int)]
    out: dict = {}
    for rid, k in zip(reg, d["demag_knee_cells_kA"]):
        out.setdefault(names[int(rid)], set()).add(k)
    return out


def test_2d_two_magnets_of_one_grade_are_one_grade():
    body = {"objects": [_flat2d("магнит 1", -15, "ndfeb"), _flat2d("магнит 2", 15, "ndfeb")],
            "default_mesh_mm": 1.0, "margin": 3.0}
    m = api_object_model(dict(body))                         # прежде — 500 на has_magnet
    assert "error" not in m and m["has_magnet"] is True
    mid = _build_object_model(dict(body))
    d = _do_object_solve({"model_id": mid, "T": T_HOT})
    assert d["converged"]
    knee = round(ND.knee_field(T_HOT) / 1e3, 1)
    assert d["demag_knee_kA"] == knee                        # марка одна — и колено одно на модель
    assert _knee_by_magnet(mid, d) == {"магнит 1": {knee}, "магнит 2": {knee}}
    mags = {x["name"]: x for x in d["magnets"]}
    assert set(mags) == {"магнит 1", "магнит 2"}
    assert mags["магнит 1"]["past_knee"] > 0.0 and mags["магнит 2"]["past_knee"] > 0.0   # NdFeB при 150 °C


def test_2d_two_grades_each_cell_on_its_own_knee():
    body = {"objects": [_flat2d("NdFeB", -15, "ndfeb"), _flat2d("SmCo", 15, "smco")],
            "default_mesh_mm": 1.0, "margin": 3.0}
    m = api_object_model(dict(body))
    assert "error" not in m and m["has_magnet"] is True
    mid = _build_object_model(dict(body))
    d = _do_object_solve({"model_id": mid, "T": T_HOT})
    assert d["converged"]
    assert d["demag_knee_kA"] is None                        # одного колена на модель из двух марок нет
    assert _knee_by_magnet(mid, d) == {"NdFeB": {round(ND.knee_field(T_HOT) / 1e3, 1)},
                                       "SmCo": {round(SM.knee_field(T_HOT) / 1e3, 1)}}
    mags = {x["name"]: x for x in d["magnets"]}
    assert mags["NdFeB"]["past_knee"] > 0.0 and mags["SmCo"]["past_knee"] == 0.0
    assert mags["NdFeB"]["knee_kA"] == round(ND.knee_field(T_HOT) / 1e3, 1)
    assert mags["SmCo"]["knee_kA"] == round(SM.knee_field(T_HOT) / 1e3, 1)
    # ячейка за коленом своей марки — по полю и колену из ответа; оба округлены до 0,1 кА/м, поэтому число
    # ячеек за коленом зажато между подсчётами с запасом ±0,1 кА/м
    hop, kc = np.asarray(d["demag_hop_kA"]), np.asarray(d["demag_knee_cells_kA"])
    assert int((hop < kc - 0.1).sum()) <= d["n_demag"] <= int((hop < kc + 0.1).sum())
    assert 0.0 < d["demag_frac"] < mags["NdFeB"]["past_knee"]           # SmCo цел — общая доля меньше


def test_3d_two_grades_solve_with_a_verdict_per_magnet():
    objs = [_plate3d("NdFeB", -20, "ndfeb"), _plate3d("SmCo", 20, "smco")]
    md = _post("/api/3d/model", {"objects": objs, "default_mesh_mm": 2.0, "margin": 1.0})
    assert "error" not in md, md
    st = _wait(_post("/api/3d/solve", {"model_id": md["model_id"], "T": T_HOT})["job_id"])
    assert st["status"] == "done", st                        # прежде — «Некорректная постановка Problem3D»
    res = st["result"]
    assert res["converged"] and res["flux_loss_error"] is None
    dm = {x["name"]: x for x in res["demag"]}
    assert set(dm) == {"NdFeB", "SmCo"}
    assert dm["NdFeB"]["damaged"] > 0.5 and dm["NdFeB"]["flux_loss"] > 0.05
    assert dm["SmCo"]["damaged"] == 0.0 and dm["SmCo"]["retained"] == pytest.approx(1.0, abs=1e-12)
    # SmCo не повреждён; его поток меняет лишь повреждённый сосед через воздух — вклад сборки, он мал
    assert abs(dm["SmCo"]["flux_loss"]) < 0.01
    assert dm["SmCo"]["flux_loss"] < res["flux_loss_total"] < dm["NdFeB"]["flux_loss"]


def test_3d_two_magnets_of_one_grade_solve():
    objs = [_plate3d("магнит 1", -20, "ndfeb"), _plate3d("магнит 2", 20, "ndfeb")]
    md = _post("/api/3d/model", {"objects": objs, "default_mesh_mm": 2.0, "margin": 1.0})
    assert "error" not in md, md
    st = _wait(_post("/api/3d/solve", {"model_id": md["model_id"], "T": 20.0})["job_id"])
    assert st["status"] == "done", st
    assert [x["name"] for x in st["result"]["demag"]] == ["магнит 1", "магнит 2"]
