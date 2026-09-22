import json
import time

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from magcore.fem3d.scene import unpack  # noqa: E402
from webapp.server import app  # noqa: E402

# Этап 3D-5: обработчики 3D на сервере — вся цепочка на малой модели (магнит и стальная пластина
# над ним с зазором 1 мм): предпросмотр → сетка и сцена → расчёт фоновой задачей → величины, разрез,
# сила, поток, выгрузка в ParaView. Неверные входы — понятная ошибка, а не падение сервера.
# Точность здесь не проверяется (она — в тестах ядра): только связность и знаки.

client = TestClient(app)
MAGNET = {"name": "магнит", "kind": "box", "params": {"lx": 10, "ly": 10, "lz": 4}, "material": "ndfeb",
          "magnet_dir": "axial", "center": [0, 0, 0], "rotation": [0, 0, 0], "priority": 1}
PLATE = {"name": "пластина", "kind": "box", "params": {"lx": 12, "ly": 12, "lz": 2}, "material": "steel",
         "center": [0, 0, 4], "rotation": [0, 0, 0], "priority": 1}


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


def test_3d_pipeline_roundtrip():
    objs = [MAGNET, PLATE]
    pv = _post("/api/3d/preview", {"objects": objs})
    assert "error" not in pv, pv
    assert [o["name"] for o in pv["objects"]] == ["магнит", "пластина"]
    assert all(o["n"] > 0 and unpack(o["tris"], np.float32).size == 9 * o["n"] for o in pv["objects"])
    # осевые линии (этап 3D-7): ось z через центр тела на длине проекции плюс 0,15 от max(проекция, габарит)
    assert np.allclose(pv["objects"][0]["axis"], [[0, 0, -3.5], [0, 0, 3.5]], rtol=0.0, atol=1e-9)
    assert np.allclose(pv["objects"][1]["axis"], [[0, 0, 1.2], [0, 0, 6.8]], rtol=0.0, atol=1e-9)
    # стрелки намагничивания (этап 3D-7): только по запросу и только у магнита; магнит «вдоль оси» — +z
    assert all(o["arrows"] is None for o in pv["objects"])
    pa = _post("/api/3d/preview", {"objects": objs, "arrows": True})["objects"]
    assert pa[1]["arrows"] is None and pa[0]["arrows"]["n"] > 0
    assert np.allclose(unpack(pa[0]["arrows"]["dirs"], np.float32).reshape(-1, 3), [0.0, 0.0, 1.0])
    # поворот намагниченности в градусах (этап 3D-10): «по +X» и 45° вокруг Z — под 45° в плоскости XY
    turned = dict(MAGNET, magnet_dir=[1, 0, 0], magnet_rotation=[0, 0, 45])
    pt = _post("/api/3d/preview", {"objects": [turned, PLATE], "arrows": True})["objects"]
    assert np.allclose(unpack(pt[0]["arrows"]["dirs"], np.float32).reshape(-1, 3), [2 ** -0.5, 2 ** -0.5, 0.0],
                       rtol=0.0, atol=1e-6)
    assert "error" in _post("/api/3d/preview", {"objects": [dict(turned, magnet_rotation=[0, "x", 0])]})
    md = _post("/api/3d/model", {"objects": objs, "default_mesh_mm": 1.5, "margin": 2.0})
    assert "error" not in md, md
    mid = md["model_id"]
    assert md["n_cells"] > 0 and {o["name"] for o in md["scene"]["objects"]} == {"магнит", "пластина"}
    sc = {o["name"]: o for o in md["scene"]["objects"]}
    assert sc["пластина"]["arrows"] is None and sc["магнит"]["arrows"]["n"] > 0         # стрелки по ячейкам сетки
    st = _wait(_post("/api/3d/solve", {"model_id": mid, "T": 20.0})["job_id"])
    assert st["status"] == "done", st
    res = st["result"]
    assert res["converged"] and [d["name"] for d in res["demag"]] == ["магнит"]
    assert res["ranges"]["B"][1] > 0.0 and np.isfinite(res["coenergy_J"])
    # вердикт — потеря потока (Л-104): при 20 °C повреждения нет — ровно 0; одна ячейка в сводку не выносится
    assert res["flux_measure_T"] == 20.0 and res["flux_loss_error"] is None
    assert res["demag"][0]["flux_loss"] == 0.0 and res["flux_loss_total"] == 0.0
    assert "worst_margin_kA" not in res["demag"][0]
    q = _post("/api/3d/quantity", {"model_id": mid, "quantity": "B"})
    assert "error" not in q, q
    assert unpack(q["values"], np.float32).size == md["n_cells"] and q["unit"] == "Тл"
    sec = _post("/api/3d/section", {"model_id": mid, "point_mm": [0, 0, 0], "normal": [1, 0, 0]})
    cells = unpack(sec["cells"], np.uint32)
    assert sec["n"] > 0 and cells.size == sec["n"] and int(cells.max()) < md["n_cells"]
    assert unpack(sec["tris"], np.float32).size == 9 * sec["n"]
    fp = _post("/api/3d/force", {"model_id": mid, "bodies": ["пластина"]})
    fm = _post("/api/3d/force", {"model_id": mid, "bodies": ["магнит"]})
    assert "error" not in fp and "error" not in fm, (fp, fm)
    assert fp["force_N"][2] < 0.0 < fm["force_N"][2]                  # пластина и магнит притягиваются
    fl = _post("/api/3d/flux", {"model_id": mid, "point_mm": [0, 0, 0], "normal": [0, 0, 1],
                                "objects": ["магнит"]})
    assert fl["flux_Wb"] > 0.0                                          # поток вдоль намагничивания
    fl = _post("/api/3d/field_lines", {"model_id": mid, "n_lines": 12})     # силовые линии (этап 3D-7)
    assert "error" not in fl, fl
    pts = unpack(fl["points"], np.float32).reshape(-1, 3)
    offs = unpack(fl["offsets"], np.uint32)
    assert fl["n_lines"] == 12 and offs.size == 13 and offs[-1] == pts.shape[0]
    assert unpack(fl["values"], np.float32).size == pts.shape[0] and fl["delta_flux_Wb"] > 0.0
    assert all(offs[i + 1] - offs[i] >= 2 for i in range(12))               # линия — не одна точка
    assert "error" in _post("/api/3d/field_lines", {"model_id": mid, "objects": ["пластина"]})   # не магнит
    sl = _post("/api/3d/section_lines", {"model_id": mid, "point_mm": [0, 0, 0], "normal": [0, 1, 0],
                                         "n_lines": 20})                    # плоскость симметрии модели
    assert "error" not in sl, sl
    spts = unpack(sl["points"], np.float32).reshape(-1, 3)
    assert sl["n_lines"] >= 5 and np.abs(spts[:, 1]).max() == 0.0 and sl["out_of_plane"] < 0.05
    r = client.get("/api/3d/export_vtu", params={"model_id": mid})
    assert r.status_code == 200 and r.content.startswith(b"<?xml") and b"UnstructuredGrid" in r.content
    # нагрев до 150 °C повреждает магнит у полюсной грани: после остывания потока меньше
    hot = _wait(_post("/api/3d/solve", {"model_id": mid, "T": 150.0})["job_id"])
    assert hot["status"] == "done", hot
    d = hot["result"]["demag"][0]
    assert hot["result"]["flux_loss_error"] is None and d["damaged"] > 0.0 and 0.0 < d["flux_loss"] < 1.0
    assert d["flux_loss"] == pytest.approx(hot["result"]["flux_loss_total"], rel=1e-12)   # магнит один


def test_3d_saved_field_reopens_without_recalculation(monkeypatch):
    # Этап 3D-9: решили → забрали сетку и φ для файла расчёта → сервер «перезапущен» (кэши пусты) → открыли
    # из файла: поле, карта риска, сила — те же до бита, без сетки и без решения. Файл с другими уравнениями
    # (другая температура) не принимается — браузер предложит пересчёт; испорченный — ошибка задачи.
    import webapp.server as srv

    model = {"objects": [MAGNET, PLATE], "default_mesh_mm": 1.5, "margin": 2.0}
    md = _post("/api/3d/model", model)
    mid = md["model_id"]
    st = _wait(_post("/api/3d/solve", {"model_id": mid, "T": 150.0})["job_id"])
    assert st["status"] == "done", st
    saved = json.loads(json.dumps(_post("/api/3d/solution", {"model_id": mid})))     # файл расчёта — JSON
    assert "error" not in saved and saved["n_cells"] == md["n_cells"] and saved["T"] == 150.0
    quantities = ("B", "margin", "loss")
    before = {q: unpack(_post("/api/3d/quantity", {"model_id": mid, "quantity": q})["values"], np.float32)
              for q in quantities}
    force = _post("/api/3d/force", {"model_id": mid, "bodies": ["пластина"]})["force_N"]
    for name in ("_OBJ3D", "_SOL3D", "_NEWFLUX3D"):
        monkeypatch.setattr(srv, name, {})
    assert "error" in _post("/api/3d/quantity", {"model_id": mid, "quantity": "B"})          # решения нет
    r = _wait(_post("/api/3d/restore", {"model": model, "field": saved})["job_id"])
    assert r["status"] == "done" and r["result"]["ok"] and r["result"]["model_id"] == mid, r
    assert r["result"]["residual"] == r["result"]["stored_residual"]
    sc = _post("/api/3d/scene", {"model_id": mid})
    assert sc["n_cells"] == md["n_cells"] and {o["name"] for o in sc["scene"]["objects"]} == {"магнит", "пластина"}
    for q in quantities:
        after = unpack(_post("/api/3d/quantity", {"model_id": mid, "quantity": q})["values"], np.float32)
        assert np.array_equal(after, before[q], equal_nan=True), q          # вне магнитов запас — NaN
    assert _post("/api/3d/force", {"model_id": mid, "bodies": ["пластина"]})["force_N"] == force
    stale = _wait(_post("/api/3d/restore", {"model": model, "field": dict(saved, T=170.0)})["job_id"])
    assert stale["status"] == "done" and stale["result"]["stale"] and not stale["result"]["ok"]
    assert stale["result"]["residual"] > stale["result"]["stored_residual"]
    broken = _wait(_post("/api/3d/restore", {"model": model, "field": {"format": "другое"}})["job_id"])
    assert broken["status"] == "error" and "не сохранённое поле" in broken["error"]
    assert "error" in _post("/api/3d/scene", {"model_id": "нет-такой"})


def _step_bytes(tmp_path, make, name="m.step"):
    """STEP в миллиметрах из тел, построенных функцией make(occ) (как выгружают КОМПАС и T-FLEX)."""
    import gmsh

    path = tmp_path / name
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        make(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path.read_bytes()


def test_3d_step_upload_preview_model_and_errors(tmp_path, monkeypatch):
    # Этап 3D-1б: файл STEP приходит телом запроса, хранится под SHA-256, тела — объекты модели.
    import webapp.server as server

    store = tmp_path / "magfield_step"                     # не во временную папку системы, где лежат файлы сервера
    store.mkdir()
    monkeypatch.setattr(server, "_step_dir", lambda: store)
    data = _step_bytes(tmp_path, lambda occ: (occ.addBox(-15, -5, -2, 10, 10, 4), occ.addBox(5, -5, -2, 10, 10, 4)))
    up = client.post("/api/3d/step_upload", params={"name": "два бруска.step"}, content=data).json()
    assert "error" not in up, up
    fid = up["file_id"]
    assert len(fid) == 64 and up["name"] == "два бруска.step" and len(up["bodies"]) == 2
    b0 = up["bodies"][0]
    assert abs(b0["volume_mm3"] - 400.0) < 1e-9 and np.allclose(b0["centroid_mm"], [-10.0, 0.0, 0.0], atol=1e-9)
    # габарит OpenCASCADE расширен на допуск формы 1e-7 м = 1e-4 мм с каждой стороны — допуск отсюда, с запасом
    assert np.allclose(b0["bbox_mm"], [[-15, -5, -2], [-5, 5, 2]], rtol=0.0, atol=2e-4)
    assert client.post("/api/3d/step_upload", content=data).json()["file_id"] == fid    # тот же файл — тот же id
    assert _post("/api/3d/step_has", {"file_ids": [fid, "0" * 64, "плохой"]})["missing"] == ["0" * 64, "плохой"]
    steel = {"name": "брусок STEP", "kind": "step", "material": "steel", "center": [0, 0, 0], "rotation": [0, 0, 0],
             "priority": 1, "params": {"file_id": fid, "body": 0, "volume_mm3": b0["volume_mm3"],
                                       "centroid_mm": b0["centroid_mm"]}}
    magnet = {"name": "магнит STEP", "kind": "step", "material": "ndfeb", "magnet_dir": "radial", "center": [0, 0, 0],
              "rotation": [0, 0, 0], "priority": 1,
              "params": {"file_id": fid, "body": 1, "axis_origin_mm": [0, 0, 0], "axis_dir": [0, 0, 1]}}
    pv = _post("/api/3d/preview", {"objects": [steel, magnet]})
    assert "error" not in pv, pv
    assert all(o["n"] > 0 for o in pv["objects"])
    md = _post("/api/3d/model", {"objects": [steel, magnet], "default_mesh_mm": 2.0, "margin": 1.0})
    assert "error" not in md, md
    assert md["n_cells"] > 0 and {o["name"] for o in md["scene"]["objects"]} == {"брусок STEP", "магнит STEP"}
    wrong = dict(steel, params=dict(steel["params"], volume_mm3=b0["volume_mm3"] * 1.01))
    assert "изменился" in _post("/api/3d/preview", {"objects": [wrong]})["error"]            # отпечаток не совпал
    assert "error" in _post("/api/3d/preview", {"objects": [dict(steel, params=dict(steel["params"], file_id="0" * 64))]})
    assert "error" in _post("/api/3d/preview", {"objects": [dict(steel, params=dict(steel["params"], body=5))]})
    assert "error" in _post("/api/3d/preview", {"objects": [dict(steel, params=dict(steel["params"], body=True))]})
    surf = _step_bytes(tmp_path, lambda occ: occ.addRectangle(0, 0, 0, 5, 5), name="s.step")
    err = client.post("/api/3d/step_upload", params={"name": "оболочка.step"}, content=surf).json()["error"]
    assert "нет тел" in err and "оболочка.step" in err and "magfield_step" not in err    # имя, а не путь сервера
    assert "error" in client.post("/api/3d/step_upload", content=b"").json()


def test_3d_api_reports_bad_input_as_errors():
    assert "error" in _post("/api/3d/preview", {"objects": []})
    assert "error" in _post("/api/3d/preview", {"objects": [MAGNET, dict(MAGNET)]})    # одинаковые имена
    assert "error" in _post("/api/3d/model", {"objects": [dict(MAGNET, kind="cone")]})
    assert "error" in _post("/api/3d/model", {"objects": [dict(MAGNET, name="domain")]})
    st = _wait(_post("/api/3d/solve", {"model_id": "нет-такой"})["job_id"])
    assert st["status"] == "error"
    assert "error" in _post("/api/3d/quantity", {"model_id": "нет-такой", "quantity": "B"})
    assert "error" in _post("/api/3d/section", {"model_id": "нет-такой"})
