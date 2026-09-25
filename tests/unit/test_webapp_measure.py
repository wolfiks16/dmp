import time

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from magcore.constants import MU0  # noqa: E402
from magcore.fem3d.probes import sample  # noqa: E402
from webapp.server import _SOL3D, app  # noqa: E402

# ИЗМЕРЕНИЯ 3D ЧЕРЕЗ СЕРВЕР (пункт 6 плана интерфейса): /api/3d/probe (точка, линия, окружность), /api/3d/body_mean,
# /api/3d/saturation. Точность — в test_fem3d_probes.py; здесь — что сервер отдаёт те же числа, что решение, в
# единицах интерфейса (мм, Тл, кА/м), точка вне сетки — null, плохой ввод — понятная ошибка, а не падение.

client = TestClient(app)
MAGNET = {"name": "магнит", "kind": "box", "params": {"lx": 10, "ly": 10, "lz": 4}, "material": "ndfeb",
          "magnet_dir": "axial", "center": [0, 0, 0], "rotation": [0, 0, 0], "priority": 1}
PLATE = {"name": "пластина", "kind": "box", "params": {"lx": 12, "ly": 12, "lz": 2}, "material": "steel",
         "center": [0, 0, 4], "rotation": [0, 0, 0], "priority": 1}


def _post(url, body):
    r = client.post(url, json=body)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def mid():
    md = _post("/api/3d/model", {"objects": [MAGNET, PLATE], "default_mesh_mm": 1.5, "margin": 2.0})
    assert "error" not in md, md
    jid = _post("/api/3d/solve", {"model_id": md["model_id"], "T": 20.0})["job_id"]
    t0 = time.time()
    while time.time() - t0 < 300:
        st = client.get(f"/api/jobs/{jid}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert st["status"] == "done", st
    return md["model_id"]


def test_point_probe_returns_the_cell_value_in_interface_units(mid):
    f = _SOL3D[mid]
    d = _post("/api/3d/probe", {"model_id": mid, "kind": "point", "point_mm": [1.0, -2.0, 0.5]})
    ref = sample(f, np.array([[1.0e-3, -2.0e-3, 0.5e-3]]))
    assert d["body"] == ["магнит"] and d["outside"] == 0
    for k, key in enumerate(("Bx", "By", "Bz")):
        assert d[key][0] == pytest.approx(ref.B[0, k], abs=5e-6)                  # округление до 1e-5 Тл
    for k, key in enumerate(("Hx", "Hy", "Hz")):
        assert d[key][0] == pytest.approx(ref.H[0, k] / 1e3, abs=5e-4)            # до 1e-3 кА/м
    far = _post("/api/3d/probe", {"model_id": mid, "kind": "point", "point_mm": [900, 0, 0]})
    assert far["outside"] == 1 and far["Bx"] == [None] and far["body"] == [""]


def test_line_and_circle_probes(mid):
    ln = _post("/api/3d/probe", {"model_id": mid, "kind": "line", "p1_mm": [-8, 0, 0], "p2_mm": [8, 0, 0], "n": 33})
    assert len(ln["s_mm"]) == len(ln["Bz"]) == 33 and ln["s_mm"][0] == 0.0 and ln["s_mm"][-1] == pytest.approx(16.0)
    assert ln["outside"] == 0 and ln["Bz"][16] > 0.0                             # в центре магнита B вдоль оси
    c = _post("/api/3d/probe", {"model_id": mid, "kind": "circle", "center_mm": [0, 0, 0], "normal": [0, 0, 1],
                                "radius_mm": 3.0, "n": 36})
    assert len(c["theta_deg"]) == len(c["Br"]) == len(c["Bn"]) == 36 and c["theta_deg"][1] == pytest.approx(10.0)
    # разложение по базису окружности — из того же ответа: нормаль Z, θ от +X к +Y (как в 2D)
    th = np.radians(c["theta_deg"])
    Bx, By, Bz = (np.asarray(c[k], dtype=float) for k in ("Bx", "By", "Bz"))
    assert np.allclose(c["Bn"], Bz, rtol=0, atol=2e-5)                            # оба округлены до 1e-5 Тл
    assert np.allclose(c["Br"], Bx * np.cos(th) + By * np.sin(th), rtol=0, atol=3e-5)
    assert np.allclose(c["Bt"], -Bx * np.sin(th) + By * np.cos(th), rtol=0, atol=3e-5)


def test_body_mean_and_saturation(mid):
    m = _post("/api/3d/body_mean", {"model_id": mid, "body": "магнит"})
    assert m["magnet"] is True and m["Bd"] > 0.0 > m["Hd_kA"]
    assert m["Pc"] == pytest.approx(m["Bd"] / (MU0 * abs(m["Hd_kA"]) * 1e3), rel=1e-3)
    assert m["volume_cm3"] == pytest.approx(10 * 10 * 4 / 1000.0, rel=1e-9)      # брусок — объём точный
    st = _post("/api/3d/body_mean", {"model_id": mid, "body": "пластина"})
    assert st["magnet"] is False and "Pc" not in st
    s = _post("/api/3d/saturation", {"model_id": mid, "threshold_T": 1.0})
    (row,) = s["steel"]
    assert row["body"] == "пластина" and 0.0 <= row["fraction_above"] <= 1.0 and row["B_max"] > 0.0


def test_bad_input_is_an_error_not_a_crash(mid):
    for body in ({"model_id": mid, "kind": "плоскость"}, {"model_id": mid, "kind": "line", "p1_mm": [0, 0], "p2_mm": [1, 1, 1]},
                 {"model_id": mid, "kind": "circle", "center_mm": [0, 0, 0], "normal": [0, 0, 0], "radius_mm": 3},
                 {"model_id": mid, "kind": "line", "p1_mm": [0, 0, 0], "p2_mm": [1, 1, 1], "n": 1},
                 {"model_id": "нет-такой", "kind": "point", "point_mm": [0, 0, 0]}):
        assert "error" in _post("/api/3d/probe", body), body
    assert "error" in _post("/api/3d/body_mean", {"model_id": mid, "body": "нет такого"})
    assert "error" in _post("/api/3d/saturation", {"model_id": mid, "threshold_T": -1})
