import threading
import time

import pytest

pytest.importorskip("gmsh")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from magcore.mesh.gmsh_session import GMSH_LOCK, close_gmsh, open_gmsh  # noqa: E402
from webapp.server import app  # noqa: E402

# СЕРВЕР НЕ ЗАМИРАЕТ НА ВРЕМЯ СЕТКИ. Раньше сетка строилась на потоке, который принимает все запросы: пока
# gmsh работал (для 3D — секунды и десятки секунд), не отвечали ни ход расчёта, ни предпросмотр, ни файлы
# (Л-102: «сервер не отвечал»). Теперь сетка строится в рабочем потоке, а gmsh — один на процесс — под общим
# замком: две сетки одновременно не строятся и друг другу не мешают.

MAGNET = {"name": "магнит", "kind": "box", "params": {"lx": 10, "ly": 10, "lz": 4}, "material": "ndfeb",
          "magnet_dir": "axial", "center": [0, 0, 0], "rotation": [0, 0, 0], "priority": 1}
PLATE = {"name": "пластина", "kind": "box", "params": {"lx": 12, "ly": 12, "lz": 2}, "material": "steel",
         "center": [0, 0, 4], "rotation": [0, 0, 0], "priority": 1}
MODEL3D = {"objects": [MAGNET, PLATE], "default_mesh_mm": 0.7}             # ~84 тыс. ячеек, ~3 с
MODEL2D = {"objects": [{"name": "магнит", "kind": "rect", "params": {"cx": 0, "cy": 0, "w": 10, "h": 4, "angle": 0},
                        "material": "ndfeb", "magnet_dir": [0, 1], "current": 0, "priority": 10}],
           "default_mesh_mm": 0.5, "margin": 4}


def test_server_answers_while_a_3d_mesh_is_being_built():
    with TestClient(app) as c:
        out = []
        t = threading.Thread(target=lambda: out.append(c.post("/api/3d/model", json=MODEL3D).json()))
        t0 = time.perf_counter()
        t.start()
        time.sleep(0.3)                                     # сетка уже строится
        waits = []
        while t.is_alive():
            s = time.perf_counter()
            assert c.get("/api/jobs").status_code == 200
            waits.append(time.perf_counter() - s)
            time.sleep(0.1)
        t.join()
        mesh_time = time.perf_counter() - t0
        assert "error" not in out[0] and out[0]["n_cells"] > 50_000
        assert mesh_time > 2.0                              # иначе проверять нечего
        assert len(waits) >= 5 and max(waits) < 0.5         # всё время построения сервер отвечал


def test_two_meshes_at_once_are_the_same_as_one_after_another():
    """2D и 3D строятся одновременно из разных потоков — результат тот же, что по очереди (gmsh под замком)."""
    with TestClient(app) as c:
        seq = (c.post("/api/object_model", json=MODEL2D).json()["n_cells"],
               c.post("/api/3d/model", json=MODEL3D).json()["n_cells"])
        res = {}
        t2 = threading.Thread(target=lambda: res.__setitem__("2d", c.post("/api/object_model", json=MODEL2D).json()))
        t3 = threading.Thread(target=lambda: res.__setitem__("3d", c.post("/api/3d/model", json=MODEL3D).json()))
        t3.start()
        t2.start()
        t2.join()
        t3.join()
        assert "error" not in res["2d"] and "error" not in res["3d"]
        assert (res["2d"]["n_cells"], res["3d"]["n_cells"]) == seq


def _lock_free_for_other_thread():
    got = []

    def probe():
        ok = GMSH_LOCK.acquire(timeout=2.0)
        got.append(ok)
        if ok:
            GMSH_LOCK.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return got == [True]


def test_nested_session_is_an_explicit_error_and_the_lock_stays_usable():
    open_gmsh()
    try:
        with pytest.raises(RuntimeError, match="вложенные"):
            open_gmsh()
    finally:
        close_gmsh()
    assert _lock_free_for_other_thread()
    open_gmsh()                                             # следующий сеанс открывается как обычно
    close_gmsh()


def test_failed_start_of_gmsh_does_not_leave_the_lock_taken(monkeypatch):
    import gmsh

    def broken(**kw):
        raise OSError("gmsh не запустился")

    monkeypatch.setattr(gmsh, "initialize", broken)
    with pytest.raises(OSError):
        open_gmsh()
    assert _lock_free_for_other_thread()                    # иначе все следующие сетки ждали бы вечно
