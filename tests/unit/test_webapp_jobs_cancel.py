import threading
import time

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from magcore.cancel import check  # noqa: E402
from webapp.server import _JobManager, app  # noqa: E402

# «ОТМЕНИТЬ» ИДУЩИЙ РАСЧЁТ: задача из очереди снимается сразу и не запускается; идущая получает флаг и
# останавливается на ближайшей точке отмены (статус cancelling → cancelled); готовую отмена не трогает —
# результат остаётся. Сквозная проверка: настоящий 3D-расчёт через API останавливается посреди работы.


def _wait(jm, jid, states, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = jm.status(jid)["status"]
        if st in states:
            return st
        time.sleep(0.02)
    raise AssertionError(f"задача не дошла до {states}: {jm.status(jid)}")


def test_queued_job_is_cancelled_at_once_and_never_runs():
    jm = _JobManager(max_parallel=1)
    gate, ran = threading.Event(), []
    first = jm.submit("t", "занимает очередь", lambda b: gate.wait(10.0), {})
    queued = jm.submit("t", "ждёт", lambda b: ran.append(1), {})
    assert jm.status(queued)["status"] == "queued"
    assert jm.cancel(queued) == {"status": "cancelled"}
    gate.set()
    _wait(jm, first, {"done"})
    time.sleep(0.2)
    assert ran == [] and jm.status(queued) == {"status": "cancelled"}


def test_running_job_stops_at_the_next_cancel_point():
    jm = _JobManager(max_parallel=1)
    steps = []

    def work(_):
        while True:                                        # «решатель»: итерации с точкой отмены
            check()
            steps.append(1)
            time.sleep(0.01)

    jid = jm.submit("t", "долгий", work, {})
    _wait(jm, jid, {"running"})
    time.sleep(0.1)
    assert jm.cancel(jid) == {"status": "cancelling"}
    assert _wait(jm, jid, {"cancelled", "error", "done"}, timeout=5.0) == "cancelled"
    n = len(steps)
    time.sleep(0.1)
    assert n > 0 and len(steps) == n                       # после отмены итераций больше нет
    assert any(j["id"] == jid and j["status"] == "cancelled" for j in jm.listing())
    jm.clear_finished()
    assert jm.status(jid) == {"status": "unknown"}          # отменённые чистятся вместе с готовыми


def test_finished_or_unknown_job_is_not_touched():
    jm = _JobManager(max_parallel=1)
    jid = jm.submit("t", "быстрый", lambda b: {"x": 1}, {})
    _wait(jm, jid, {"done"})
    assert jm.cancel(jid) == {"status": "done"}
    assert jm.status(jid) == {"status": "done", "result": {"x": 1}}
    assert jm.cancel("нет-такой") == {"status": "unknown"}


def test_real_3d_solve_is_cancelled_through_the_api():
    """Настоящий расчёт 3D (Ньютон и сопряжённые градиенты на 84 тыс. ячеек) останавливается по «Отменить»."""
    pytest.importorskip("gmsh")
    magnet = {"name": "магнит", "kind": "box", "params": {"lx": 10, "ly": 10, "lz": 4}, "material": "ndfeb",
              "magnet_dir": "axial", "center": [0, 0, 0], "rotation": [0, 0, 0], "priority": 1}
    plate = {"name": "пластина", "kind": "box", "params": {"lx": 12, "ly": 12, "lz": 2}, "material": "steel",
             "center": [0, 0, 4], "rotation": [0, 0, 0], "priority": 1}
    with TestClient(app) as c:
        model = c.post("/api/3d/model", json={"objects": [magnet, plate], "default_mesh_mm": 0.7}).json()
        assert "error" not in model and model["n_cells"] > 50_000
        # полный расчёт — для сравнения по времени
        t0 = time.time()
        jid = c.post("/api/3d/solve", json={"model_id": model["model_id"], "label": "полный"}).json()["job_id"]
        while c.get(f"/api/jobs/{jid}").json()["status"] in ("queued", "running"):
            time.sleep(0.05)
        full = time.time() - t0
        assert c.get(f"/api/jobs/{jid}").json()["status"] == "done"

        jid = c.post("/api/3d/solve", json={"model_id": model["model_id"], "label": "отменяемый"}).json()["job_id"]
        while c.get(f"/api/jobs/{jid}").json()["status"] == "queued":
            time.sleep(0.02)
        time.sleep(0.25 * full)                             # расчёт идёт
        t1 = time.time()
        assert c.post(f"/api/jobs/{jid}/cancel").json()["status"] in ("cancelling", "done")
        while c.get(f"/api/jobs/{jid}").json()["status"] in ("running", "cancelling"):
            time.sleep(0.02)
        stopped = time.time() - t1
        final = c.get(f"/api/jobs/{jid}").json()
        assert final == {"status": "cancelled"}, final
        assert stopped < 0.5 * full + 0.5                   # остановился по ходу, а не доработал до конца
        assert c.post(f"/api/jobs/{jid}/cancel").json() == {"status": "cancelled"}   # повторная — ничего
        assert c.post("/api/jobs/нет-такой/cancel").json() == {"status": "unknown"}
