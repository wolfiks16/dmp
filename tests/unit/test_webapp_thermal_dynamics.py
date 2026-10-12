import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from magcore.fem2d.verification import PASS

pytest.importorskip("gmsh")

ROOT = Path(__file__).resolve().parents[2]

# РЕГРЕССИЯ 2026-10-01: «Тепловая динамика» на проекте «Двигатель» по умолчанию падала сразу —
# «Unable to allocate 7.88 GiB for an array with shape (32522, 32522)»: тепловая сборка строила
# ПЛОТНЫЕ (n,n), по 8n² байт на матрицу. Маленькой сетки у этого двигателя не бывает: сервер держит
# элемент зазора ≤ gap/5 (0,04 мм при зазоре 0,20 мм) ⇒ 64 990 ячеек, 32 522 узла.
#
# Сценарий зовётся так же, как из интерфейса (`_do_scenario`, поля панели по умолчанию), на двух шагах:
# память задаёт сборка, а не длина расчёта, а со второго шага появляется проверка энергобаланса.
# Память — ПИК выделенной памяти процесса (Windows: PeakPagefileUsage, растёт уже при выделении, даже
# нетронутом; POSIX: ru_maxrss). Поэтому расчёт идёт в ОТДЕЛЬНОМ процессе: пик текущего процесса
# помнит всё, что считали тесты до этого. tracemalloc не годится — на Python-циклах магнитной части
# он замедляет шаг в разы.
# Граница — половина ОДНОЙ плотной (n,n): любая плотная матрица превышает её вдвое, а исправному
# расчёту до неё далеко (измерено 1,45 ГиБ — весь процесс с gmsh и двумя сетками — при границе 3,94 ГиБ).
# Время (≈1,5 мин) задаёт магнитная часть: ≈120 итераций Пикара на первом шаге и ещё два нелинейных
# решения — характеристики исправной машины; тепловая часть — доли секунды. Итерация — 0,2 с: 0,1 с
# разрежённое решение и 0,07 с сборка жёсткости. До векторизации источника магнита и B по ячейкам
# (2026-10-02) она стоила 1,8 с, а тест шёл 6–12 мин.

PANEL = {"material": "ndfeb", "steel": "steel", "i_peak": 60, "gamma_deg": 0, "turns": 20,
         "slot_fill": 0.45, "T_amb": 150, "h": 400, "dt": 2, "steady_tol": 0.05,
         "with_losses": False}          # поля панели «Тепловая динамика» по умолчанию (static/index.html)

_CHILD = r'''
import json, sys
from webapp import server


def peak_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes as w

        class PMC(ctypes.Structure):
            _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [(f, ctypes.c_size_t) for f in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = w.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(PMC), w.DWORD]
        c = PMC()
        c.cb = ctypes.sizeof(c)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError("K32GetProcessMemoryInfo")
        return c.PeakPagefileUsage
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


params, mid = server._spoke_params_from({})
mid = server._build_spoke_mesh(params, mid)
mesh = server._GEOM[mid][0].mesh
out = server._do_scenario({"mesh_id": mid, **json.loads(sys.argv[1])})
print(json.dumps({"n_nodes": mesh.n_vertices, "n_cells": mesh.n_cells, "peak": peak_bytes(),
                  "stop_reason": out["stop_reason"], "T_cells": len(out["T_cells"]),
                  "frames": [len(f) for f in out["temp_frames"]], "frame_t": out["temp_frame_t"],
                  "T_amb": out["T_amb"], "T_max": out["T_max"],
                  "checks": {c["name"]: c["status"] for c in out["checks"]["checks"]}}))
'''


@pytest.mark.slow
def test_thermal_dynamics_runs_on_the_default_spoke_motor_with_bounded_memory():
    run = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps({**PANEL, "n_steps": 2})],
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=1800,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert run.returncode == 0, run.stderr[-3000:]
    r = json.loads(run.stdout.strip().splitlines()[-1])
    n = r["n_nodes"]
    assert n > 30_000                                     # тот самый масштаб, где падало (32 522 узла)
    bound = 0.5 * 8 * n * n                               # 8n² байт — одна плотная (n,n)
    assert r["peak"] < bound, f"пик {r['peak'] / 2**30:.2f} ГиБ при границе {bound / 2**30:.2f} ГиБ"
    # Сценарий дошёл до конца, и интерфейсу есть что показать: поле по ячейкам и кадры нагрева.
    assert r["stop_reason"] == "завершено"
    assert r["T_cells"] == r["n_cells"] and r["frames"] == [r["n_cells"]] * 3
    assert r["frame_t"] == [0.0, 2.0, 4.0]
    assert r["T_max"] > r["T_amb"]                        # обмотка греет
    # Тепловая сборка на настоящей сетке точна (баланс до 1e-9), магнитная часть сошлась.
    assert r["checks"]["Энергобаланс"] == PASS
    assert r["checks"]["Сходимость нелинейной итерации"] == PASS
