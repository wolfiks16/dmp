import numpy as np
import pytest

pytest.importorskip("gmsh")

from magcore.fem2d.model.scene import problem_to_scene  # noqa: E402
from webapp.server import _OBJ, _build_object_model, _do_object_solve  # noqa: E402

# Этап 3D-7 (то же в 2D): в ответе расчёта появился A_z в узлах — силовые линии 2D это его линии уровня с
# равным шагом (между соседними линиями одинаковый поток на единицу длины машины). Оракул: поле, которое
# интерфейс красит и по которому ставит стрелки, — это ротор того же A_z, B = (∂A/∂y, −∂A/∂x) в каждой
# ячейке; значит линии уровня A_z и есть линии этого поля. Плюс ось магнита в сцене (стрелки намагничивания).
# Обработчики вызываются напрямую: gmsh в 2D требует главного потока, а тестовый клиент FastAPI держит
# обработчик в рабочем потоке.

MAGNET = {"name": "магнит", "kind": "rect", "params": {"cx": 0, "cy": 0, "w": 10, "h": 4, "angle": 0},
          "material": "ndfeb", "magnet_dir": [0, 1], "current": 0, "priority": 10}   # как шлёт интерфейс
STEEL = {"name": "ярмо", "kind": "rect", "params": {"cx": 0, "cy": 5, "w": 14, "h": 3, "angle": 0},
         "material": "steel", "current": 0, "priority": 10}


def test_object_solve_returns_Az_whose_curl_is_the_drawn_field():
    mid = _build_object_model({"objects": [MAGNET, STEEL], "default_mesh_mm": 1.2, "margin": 3.0})
    scene = problem_to_scene(_OBJ[mid])
    d = _do_object_solve({"model_id": mid, "T": 20.0})
    A = np.asarray(d["A"], dtype=float)
    V = np.asarray(scene["vertices"], dtype=float) / 1000.0                  # сцена в мм → метры
    C = np.asarray(scene["cells"], dtype=int)
    assert A.size == V.shape[0]
    x, y, a = V[C][:, :, 0], V[C][:, :, 1], A[C]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    dax = (a * np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1)).sum(axis=1) / det
    day = (a * np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1)).sum(axis=1) / det
    # B в ответе округлено до 1e-4 Тл, A — до 1e-10 Вб/м (это ≤ 1e-7 Тл в градиенте на ячейке 1 мм)
    assert np.allclose(day, np.asarray(d["Bx"], dtype=float), rtol=0.0, atol=3e-4)
    assert np.allclose(-dax, np.asarray(d["By"], dtype=float), rtol=0.0, atol=3e-4)
    # ось магнита для стрелок намагничивания: в ячейках магнита — заданное направление, вне — ноль
    axis = np.asarray(scene["magnet_axis"], dtype=float)
    reg = np.asarray(scene["region"], dtype=int)
    rid = next(int(k) for k, r in scene["regions"].items() if r["name"] == "магнит")
    inside = reg == rid
    assert inside.any() and np.allclose(axis[inside], [0.0, 1.0], rtol=0.0, atol=1e-9)
    assert np.allclose(axis[~inside], 0.0, rtol=0.0, atol=1e-12)


def test_magnet_angle_from_the_interface_turns_the_axis():
    # Этап 3D-10: интерфейс шлёт поворот намагниченности в градусах; «по +Y» и 30° — ось (−sin 30°, cos 30°)
    # (против часовой). В сцене ось округлена до 1e-4.
    mid = _build_object_model({"objects": [dict(MAGNET, magnet_angle=30), STEEL], "default_mesh_mm": 1.2, "margin": 3.0})
    scene = problem_to_scene(_OBJ[mid])
    axis = np.asarray(scene["magnet_axis"], dtype=float)
    rid = next(int(k) for k, r in scene["regions"].items() if r["name"] == "магнит")
    inside = np.asarray(scene["region"], dtype=int) == rid
    assert inside.any() and np.allclose(axis[inside], [-0.5, 3 ** 0.5 / 2], rtol=0.0, atol=1e-4)
