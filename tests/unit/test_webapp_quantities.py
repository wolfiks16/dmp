import numpy as np
import pytest

pytest.importorskip("gmsh")

from magcore.constants import MU0  # noqa: E402
from webapp.server import _OBJ, _build_object_model, _do_object_solve  # noqa: E402

# СПИСОК ВЕЛИЧИН ОДИН В 2D И 3D (docs/ui_rules.md §1): B, H и потеря B_r. Сервер 2D отдаёт H по ячейкам и потерю
# B_r по ячейкам магнитов; здесь — что это те же числа, что у решения, а не новая физика:
#   в воздухе ν = 1, поэтому H = B/μ₀ (с точностью до округления ответа);
#   в магните проекция H на ось намагничивания — это рабочее поле H_op, которое сервер отдаёт и раньше;
#   потеря B_r > 0 ровно там, где рабочая точка за коленом своей марки (новый магнит, без истории).

FLAT = {"name": "магнит", "kind": "rect", "params": {"cx": 0, "cy": 0, "w": 20, "h": 4, "angle": 0},
        "material": "ndfeb", "magnet_dir": [0, 1], "current": 0, "priority": 10}


def _solve(T):
    mid = _build_object_model({"objects": [FLAT], "default_mesh_mm": 1.0, "margin": 3.0})
    return mid, _do_object_solve({"model_id": mid, "T": T})


def test_H_in_air_is_B_over_mu0_and_in_the_magnet_its_axial_part_is_the_operating_field():
    mid, d = _solve(20.0)
    prob = _OBJ[mid]
    Bx, By = np.asarray(d["Bx"]), np.asarray(d["By"])
    Hx, Hy = np.asarray(d["Hx"]), np.asarray(d["Hy"])
    assert Hx.shape == Bx.shape == (prob.mesh.n_cells,)
    air = ~prob.magnet_mask()
    # B округлено до 1e-4 Тл (≈ 0,08 кА/м в H), H — до 0,01 кА/м
    tol = 1e-4 / MU0 / 1e3 + 0.01
    assert np.abs(Hx[air] - Bx[air] / MU0 / 1e3).max() < tol
    assert np.abs(Hy[air] - By[air] / MU0 / 1e3).max() < tol
    cells = np.asarray(d["demag_cells"])
    axis = np.asarray(prob.magnet_axis)[cells]
    h_par = Hx[cells] * axis[:, 0] + Hy[cells] * axis[:, 1]
    assert np.abs(h_par - np.asarray(d["demag_hop_kA"])).max() < 0.06     # оба округлены
    # Hmax округлён до 0,1 кА/м (±0,05), составляющие — до 0,01 (модуль ±0,007)
    assert d["Hmax_kA"] == pytest.approx(float(np.hypot(Hx, Hy).max()), abs=0.06)
    assert len(d["demag_loss_T"]) == cells.size and max(d["demag_loss_T"]) == 0.0   # при 20 °C магнит цел


def test_loss_is_positive_exactly_where_the_operating_point_is_past_the_knee():
    _, d = _solve(150.0)
    hop, knee = np.asarray(d["demag_hop_kA"]), np.asarray(d["demag_knee_cells_kA"])
    loss = np.asarray(d["demag_loss_T"])
    past, safe = hop < knee - 0.1, hop > knee + 0.1                         # вне полосы округления
    assert past.any() and (loss[past] > 0.0).all()
    assert (loss[safe] == 0.0).all()
