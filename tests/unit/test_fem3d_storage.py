import dataclasses
import json

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model.materials import Air, LinearMaterial, MagnetMaterial, SteelMaterial
from magcore.fem3d import (
    GeoObject3D,
    auto_domain3d,
    build_object_problem3d,
    demag_summary,
    field_payload,
    magnetic_force_torque,
    restore_saved_field,
    solve_nonlinear3d,
)

# Этап 3D-9: решение 3D в файле расчёта — сетка и узловой потенциал φ, остальное — одна оценка тех же
# законов при φ. Оракулы:
#  (1) тождество: решили → упаковали через JSON, как в файл → восстановили — модель и поле те же ДО БИТА:
#      регионы и оси намагничивания, H, B, проницаемость, намагниченность, коэнергия, доля r, карта риска,
#      невязка; отсюда те же сила, момент и сводка риска. Задача с насыщенной сталью и магнитом за коленом
#      (оба закона нелинейны), обе границы, с внешним полем и без;
#  (2) годность проверяется уравнениями текущего кода: то же φ при другой температуре или другом материале —
#      невязка растёт на порядки, поле отвергается; при тех же — невязка ровно та, что при решении;
#  (3) повреждённое сохранённое поле — понятная ошибка, а не молча неверное поле.

MM = 1.0e-3


def _model():
    pytest.importorskip("gmsh")
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    cube = GeoObject3D("m", "box", {"lx": 10 * MM, "ly": 10 * MM, "lz": 10 * MM}, mag, magnet_dir="axial",
                       mesh_size=1.5 * MM)
    fe = GeoObject3D("fe", "box", {"lx": 8 * MM, "ly": 6 * MM, "lz": 3 * MM}, SteelMaterial(m270_35a_bh_curve()),
                     center=(1 * MM, 0.5 * MM, 7 * MM), rotation=(0.1, 0.0, 0.2), mesh_size=1.5 * MM)
    objs = [cube, fe]
    return objs, auto_domain3d(objs, material=Air(), margin_frac=2.0)


def _as_in_file(payload: dict) -> dict:
    return json.loads(json.dumps(payload))            # файл расчёта — JSON


@pytest.mark.parametrize("bc, H0", [("neumann", None), ("dirichlet", (0.0, 0.0, -2.0e5))])
def test_saved_field_restores_bit_for_bit(bc, H0):
    objs, dom = _model()
    p = build_object_problem3d(objs, dom, default_mesh_size=1.5 * MM, T=155.0)
    f = solve_nonlinear3d(p, bc=bc, applied_field=H0, solver="cg")
    fe = np.asarray(p.cell_region) == 2
    assert f.converged and f.risk.n_demagnetized > 0                     # предпосылки: магнит за коленом,
    assert np.linalg.norm(f.B_cells[fe], axis=1).max() > 1.5             # сталь в насыщении
    saved = restore_saved_field(_as_in_file(field_payload(f)), objs, dom)
    q, g = saved.problem, saved.field
    assert saved.ok and saved.residual == saved.stored_residual == f.residual
    assert np.array_equal(q.mesh.vertices, p.mesh.vertices) and np.array_equal(q.mesh.cells, p.mesh.cells)
    assert np.array_equal(q.cell_region, p.cell_region) and np.array_equal(q.magnet_axis, p.magnet_axis)
    assert q.T == p.T and [(i, r.name, r.material) for i, r in q.regions.items()] == \
        [(i, r.name, r.material) for i, r in p.regions.items()]
    for key in ("phi", "H_cells", "B_cells", "mu_cells", "M_cells", "volumes", "coenergy_density", "retention",
                "applied_field"):
        assert np.array_equal(getattr(g, key), getattr(f, key)), key
    for key in ("margin", "Br_eff", "loss", "retention"):
        assert np.array_equal(getattr(g.risk, key), getattr(f.risk, key)), key
    assert g.bc == f.bc and demag_summary(g) == demag_summary(f)
    for body in ("m", "fe"):
        a, b = magnetic_force_torque(f, body), magnetic_force_torque(g, body)
        assert np.array_equal(a.force, b.force) and np.array_equal(a.torque, b.torque)


def test_saved_field_is_checked_by_the_current_equations():
    # Л-80: сохранённые числа без версии расчётной формулы молча устаревают. Здесь версия не нужна: φ
    # проверяется уравнениями текущего кода. Изменение уравнений изображаем другой температурой (закон
    # магнита другой) и другим материалом стали (линейная вместо кривой) — φ тот же.
    objs, dom = _model()
    p = build_object_problem3d(objs, dom, default_mesh_size=1.5 * MM, T=155.0)
    f = solve_nonlinear3d(p, solver="cg")
    payload = _as_in_file(field_payload(f))
    same = restore_saved_field(payload, objs, dom)
    assert same.ok and same.residual == same.stored_residual
    hotter = restore_saved_field(dict(payload, T=165.0), objs, dom)
    linear = restore_saved_field(payload, [objs[0], dataclasses.replace(objs[1], material=LinearMaterial(1000.0))],
                                 dom)
    for s in (hotter, linear):
        assert not s.ok and not s.field.converged
        assert s.residual > 1.0e3 * max(s.stored_residual, 1.0e-9)       # не на грани допуска — на порядки


def test_damaged_saved_field_is_a_clear_error():
    objs, dom = _model()
    p = build_object_problem3d(objs, dom, default_mesh_size=3.0 * MM)
    payload = _as_in_file(field_payload(solve_nonlinear3d(p, solver="cg")))
    bad = [dict(payload, format="другое"), dict(payload, version=99), dict(payload, n_cells=payload["n_cells"] + 1),
           {k: v for k, v in payload.items() if k != "phi"}, dict(payload, phi="!!!"), dict(payload, T="горячо"),
           dict(payload, bc="robin"), dict(payload, applied_field=[1.0, 2.0]), "не словарь"]
    for b in bad:
        with pytest.raises(ValueError):
            restore_saved_field(b, objs, dom)
    with pytest.raises(ValueError, match="cell_region"):                  # объект пропал из модели
        restore_saved_field(payload, objs[:1], dom)
