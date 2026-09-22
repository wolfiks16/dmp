import dataclasses
import math

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_model import magnet_from_datasheet
from magcore.fem2d.model.materials import Air, MagnetMaterial, SteelMaterial
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem3d import GeoObject3D, auto_domain3d, build_object_problem3d, solve_linear3d
from magcore.fem3d.fieldlines import (
    STOP_BOUNDARY,
    STOP_MAGNET,
    nodal_field,
    trace_field_lines,
    trace_section_lines,
)
from magcore.fem3d.scalar import ScalarField3D

# Этап 3D-7: силовые линии поля B. Оракулы:
#  (1) однородное поле — линии обязаны быть строго прямыми и уходить на границу области (проверяет шаг,
#      поиск ячейки и восстановление поля в узлах: у однородного поля оно точное);
#  (2) поле точечного диполя, записанное в ячейки, — вдоль линии сохраняется r/sin²θ (уравнение линии
#      диполя); ошибка убывает при сгущении сетки;
#  (3) настоящее решение намагниченного шара: поток, по которому расставлены начала линий, сходится к
#      точному (2/3)·μ₀·M·πR² (поток через экваториальное сечение), почти все линии замыкаются —
#      заканчиваются входом в магнит, — и число линий, пересекающих экваториальную плоскость вне шара,
#      совпадает с Φ/ΔΦ;
#  (4) восстановление поля в узлах не меняет однородное поле и сохраняет скачок на границе материалов;
#  (5) негативные случаи: нет магнитов, неверное число линий, несуществующее тело.

MM = 1.0e-3
R = 4 * MM
RIGID = magnet_from_datasheet("rigid", "rigid", (0.0, 0.0, 1.0), Br=1.2, Hcb=1.2 / MU0, Hk=1.1e6, Hcj=1.6e6)
M_RIGID = 1.2 / MU0                                   # намагниченность жёсткого магнита, А/м
FLUX_EXACT = 2.0 / 3.0 * MU0 * M_RIGID * math.pi * R ** 2      # поток шара через экватор [Вб]


def _ball(h, margin=2.0):
    pytest.importorskip("gmsh")
    ball = GeoObject3D("m", "sphere", {"r": R}, MagnetMaterial(RIGID), magnet_dir="axial", mesh_size=h)
    dom = auto_domain3d([ball], material=Air(), margin_frac=margin, mesh_size=4 * h)
    return build_object_problem3d([ball], dom, default_mesh_size=h)


def _field(prob, B):
    """Готовое решение с заданным полем по ячейкам — для проверки одной трассировки, без решателя."""
    nc = prob.mesh.n_cells
    return ScalarField3D(problem=prob, phi=np.zeros(prob.mesh.n_vertices), H_cells=B / MU0, B_cells=B,
                         mu_cells=np.tile(np.eye(3), (nc, 1, 1)), M_cells=np.zeros((nc, 3)),
                         volumes=prob.mesh.cell_volumes(), bc="neumann", applied_field=np.zeros(3), residual=0.0)


def _spread(fl, r_lo, r_hi, sin2_min=0.2):
    """Разброс r/sin²θ вдоль каждой линии на участке r_lo < r < r_hi и вдали от оси."""
    out = []
    for i in range(fl.n_lines):
        q = fl.line(i)
        rr = np.linalg.norm(q, axis=1)
        s2 = 1.0 - (q[:, 2] / rr) ** 2
        m = (rr > r_lo) & (rr < r_hi) & (s2 > sin2_min)
        if m.sum() < 5:
            continue
        v = rr[m] / s2[m]
        out.append(float(np.ptp(v) / np.mean(v)))
    return np.array(out)


def test_uniform_field_lines_are_straight():
    prob = _ball(R / 4)
    f = solve_linear3d(prob, bc="neumann")
    uni = dataclasses.replace(f, B_cells=np.tile([0.0, 0.0, 1.0], (prob.mesh.n_cells, 1)))
    fl = trace_field_lines(uni, n_lines=30)
    assert fl.n_lines == 30
    # Начала стоят на гранях шара с B·n > 0; у граней близ экватора линия вверх снова входит в шар — это
    # правильный конец участка, остальные уходят на границу области.
    assert np.all((fl.stop == STOP_BOUNDARY) | (fl.stop == STOP_MAGNET))
    assert np.mean(fl.stop == STOP_BOUNDARY) > 0.8
    hi = prob.mesh.vertices[:, 2].max()
    lo = prob.mesh.vertices[:, 2].min()
    for i in range(fl.n_lines):
        q = fl.line(i)
        assert q.shape[0] > 3
        assert np.abs(q[:, :2] - q[0, :2]).max() == 0.0                    # ни шага в сторону
        assert np.all(np.diff(q[:, 2]) > 0.0)                              # строго вверх, по полю
        if fl.stop[i] == STOP_BOUNDARY:
            assert q[-1, 2] > hi - 0.05 * (hi - lo)
    assert np.allclose(fl.values, 1.0)                                     # |B| в точках — то самое поле


def test_dipole_field_lines_keep_r_over_sin2():
    # Поле точечного диполя, записанное в ячейки (без решателя — проверяется одна трассировка): точная линия
    # диполя — r = C·sin²θ. Замерено: разброс r/sin²θ вдоль линии 0,157 при h = R/3, 0,027 при R/6 и 0,009
    # при R/12, то есть трассировка сходится к точной линии быстрее первого порядка.
    m = np.array([0.0, 0.0, 1.0])
    err = []
    for k in (3, 6):
        prob = _ball(R / k)
        c = prob.mesh.cell_centroids()
        rn = np.linalg.norm(c, axis=1)
        dip = 1.0e-7 * (3.0 * c * (c @ m)[:, None] / rn[:, None] ** 5 - m / rn[:, None] ** 3)
        dip[rn < 1.2 * R] = [0.0, 0.0, 2.0e-7 / R ** 3]                    # внутри шара — как у диполя на оси
        fl = trace_field_lines(_field(prob, dip), n_lines=24)
        s = _spread(fl, 1.5 * R, 3.0 * R)
        assert s.size >= 8
        err.append(float(np.median(s)))
    assert err[0] < 0.25 and err[1] < 0.05 and err[1] < 0.5 * err[0]


@pytest.mark.slow
def test_magnetized_ball_lines_close_and_carry_the_exact_flux():
    # Настоящее решение: поток начал линий сходится к (2/3)·μ₀·M·πR², линии замыкаются через магнит,
    # а число линий через экваториальную плоскость вне шара равно Φ/ΔΦ.
    seen = []
    for k in (4, 8):
        prob = _ball(R / k)
        f = solve_linear3d(prob, bc="neumann")
        fl = trace_field_lines(f, n_lines=60)
        seen.append(abs(fl.delta_flux * fl.n_lines / FLUX_EXACT - 1.0))
        assert np.mean(fl.stop == STOP_MAGNET) >= 0.95                     # почти все вернулись в магнит
        if k == 8:
            crossings = 0
            for i in range(fl.n_lines):
                q = fl.line(i)
                r_xy = np.hypot(q[:, 0], q[:, 1])
                up = (q[:-1, 2] > 0.0) & (q[1:, 2] <= 0.0) & (r_xy[:-1] > R)
                down = (q[:-1, 2] < 0.0) & (q[1:, 2] >= 0.0) & (r_xy[:-1] > R)
                crossings += int(up.sum() + down.sum())
            assert 0.8 * fl.n_lines <= crossings <= 1.25 * fl.n_lines      # каждая линия пересекает экватор раз
    assert seen[0] < 0.20 and seen[1] < 0.5 * seen[0]                      # ошибка потока убывает с сеткой


def test_nodal_field_is_exact_on_uniform_and_keeps_the_jump():
    # Восстановление в узлах: однородное поле восстанавливается точно; на стыке материалов у каждого тела
    # свой узловой вектор, поэтому скачок сохраняется (иначе поле размазалось бы через границу магнита).
    pytest.importorskip("gmsh")
    mag = GeoObject3D("m", "box", {"lx": 6 * MM, "ly": 6 * MM, "lz": 4 * MM}, MagnetMaterial(RIGID),
                      magnet_dir="axial", mesh_size=1.5 * MM)
    fe = GeoObject3D("fe", "box", {"lx": 6 * MM, "ly": 6 * MM, "lz": 4 * MM}, SteelMaterial(m270_35a_bh_curve()),
                     center=(0.0, 0.0, 4 * MM), mesh_size=1.5 * MM)
    dom = auto_domain3d([mag, fe], material=Air(), margin_frac=1.0, mesh_size=6 * MM)
    prob = build_object_problem3d([mag, fe], dom, default_mesh_size=1.5 * MM)
    nc = prob.mesh.n_cells
    nodal, node_of = nodal_field(prob, np.tile([0.3, -0.2, 0.7], (nc, 1)))
    assert np.allclose(nodal, [0.3, -0.2, 0.7], rtol=0.0, atol=1e-15)
    reg = np.asarray(prob.cell_region)
    vals = np.zeros((nc, 3))
    vals[reg == 1] = [0.0, 0.0, 1.0]                                        # в магните одно, в стали другое
    vals[reg == 2] = [0.0, 0.0, -1.0]
    nodal, node_of = nodal_field(prob, vals)
    for rid, want in ((1, 1.0), (2, -1.0)):
        cells = np.where(reg == rid)[0]
        assert np.allclose(nodal[node_of[cells]][:, :, 2], want, rtol=0.0, atol=1e-15)


def _line_lengths(fl):
    return np.array([np.linalg.norm(np.diff(fl.line(i), axis=0), axis=1).sum() for i in range(fl.n_lines)])


def test_section_lines_lie_in_the_plane_and_are_evenly_spread():
    # Этап 3D-7, линии на разрезе. Оракулы: (1) однородное поле — прямые в плоскости; (2) плоскость симметрии
    # намагниченного шара (y = 0): поле лежит в плоскости, поэтому линии — настоящие линии поля, и вдоль них
    # сохраняется r/sin²θ, а доля поля, выходящая из плоскости, близка к нулю; (3) линии не жмутся друг к другу.
    prob = _ball(R / 5)
    f = solve_linear3d(prob, bc="neumann")
    uni = dataclasses.replace(f, B_cells=np.tile([0.0, 0.0, 1.0], (prob.mesh.n_cells, 1)))
    sl = trace_section_lines(uni, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), n_lines=20)
    assert sl.n_lines >= 18 and sl.delta_flux == 0.0          # у однородного поля выходит ровно n_lines
    assert np.abs(sl.points[:, 1]).max() == 0.0                            # точки ровно в плоскости
    assert sl.out_of_plane < 1e-12
    for i in range(sl.n_lines):
        q = sl.line(i)
        assert np.abs(q[:, 0] - q[0, 0]).max() == 0.0                      # прямые вдоль поля
        assert np.all(np.diff(q[:, 2]) > 0.0)

    fl = trace_section_lines(f, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), n_lines=40)
    assert np.abs(fl.points[:, 1]).max() == 0.0
    assert fl.out_of_plane < 0.05                                          # плоскость симметрии: поле в ней
    s = _spread(fl, 1.5 * R, 3.0 * R)
    assert s.size >= 3 and np.median(s) < 0.3
    # Линии не сливаются: подойти ближе половины шага расстановки может только последняя точка ветви —
    # на ней линия и останавливается, значит таких точек у линии не больше двух.
    span = float(np.ptp(prob.mesh.vertices[:, 0]))
    d_sep = span / 40.0
    pts = [fl.line(i) for i in range(fl.n_lines)]
    for i in range(len(pts)):
        others = np.concatenate([pts[j] for j in range(len(pts)) if j != i])
        near = np.linalg.norm(others[:, None, :] - pts[i][None, :, :], axis=2).min(axis=0)
        assert int((near < 0.5 * 0.5 * d_sep).sum()) <= 2, i


def test_section_lines_refuse_a_plane_across_the_field():
    # В экваториальной плоскости шара поле перпендикулярно ей, и проекция — численный шум: рисовать по ней
    # линии нельзя. Проверяем, что от такой плоскости почти ничего не остаётся (порог SECTION_MIN_IN_PLANE).
    prob = _ball(R / 5)
    f = solve_linear3d(prob, bc="neumann")
    along = trace_section_lines(f, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), n_lines=40)
    across = trace_section_lines(f, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), n_lines=40)
    assert _line_lengths(across).sum() < 0.1 * _line_lengths(along).sum()
    with pytest.raises(ValueError):                                        # плоскость вне модели (y = 10 м)
        trace_section_lines(f, (0.0, 10.0, 0.0), (0.0, 1.0, 0.0), n_lines=10)


def test_invalid_field_line_requests():
    prob = _ball(R / 3)
    f = solve_linear3d(prob, bc="neumann")
    with pytest.raises(ValueError):
        trace_field_lines(f, n_lines=0)
    with pytest.raises(ValueError, match="нет магнитов"):
        trace_field_lines(f, objects=["нет такого"])
    with pytest.raises(ValueError):
        trace_section_lines(f, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), n_lines=0)
    air = GeoObject3D("air", "box", {"lx": 4 * MM, "ly": 4 * MM, "lz": 4 * MM}, Air(), mesh_size=2 * MM)
    dom = auto_domain3d([air], material=Air(), margin_frac=1.0, mesh_size=4 * MM)
    no_magnet = build_object_problem3d([air], dom, default_mesh_size=2 * MM)
    with pytest.raises(ValueError, match="нет магнитов"):
        trace_field_lines(solve_linear3d(no_magnet, bc="neumann"), n_lines=5)
