import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.fem2d.mesh import TriangleMesh
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.model import Air, MagnetMaterial, Problem2D, Region2D, solve_problem2d
from magcore.fem2d.model.postproc import operating_point

# НЕСКОЛЬКО МАГНИТОВ И РАЗНЫЕ МАРКИ В ОДНОЙ МОДЕЛИ.
# (1) Марка — это закон материала (B_r, H_cB, H_k, H_cJ, μ⊥, два температурных коэффициента, T0), а не объект в
#     памяти. Прежняя проверка сравнивала объекты, и сервер, создающий каждому телу свою копию материала,
#     получал отказ «несколько марок магнита» уже на двух одинаковых магнитах. Теперь магниты одной марки,
#     созданные порознь, считаются ровно как с общим материалом — бит в бит.
# (2) Разные марки: закон у каждой ячейки свой. ОРАКУЛ — модель из двух НЕСВЯЗАННЫХ частей (две области сетки
#     без общих узлов, у каждой своя граница A = 0 / φ = 0): задача распадается на две независимые, и решение
#     каждой части обязано совпасть с расчётом этой части в одиночку — поле, карта риска, вердикт. Плюс: в
#     каждой ячейке рабочая точка лежит на кривой СВОЕЙ марки и не лежит на кривой чужой.
# Режим выбран так, чтобы марки вели себя по-разному: плоский магнит, намагниченный поперёк, при 150 °C —
# NdFeB уходит за колено, SmCo остаётся на линии возврата.

T_HOT = 150.0
ND, SM = n42sh_magnet((1.0, 0.0, 0.0)), sm2co17_magnet((1.0, 0.0, 0.0))


# ------------------------------------------------------------------------------------------------ 2D
def _flat_magnet_part(x0: float = 0.0, n: int = 40):
    """Квадрат 40×40 мм (A = 0 на границе) с плоским магнитом 20×4 мм в центре; ось — поперёк (+y); сетка n×n
    квадратов (n = 40 — шаг 1 мм)."""
    mesh = build_structured_rectangle_tri_mesh(n, n, x0=x0, x1=x0 + 0.04, y0=0.0, y1=0.04)
    cen = np.array([mesh.cell_centroid(c) for c in range(mesh.n_cells)])
    mag = (np.abs(cen[:, 0] - (x0 + 0.02)) < 0.010) & (np.abs(cen[:, 1] - 0.02) < 0.002)
    return mesh, mag


def _problem2d(mesh, parts, T=T_HOT):
    """parts: [(маска ячеек, MagnetMaterial)] — каждой части свой регион 1, 2, …; остальное — воздух (0)."""
    reg = np.zeros(mesh.n_cells, dtype=int)
    regions = {0: Region2D(0, "воздух", Air())}
    axis = np.zeros((mesh.n_cells, 2))
    for k, (mask, mat) in enumerate(parts, start=1):
        reg[mask] = k
        regions[k] = Region2D(k, f"магнит {k}", mat)
        axis[mask] = (0.0, 1.0)
    return Problem2D(mesh=mesh, cell_region=reg, regions=regions, magnet_axis=axis, T=T)


def _merged_mesh(m1: TriangleMesh, m2: TriangleMesh) -> TriangleMesh:
    """Две сетки в одной, без общих узлов: вторая — отдельная связная часть."""
    return TriangleMesh(vertices=np.vstack([m1.vertices, m2.vertices]),
                        cells=np.vstack([m1.cells, m2.cells + m1.n_vertices]))


def _b_par(sol):
    idx = sol.risk.cell_indices
    return np.einsum("ij,ij->i", sol.field.B_cells[idx], np.asarray(sol.problem.magnet_axis)[idx])


def test_grade_is_the_material_law_not_the_object():
    a, b = n42sh_magnet((1.0, 0.0, 0.0)), n42sh_magnet((0.0, 0.0, 1.0))      # та же марка, другая ось
    assert a is not b and a.law_key() == b.law_key()
    assert a.law_key() != SM.law_key()


def test_one_grade_in_separate_copies_is_one_group_and_two_grades_are_two():
    mesh, mask = _flat_magnet_part(0.0)
    left, right = mask & (np.arange(mask.size) % 2 == 0), mask & (np.arange(mask.size) % 2 == 1)
    apart = _problem2d(mesh, [(left, MagnetMaterial(n42sh_magnet((1.0, 0.0, 0.0)))),
                              (right, MagnetMaterial(n42sh_magnet((1.0, 0.0, 0.0))))])
    (law, cells), = apart.magnet_groups()                                      # одна марка — одна группа
    assert np.array_equal(cells, apart.magnet_mask()) and law.law_key() == ND.law_key()
    assert apart.magnet().law_key() == ND.law_key()
    two = _problem2d(mesh, [(left, MagnetMaterial(SM)), (right, MagnetMaterial(ND))])
    assert two.validate() == []                                                # разные марки — не ошибка постановки
    assert [m.law_key() for m, _ in two.magnet_groups()] == [SM.law_key(), ND.law_key()]   # по первому региону
    assert [c.sum() for _, c in two.magnet_groups()] == [left.sum(), right.sum()]
    with pytest.raises(ValueError, match="несколько марок"):                  # одного закона на задачу нет
        two.magnet()


def test_merged_risk_map_keeps_each_cell_with_its_own_grade():
    from types import SimpleNamespace

    from magcore.constants import MU0
    from magcore.hybrid.magnet_demag import DemagRiskMap, compute_demag_risk_map, merge_risk_maps

    n = 8
    h = np.linspace(-700e3, -50e3, n)                        # поле вдоль оси [А/м]: от глубоко за коленом до безопасного
    res = SimpleNamespace(H_cells=MU0 * np.c_[h, np.zeros(n)])
    nd_cells = np.arange(n) % 2 == 1                          # марки вперемешку, SmCo — первой группой
    sm = compute_demag_risk_map(SM, res, ~nd_cells, T_HOT, axis=(1.0, 0.0))
    nd = compute_demag_risk_map(ND, res, nd_cells, T_HOT, axis=(1.0, 0.0))
    m = merge_risk_maps([sm, nd])
    assert np.array_equal(m.cell_indices, np.arange(n))       # по возрастанию номера, как magnet_mask
    for part, cells in ((sm, ~nd_cells), (nd, nd_cells)):
        for f in ("H_par", "margin", "Br_eff", "loss", "demagnetized", "retention", "beyond_hcj",
                  "knee_field_cells", "Br_nominal_cells"):
            assert np.array_equal(getattr(m, f)[cells], getattr(part, f)), f
    assert m.knee_field is None and m.Br_nominal is None      # общего колена у двух марок нет
    assert m.n_demagnetized == sm.n_demagnetized + nd.n_demagnetized > 0
    same = merge_risk_maps([compute_demag_risk_map(ND, res, ~nd_cells, T_HOT, axis=(1.0, 0.0)), nd])
    assert same.knee_field == nd.knee_field and same.Br_nominal == nd.Br_nominal   # марка одна — скаляры есть
    with pytest.raises(ValueError, match="пересекаются"):
        merge_risk_maps([nd, nd])
    with pytest.raises(ValueError, match="температурах"):
        merge_risk_maps([nd, compute_demag_risk_map(SM, res, ~nd_cells, 20.0, axis=(1.0, 0.0))])
    z = np.zeros(2)
    with pytest.raises(ValueError, match="по ячейкам"):       # ни общего значения, ни значений по ячейкам
        DemagRiskMap(cell_indices=np.arange(2), H_par=z, margin=z, Br_eff=z, loss=z,
                     demagnetized=z < 0.0, T=20.0, Br_nominal=None, knee_field=-1.0)


def test_two_magnets_of_one_grade_count_exactly_as_with_a_shared_material():
    mesh = build_structured_rectangle_tri_mesh(48, 40, x0=0.0, x1=0.06, y0=0.0, y1=0.04)
    cen = np.array([mesh.cell_centroid(c) for c in range(mesh.n_cells)])
    left = (np.abs(cen[:, 0] - 0.018) < 0.008) & (np.abs(cen[:, 1] - 0.02) < 0.002)
    right = (np.abs(cen[:, 0] - 0.042) < 0.008) & (np.abs(cen[:, 1] - 0.02) < 0.002)
    shared = MagnetMaterial(n42sh_magnet((1.0, 0.0, 0.0)))
    one = solve_problem2d(_problem2d(mesh, [(left, shared), (right, shared)]))
    apart = solve_problem2d(_problem2d(mesh, [(left, MagnetMaterial(n42sh_magnet((1.0, 0.0, 0.0)))),
                                              (right, MagnetMaterial(n42sh_magnet((1.0, 0.0, 0.0))))]))
    assert apart.converged
    assert np.array_equal(one.field.B_cells, apart.field.B_cells)
    assert np.array_equal(one.risk.margin, apart.risk.margin) and np.array_equal(one.risk.loss, apart.risk.loss)


@pytest.mark.parametrize("method", ["newton", "picard"])
def test_two_grades_in_disconnected_parts_equal_each_part_alone(method):
    # Пикар (прежняя схема, эталон) за коленом устойчив лишь при малой релаксации (Л-107): здесь 0,2 и 0,3 не
    # сходятся и за 4000 итераций, 0,1 — за 178 на сетке 1 мм и за 41 на сетке 2 мм. Итерация Пикара дорогая,
    # поэтому ему — сетка 2 мм: оракулу (несвязанные части) сетка не важна.
    n = 40 if method == "newton" else 20
    ma, mask_a = _flat_magnet_part(0.0, n)
    mb, mask_b = _flat_magnet_part(0.1, n)
    both = _merged_mesh(ma, mb)
    na = ma.n_cells
    kw = dict(method=method, tol=1e-12, max_iter=200) if method == "newton" else \
        dict(method=method, tol=1e-10, max_iter=1000, relaxation=0.1)
    sol = solve_problem2d(_problem2d(both, [(np.r_[mask_a, np.zeros(mb.n_cells, bool)], MagnetMaterial(ND)),
                                             (np.r_[np.zeros(na, bool), mask_b], MagnetMaterial(SM))]), **kw)
    alone_a = solve_problem2d(_problem2d(ma, [(mask_a, MagnetMaterial(ND))]), **kw)
    alone_b = solve_problem2d(_problem2d(mb, [(mask_b, MagnetMaterial(SM))]), **kw)
    assert sol.converged and alone_a.converged and alone_b.converged
    tol_b = 1e-9 if method == "newton" else 1e-7
    assert np.abs(sol.field.B_cells[:na] - alone_a.field.B_cells).max() < tol_b
    assert np.abs(sol.field.B_cells[na:] - alone_b.field.B_cells).max() < tol_b

    # карта риска — по ячейкам своей марки: колено, номинальная B_r, запас, потеря
    idx = sol.risk.cell_indices
    in_a = idx < na
    assert np.array_equal(idx, np.where(sol.problem.magnet_mask())[0])          # порядок ячеек — как у маски
    assert np.allclose(sol.risk.knee_field_cells[in_a], ND.knee_field(T_HOT))
    assert np.allclose(sol.risk.knee_field_cells[~in_a], SM.knee_field(T_HOT))
    assert np.allclose(sol.risk.Br_nominal_cells[in_a], ND.Br(T_HOT))
    assert np.allclose(sol.risk.Br_nominal_cells[~in_a], SM.Br(T_HOT))
    assert np.allclose(sol.risk.margin[in_a], alone_a.risk.margin, rtol=0, atol=1e-3)     # А/м
    assert np.allclose(sol.risk.margin[~in_a], alone_b.risk.margin, rtol=0, atol=1e-3)
    assert sol.risk.demagnetized[in_a].sum() == alone_a.risk.n_demagnetized > 0          # NdFeB за коленом
    assert sol.risk.demagnetized[~in_a].sum() == alone_b.risk.n_demagnetized == 0        # SmCo цел
    assert sol.risk.n_demagnetized == alone_a.risk.n_demagnetized


def test_each_cell_sits_on_its_own_grade_law_and_not_on_the_other():
    ma, mask_a = _flat_magnet_part(0.0)
    mb, mask_b = _flat_magnet_part(0.1)
    na = ma.n_cells
    sol = solve_problem2d(_problem2d(_merged_mesh(ma, mb), [(np.r_[mask_a, np.zeros(mb.n_cells, bool)], MagnetMaterial(ND)),
                                                            (np.r_[np.zeros(na, bool), mask_b], MagnetMaterial(SM))]),
                          tol=1e-12)
    idx, h = sol.risk.cell_indices, sol.risk.H_par
    b = _b_par(sol)
    in_a = idx < na
    own = np.where(in_a, ND.branch_parallel(h, T_HOT, np.ones(h.size))[0], SM.branch_parallel(h, T_HOT, np.ones(h.size))[0])
    other = np.where(in_a, SM.branch_parallel(h, T_HOT, np.ones(h.size))[0], ND.branch_parallel(h, T_HOT, np.ones(h.size))[0])
    assert np.abs(own - b).max() < 1e-11
    assert np.abs(other - b).min() > 1e-3                 # на чужой кривой не лежит ни одна ячейка


def test_operating_point_reports_the_knee_of_each_cell():
    ma, mask_a = _flat_magnet_part(0.0)
    mb, mask_b = _flat_magnet_part(0.1)
    na = ma.n_cells
    sol = solve_problem2d(_problem2d(_merged_mesh(ma, mb), [(np.r_[mask_a, np.zeros(mb.n_cells, bool)], MagnetMaterial(ND)),
                                                            (np.r_[np.zeros(na, bool), mask_b], MagnetMaterial(SM))]))
    op = operating_point(sol)
    in_a = op.cell_indices < na
    assert np.allclose(op.knee_field_cells[in_a], ND.knee_field(T_HOT))
    assert np.allclose(op.knee_field_cells[~in_a], SM.knee_field(T_HOT))
    past = op.H_op < op.knee_field_cells
    assert op.fraction_past_knee() == pytest.approx(float(op.cell_volume[past].sum() / op.cell_volume.sum()))
    assert past[in_a].any() and not past[~in_a].any()
    assert op.knee_field is None                           # одного колена на модель из двух марок нет
    assert sol.risk.knee_field is None and sol.risk.Br_nominal is None


def test_single_grade_models_keep_their_scalar_knee():
    mesh, mask = _flat_magnet_part(0.0)
    sol = solve_problem2d(_problem2d(mesh, [(mask, MagnetMaterial(ND))]))
    assert sol.risk.knee_field == pytest.approx(ND.knee_field(T_HOT))
    assert sol.risk.Br_nominal == pytest.approx(ND.Br(T_HOT))
    assert operating_point(sol).knee_field == pytest.approx(ND.knee_field(T_HOT))


# ------------------------------------------------------------------------------------------------ 3D
def _plate_problem3d(magnet, *, T=T_HOT, h=2.0e-3):
    """Пластина 20×20×4 мм, намагниченная поперёк (по z), в воздухе. При 150 °C NdFeB уходит за колено
    (77 % объёма, запас −61 кА/м), SmCo цел (запас +198 кА/м); шар для этого слишком «толстый» — у него
    NdFeB при 150 °C без внешнего поля ещё выше колена (+100 кА/м)."""
    pytest.importorskip("gmsh")
    from magcore.fem3d import GeoObject3D, auto_domain3d, build_object_problem3d
    s = GeoObject3D("пластина", "box", {"lx": 20e-3, "ly": 20e-3, "lz": 4e-3}, MagnetMaterial(magnet),
                    magnet_dir="axial", mesh_size=h)
    return build_object_problem3d([s], auto_domain3d([s], material=Air(), margin_frac=1.0),
                                  default_mesh_size=2 * h, T=T)


def _merge3d(p1, p2, shift=0.2):
    """Две 3D-задачи в одной, без общих узлов; воздух — общий регион 0, магниты — регионы 1 и 2."""
    from dataclasses import replace

    from magcore.fem3d.mesh import TetMesh3D
    from magcore.fem3d.problem import Region3D
    v2 = p2.mesh.vertices + np.array([shift, 0.0, 0.0])
    mesh = TetMesh3D(np.vstack([p1.mesh.vertices, v2]), np.vstack([p1.mesh.cells, p2.mesh.cells + p1.mesh.n_vertices]))
    assert set(p1.regions) == {0, 1} and set(p2.regions) == {0, 1}
    reg2 = np.where(np.asarray(p2.cell_region) == 1, 2, 0)
    regions = {0: p1.regions[0], 1: replace(p1.regions[1], name="магнит NdFeB"),
               2: Region3D(2, "магнит SmCo", p2.regions[1].material)}
    return replace(p1, mesh=mesh, cell_region=np.r_[np.asarray(p1.cell_region), reg2], regions=regions,
                   magnet_axis=np.vstack([p1.magnet_axis, p2.magnet_axis]))


def test_3d_two_grades_in_disconnected_parts_equal_each_part_alone():
    from magcore.fem3d import solve_nonlinear3d
    from magcore.fem3d.postprocess import demag_summary, flux_loss

    pa, pb = _plate_problem3d(ND), _plate_problem3d(SM)
    both = _merge3d(pa, pb)
    na = pa.mesh.n_cells
    f = solve_nonlinear3d(both, bc="dirichlet", tol=1e-11)
    fa = solve_nonlinear3d(pa, bc="dirichlet", tol=1e-11)
    fb = solve_nonlinear3d(pb, bc="dirichlet", tol=1e-11)
    assert f.converged and fa.converged and fb.converged
    assert np.abs(f.B_cells[:na] - fa.B_cells).max() < 1e-8
    assert np.abs(f.B_cells[na:] - fb.B_cells).max() < 1e-8
    assert np.array_equal(f.retention[:na] < 1.0, fa.retention < 1.0)
    assert np.array_equal(f.retention[na:] < 1.0, fb.retention < 1.0)

    s, sa, sb = demag_summary(f), demag_summary(fa), demag_summary(fb)
    assert s["магнит NdFeB"].retained == pytest.approx(sa["пластина"].retained, abs=1e-9)
    assert s["магнит SmCo"].retained == pytest.approx(sb["пластина"].retained, abs=1e-9)
    assert s["магнит NdFeB"].damaged_fraction == pytest.approx(sa["пластина"].damaged_fraction, abs=1e-12)
    assert s["магнит NdFeB"].damaged_fraction > 0.5 and s["магнит SmCo"].damaged_fraction == 0.0
    assert s["магнит SmCo"].retained == pytest.approx(1.0, abs=1e-12)     # сумма по ячейкам — до округления

    # вердикт — потеря потока каждого магнита при 20 °C — тот же, что у каждого в одиночку
    loss = flux_loss(both, f.retention, bc="dirichlet", solver="direct")
    loss_a = flux_loss(pa, fa.retention, bc="dirichlet", solver="direct")
    assert loss["магнит NdFeB"] == pytest.approx(loss_a["пластина"], abs=1e-6) and loss["магнит NdFeB"] > 0.05
    assert loss["магнит SmCo"] == pytest.approx(0.0, abs=1e-12)


def test_3d_two_magnets_of_one_grade_count_exactly_as_with_a_shared_material():
    pytest.importorskip("gmsh")
    from magcore.fem3d import GeoObject3D, auto_domain3d, build_object_problem3d, solve_nonlinear3d

    def build(mat1, mat2):
        a = GeoObject3D("магнит 1", "box", {"lx": 8e-3, "ly": 8e-3, "lz": 3e-3}, mat1, center=(-6e-3, 0.0, 0.0))
        b = GeoObject3D("магнит 2", "box", {"lx": 8e-3, "ly": 8e-3, "lz": 3e-3}, mat2, center=(6e-3, 0.0, 0.0))
        return build_object_problem3d([a, b], auto_domain3d([a, b], material=Air(), margin_frac=1.0),
                                      default_mesh_size=2.5e-3, T=T_HOT)

    shared = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    f1 = solve_nonlinear3d(build(shared, shared))
    f2 = solve_nonlinear3d(build(MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0))), MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))))
    assert f2.converged
    assert np.array_equal(f1.B_cells, f2.B_cells) and np.array_equal(f1.retention, f2.retention)
