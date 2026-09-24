import threading

import numpy as np
import pytest

from magcore.cancel import Cancelled, cancel_scope, check
from magcore.domain.magnet_model import n42sh_magnet
from magcore.fem2d.coupled_transient import solve_coupled_magneto_thermal_transient
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.model import Air, Problem2D, Region2D, solve_problem2d
from magcore.fem2d.spaces import LagrangeP1Space2D

# ОТМЕНА РАСЧЁТА (magcore.cancel): поток нельзя убить на полуслове, поэтому решатели сами проверяют флаг
# между итерациями и выходят исключением Cancelled. Проверяется: (1) вне области отмены проверка ничего не
# делает — и числа решателя от неё не зависят; (2) флаг действует только в своём потоке; (3) каждый долгий
# решатель действительно доходит до точки отмены: 2D Ньютон и Пикар, 3D Ньютон, ядро К6′; (4) отмена
# посреди расчёта останавливает его после текущей итерации, а не по окончании.


def _set():
    ev = threading.Event()
    ev.set()
    return ev


def _air_with_current():
    mesh = build_structured_rectangle_tri_mesh(8, 8)
    nc = mesh.n_cells
    cen = np.array([mesh.cell_centroid(c) for c in range(nc)])
    j = np.zeros(nc)
    j[(np.abs(cen[:, 0] - 0.5) < 0.18) & (np.abs(cen[:, 1] - 0.5) < 0.18)] = 1.0e5
    return Problem2D(mesh=mesh, cell_region=np.zeros(nc, int), regions={0: Region2D(0, "air", Air())}, j_cells=j)


def test_check_outside_a_scope_does_nothing():
    check()


def test_flag_works_only_while_set_and_only_in_its_own_thread():
    ev = threading.Event()
    with cancel_scope(ev):
        check()                                           # флаг не поднят
        ev.set()
        with pytest.raises(Cancelled):
            check()
        other = []
        t = threading.Thread(target=lambda: (check(), other.append("работает")))
        t.start()
        t.join()
        assert other == ["работает"]                      # у другого потока своей области нет
    check()                                               # вне области — снова ничего


def test_nested_scope_restores_the_outer_one():
    outer, inner = _set(), threading.Event()
    with cancel_scope(outer):
        with cancel_scope(inner):
            check()                                       # действует внутренняя, не поднята
        with pytest.raises(Cancelled):
            check()                                       # снова внешняя


def test_unset_flag_does_not_change_the_numbers():
    p = _air_with_current()
    free = solve_problem2d(p, max_iter=40)
    with cancel_scope(threading.Event()):
        scoped = solve_problem2d(p, max_iter=40)
    assert np.array_equal(free.B_cells, scoped.B_cells)


@pytest.mark.parametrize("method", ["newton", "picard"])
def test_2d_solvers_stop_at_the_cancel_point(method):
    with cancel_scope(_set()), pytest.raises(Cancelled):
        solve_problem2d(_air_with_current(), method=method, max_iter=40)


def test_cancel_in_the_middle_stops_after_the_current_iteration():
    """«Отменить» нажали во время первой итерации Ньютона: она доделывается, следующая не начинается."""
    pytest.importorskip("gmsh")
    import test_problem2d_magnet_knee as knee                 # магнит под стальной пластиной за коленом
    from magcore.fem2d.magnet_law import MagnetLaw2D
    from magcore.fem2d.model.problem import _reluctivity_newton
    from magcore.fem2d.newton import solve_nonlinear_2d_newton

    prob = knee._plate_problem(160.0)
    space = LagrangeP1Space2D(prob.mesh)
    nu_and_dnu, nu_init = _reluctivity_newton(prob)

    def run(on_call):
        calls = []

        def counted(B):                                   # ν считается при каждой оценке невязки
            calls.append(1)
            on_call(len(calls))
            return nu_and_dnu(B)

        law = MagnetLaw2D(prob.magnet(), prob.magnet_mask(), prob.mesh.n_cells, T=prob.T, axis=prob.magnet_axis)
        return calls, (lambda: solve_nonlinear_2d_newton(space, counted, nu_init=nu_init, magnet_law=law, tol=1e-12))

    full_calls, solve_full = run(lambda n: None)
    assert solve_full().n_iterations >= 3

    ev = threading.Event()
    calls, solve_cut = run(lambda n: ev.set())             # флаг поднят на первой же оценке
    with cancel_scope(ev), pytest.raises(Cancelled):
        solve_cut()
    # одна итерация: невязка + не больше 12 проб линейного поиска — и ни одной оценки следующей
    assert 2 <= len(calls) <= 13 and len(calls) < len(full_calls)


def test_3d_newton_stops_at_the_cancel_point():
    pytest.importorskip("gmsh")
    from magcore.fem2d.model.materials import MagnetMaterial
    from magcore.fem3d import GeoObject3D, auto_domain3d, build_object_problem3d, solve_nonlinear3d

    mag = GeoObject3D(name="магнит", kind="box", params={"lx": 0.01, "ly": 0.01, "lz": 0.004},
                      material=MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0))))
    prob = build_object_problem3d([mag], auto_domain3d([mag], material=Air(), margin_frac=1.0),
                                  default_mesh_size=0.003)
    with cancel_scope(_set()), pytest.raises(Cancelled):
        solve_nonlinear3d(prob)


def test_coupled_transient_stops_at_the_cancel_point():
    mesh = build_structured_rectangle_tri_mesh(12, 8, x0=0.0, x1=0.06, y0=0.0, y1=0.04)
    space = LagrangeP1Space2D(mesh)
    cent = np.array([mesh.cell_vertices(c).mean(axis=0) for c in range(mesh.n_cells)])
    magnet_mask = (cent[:, 0] >= 0.02) & (cent[:, 0] <= 0.03) & (cent[:, 1] >= 0.012) & (cent[:, 1] <= 0.028)
    copper_mask = (cent[:, 0] >= 0.03) & (cent[:, 0] <= 0.04) & (cent[:, 1] >= 0.012) & (cent[:, 1] <= 0.028)
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nu = np.where(magnet_mask, 1.0 / magnet.mu_rec, 1.0)
    with cancel_scope(_set()), pytest.raises(Cancelled):
        solve_coupled_magneto_thermal_transient(
            space, k_cells=np.ones(mesh.n_cells), capacity_cells=np.full(mesh.n_cells, 1.0e6), h=25.0, T_amb=20.0,
            dt=10.0, n_steps=5, j_cells=np.where(copper_mask, 1.0e7, 0.0),
            magnet=magnet, magnet_mask=magnet_mask, nu_init=nu)
