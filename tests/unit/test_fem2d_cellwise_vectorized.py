import numpy as np
import pytest

from magcore.fem2d.assembly import (
    assemble_current_rhs_piecewise,
    assemble_magnetization_rhs,
    p1_cell_geometry,
)
from magcore.fem2d.mesh import TriangleMesh, p1_gradients, triangle_area
from magcore.fem2d.mesh_generators import build_disk_tri_mesh, build_structured_rectangle_tri_mesh
from magcore.fem2d.newton import _b_on_cells, _rot_basis
from magcore.fem2d.post import reconstruct_B_on_cells
from magcore.fem2d.spaces import LagrangeP1Space2D

# Источник магнита, ток обмотки и B по ячейкам — ВЕКТОРИЗОВАНЫ: на сетке спицевого двигателя (64 990
# ячеек) прежние Python-циклы с p1_gradients/triangle_area на каждую ячейку занимали почти всю итерацию
# Пикара, а не разрежённое решение. Оракул — прежние поячеечные формулы (перенесены сюда как эталон)
# и тождество сопряжённости источника и B. Допуски выведены из округления, а не подобраны
# (u = eps/2, γ_n = n·u/(1 − n·u) — граница Хайэма для n округлений на пути слагаемого):
#   * геометрия: ∇φ и площадь в обоих путях — одна формула, те же операции IEEE ⇒ совпадают побитово;
#   * вектор нагрузки: в узел i сходится m_i вкладов (m_i − 1 сложений), вклад магнита — 3 операции,
#     тока — 2; две реализации одной суммы расходятся не больше чем на 2·γ_n·W_i, где W_i — сумма
#     модулей слагаемых; W сама считается с округлением ⇒ n берётся на 1 больше: m_i+3 и m_i+2;
#   * B: компонента — сумма трёх произведений A_a·∂φ_a ⇒ |ΔB| ≤ 2·γ_3·Σ_a|A_a·∂φ_a|, с той же
#     поправкой на округление самой суммы — 2·γ_4.

U = np.finfo(float).eps / 2.0


def _gamma(n):
    n = np.asarray(n, dtype=float)
    return n * U / (1.0 - n * U)


def _old_magnetization_rhs(space, nu_br):
    """Прежний поячеечный `assemble_magnetization_rhs` — эталон."""
    mesh = space.mesh
    f = np.zeros(space.ndofs)
    for c in range(mesh.n_cells):
        verts = mesh.cell_vertices(c)
        area = triangle_area(verts)
        grads = p1_gradients(verts)
        idx = mesh.cell_vertex_indices(c)
        bx, by = nu_br[c, 0], nu_br[c, 1]
        f[list(idx)] += area * (bx * grads[:, 1] - by * grads[:, 0])
    return f


def _old_current_rhs(space, jz):
    """Прежний поячеечный `assemble_current_rhs_piecewise` — эталон."""
    mesh = space.mesh
    f = np.zeros(space.ndofs)
    for c in range(mesh.n_cells):
        if jz[c] == 0.0:
            continue
        area = triangle_area(mesh.cell_vertices(c))
        f[list(mesh.cell_vertex_indices(c))] += jz[c] * area / 3.0
    return f


def _old_B(space, a):
    """Прежний поячеечный `reconstruct_B_on_cells` — эталон."""
    mesh = space.mesh
    B = np.empty((mesh.n_cells, 2))
    for c in range(mesh.n_cells):
        grads = p1_gradients(mesh.cell_vertices(c))
        idx = mesh.cell_vertex_indices(c)
        grad_az = a[list(idx)] @ grads
        B[c, 0] = grad_az[1]
        B[c, 1] = -grad_az[0]
    return B


def _meshes():
    # Прямоугольник — валентность узлов 2…6; диск — неравные треугольники и центр валентности 48.
    yield build_structured_rectangle_tri_mesh(9, 7, x0=-0.03, x1=0.04, y0=0.0, y1=0.025)
    yield build_disk_tri_mesh(0.02, 10, 48).mesh


def _inputs(mesh, seed):
    """Поля как в машине: источник только в «магните», ток только в «пазах», A_z ~ мТл·м."""
    rng = np.random.default_rng(seed)
    nc = mesh.n_cells
    nu_br = rng.uniform(-1.2, 1.2, (nc, 2)) * (rng.random(nc) < 0.4)[:, None]
    jz = rng.uniform(-1.6e7, 1.6e7, nc) * (rng.random(nc) < 0.3)
    a = rng.normal(0.0, 1.0e-3, mesh.n_vertices)
    return nu_br, jz, a


@pytest.mark.parametrize("mesh", list(_meshes()), ids=["rectangle", "disk"])
def test_geometry_is_bitwise_the_cellwise_formula(mesh):
    # Предпосылка допусков ниже: геометрия векторного пути — та же формула, побитово.
    cells, grad, area = p1_cell_geometry(mesh)
    for c in range(mesh.n_cells):
        v = mesh.cell_vertices(c)
        assert np.array_equal(grad[c], p1_gradients(v))
        assert area[c] == triangle_area(v)


@pytest.mark.parametrize("mesh", list(_meshes()), ids=["rectangle", "disk"])
def test_load_vectors_equal_cellwise_formulas(mesh):
    space = LagrangeP1Space2D(mesh)
    cells, grad, area = p1_cell_geometry(mesh)
    m = np.bincount(cells.ravel(), minlength=space.ndofs)          # m_i — вкладов в узел i
    for seed in (1, 2, 3):
        nu_br, jz, _ = _inputs(mesh, seed)

        f = assemble_magnetization_rhs(space, nu_br)
        w = area[:, None] * (np.abs(nu_br[:, 0:1] * grad[:, :, 1]) + np.abs(nu_br[:, 1:2] * grad[:, :, 0]))
        W = np.bincount(cells.ravel(), weights=w.ravel(), minlength=space.ndofs)
        assert f.shape == (space.ndofs,)
        assert np.all(np.abs(f - _old_magnetization_rhs(space, nu_br)) <= 2.0 * _gamma(m + 3) * W)

        g = assemble_current_rhs_piecewise(space, jz)
        Wj = np.bincount(cells.ravel(), weights=np.repeat(np.abs(jz) * area / 3.0, 3), minlength=space.ndofs)
        assert g.shape == (space.ndofs,)
        assert np.all(np.abs(g - _old_current_rhs(space, jz)) <= 2.0 * _gamma(m + 2) * Wj)
        assert np.any(W > 0.0) and np.any(Wj > 0.0)                 # источники не пусты


@pytest.mark.parametrize("mesh", list(_meshes()), ids=["rectangle", "disk"])
def test_B_equals_cellwise_formula_and_newton_path(mesh):
    space = LagrangeP1Space2D(mesh)
    cells, grad, _ = p1_cell_geometry(mesh)
    for seed in (4, 5):
        _, _, a = _inputs(mesh, seed)
        S = np.einsum("ca,cad->cd", np.abs(a[cells]), np.abs(grad))[:, ::-1]   # (Σ|A∂xφ|, Σ|A∂yφ|) → порядок B
        bound = 2.0 * _gamma(4) * S
        B = reconstruct_B_on_cells(space, a)
        assert B.shape == (mesh.n_cells, 2)
        assert np.all(np.abs(B - _old_B(space, a)) <= bound)
        # Пикар (post) и Ньютон (newton._b_on_cells) считают B одной формулой.
        c, rot, _ = _rot_basis(space)
        assert np.all(np.abs(B - _b_on_cells(c, rot, a)) <= bound)


@pytest.mark.parametrize("mesh", list(_meshes()), ids=["rectangle", "disk"])
def test_magnet_source_is_the_transpose_of_B_reconstruction(mesh):
    # Независимое тождество: f_i = Σ_c A_c·νB_r,c·rot φ_i ⇒ aᵀ·f(νB_r) = Σ_c A_c·νB_r,c·B_c(a) для ЛЮБЫХ
    # a и νB_r. Ловит знак и перестановку компонент в любой из двух функций. Обе стороны — суммы всех
    # элементарных произведений A_c·|a_i|·|νB_r|·|∂φ_i|; их модульная сумма T, а число округлений на
    # пути слагаемого ≤ max(m)+3+n_узлов слева и n_ячеек+6 справа ⇒ |Δ| ≤ 2·γ_N·T (N — с запасом +1).
    space = LagrangeP1Space2D(mesh)
    cells, grad, area = p1_cell_geometry(mesh)
    m = np.bincount(cells.ravel(), minlength=space.ndofs)
    for seed in (6, 7):
        nu_br, _, a = _inputs(mesh, seed)
        lhs = float(a @ assemble_magnetization_rhs(space, nu_br))
        rhs = float(np.sum(area * np.einsum("cd,cd->c", nu_br, reconstruct_B_on_cells(space, a))))
        T = float(np.sum(area[:, None] * np.abs(a[cells])
                         * (np.abs(nu_br[:, 0:1] * grad[:, :, 1]) + np.abs(nu_br[:, 1:2] * grad[:, :, 0]))))
        N = max(int(m.max()) + 3 + space.ndofs, mesh.n_cells + 6) + 1
        assert abs(lhs - rhs) <= 2.0 * float(_gamma(N)) * T
        assert T > 0.0


def test_unreferenced_node_gets_zero_and_vector_keeps_full_length():
    # Узел, не входящий ни в одну ячейку: вектор всё равно длины n_узлов, в узле — ноль (как у цикла).
    base = build_structured_rectangle_tri_mesh(3, 2)
    mesh = TriangleMesh(np.vstack([base.vertices, [[5.0, 5.0]]]), base.cells)
    space = LagrangeP1Space2D(mesh)
    nu_br = np.tile([0.4, -0.9], (mesh.n_cells, 1))
    f = assemble_magnetization_rhs(space, nu_br)
    g = assemble_current_rhs_piecewise(space, np.full(mesh.n_cells, 2.0))
    assert f.shape == g.shape == (mesh.n_vertices,)
    assert f[-1] == 0.0 and g[-1] == 0.0
    # Остальные узлы — те же, что без лишнего узла (та же сборка тех же ячеек).
    assert np.array_equal(f[:-1], assemble_magnetization_rhs(LagrangeP1Space2D(base), nu_br))


@pytest.mark.parametrize("shape_of", [lambda n: (n, 3), lambda n: (n + 1, 2), lambda n: (n,)])
def test_magnetization_rhs_rejects_wrong_shape(shape_of):
    mesh = build_structured_rectangle_tri_mesh(3, 3)
    space = LagrangeP1Space2D(mesh)
    with pytest.raises(ValueError, match="nu_br_cells"):
        assemble_magnetization_rhs(space, np.zeros(shape_of(mesh.n_cells)))


def test_B_from_too_short_potential_fails_loudly():
    # Негатив: A_z короче числа узлов — ошибка, а не тихо обрезанное поле.
    mesh = build_structured_rectangle_tri_mesh(3, 3)
    with pytest.raises(IndexError):
        reconstruct_B_on_cells(LagrangeP1Space2D(mesh), np.zeros(mesh.n_vertices - 1))
