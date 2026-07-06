"""
Негативные тесты слоя решателя (P3 закалки): код ДОЛЖЕН отвергать некорректные данные,
а не молча их принимать. Покрывает guard'ы femcore/hybrid-сборок и решателей, которые
до этого не были проверены ни одним тестом (177 `raise` в коде — единицы под тестами).
"""
from __future__ import annotations

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.femcore.assembly import (
    assemble_magnetization_rhs,
    assemble_mixed_coulomb_system,
)
from magcore.femcore.nonlinear import solve_nonlinear_mixed_picard
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import assemble_coupled_block_system
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.magnet_demag import MagnetDemagPolicy
from magcore.hybrid.nonlinear import solve_coupled_nonlinear_picard
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _zero_j(_x):
    return np.zeros(3)


def _J_rot(x):
    return np.array([-(x[1] - 0.5), (x[0] - 0.5), 0.0])


def _fem(n: int = 2):
    mesh = build_structured_unit_cube_tetra_mesh(n)
    return mesh, NedelecP1Space.from_mesh(mesh), LagrangeP1Space(mesh)


def _coupled(n: int = 2):
    mesh = build_structured_unit_cube_tetra_mesh(n)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    return mesh, ci, NedelecP1Space.from_mesh(mesh), LagrangeP1Space(mesh)


# ---------------------------------------------------------------------------- #
# femcore: поячеечная ν и RHS намагниченности — отвергают неверные формы.
# ---------------------------------------------------------------------------- #
def test_assembly_rejects_wrong_shape_nu_array() -> None:
    mesh, vs, ss = _fem()
    with pytest.raises(ValueError):
        assemble_mixed_coulomb_system(mesh, vs, ss, nu=np.ones(mesh.n_cells + 1), J_fn=_zero_j)


def test_magnetization_rhs_rejects_wrong_shape() -> None:
    mesh, vs, _ = _fem()
    with pytest.raises(ValueError):
        assemble_magnetization_rhs(mesh, vs, np.zeros((mesh.n_cells, 2)))  # должно быть (n,3)


# ---------------------------------------------------------------------------- #
# femcore Picard: чужая сетка, не-callable, диапазон релаксации, формы.
# ---------------------------------------------------------------------------- #
def test_fem_picard_rejects_space_on_other_mesh() -> None:
    mesh_a, _, ss = _fem()
    _, vs_b, _ = _fem()  # пространство на ДРУГОЙ сетке (другой объект)
    with pytest.raises(ValueError):
        solve_nonlinear_mixed_picard(
            mesh_a, vs_b, ss, nu_of_B=lambda B: np.ones(mesh_a.n_cells),
            J_fn=_zero_j, nu_init=np.ones(mesh_a.n_cells),
        )


def test_fem_picard_rejects_non_callable_nu() -> None:
    mesh, vs, ss = _fem()
    with pytest.raises(ValueError):
        solve_nonlinear_mixed_picard(
            mesh, vs, ss, nu_of_B=2.0, J_fn=_zero_j, nu_init=np.ones(mesh.n_cells)
        )


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_fem_picard_rejects_bad_relaxation(bad: float) -> None:
    mesh, vs, ss = _fem()
    with pytest.raises(ValueError):
        solve_nonlinear_mixed_picard(
            mesh, vs, ss, nu_of_B=lambda B: np.ones(mesh.n_cells),
            J_fn=_zero_j, nu_init=np.ones(mesh.n_cells), relaxation=bad,
        )


def test_fem_picard_rejects_wrong_shape_nu_init() -> None:
    mesh, vs, ss = _fem()
    with pytest.raises(ValueError):
        solve_nonlinear_mixed_picard(
            mesh, vs, ss, nu_of_B=lambda B: np.ones(mesh.n_cells),
            J_fn=_zero_j, nu_init=np.ones(mesh.n_cells + 3),
        )


def test_fem_picard_rejects_nu_of_B_wrong_return_shape() -> None:
    # guard срабатывает В ЦИКЛе (после первой сборки/решения) — на n=2 это быстро.
    mesh, vs, ss = _fem()
    with pytest.raises(ValueError):
        solve_nonlinear_mixed_picard(
            mesh, vs, ss,
            nu_of_B=lambda B: np.ones(mesh.n_cells + 1),  # неверная длина
            J_fn=_J_rot, nu_init=np.ones(mesh.n_cells), max_iter=5,
        )


# ---------------------------------------------------------------------------- #
# hybrid связанная сборка: формы дополнительного RHS / приложенного поля / ν.
# ---------------------------------------------------------------------------- #
def test_coupled_assembly_rejects_wrong_shape_extra_rhs() -> None:
    _, ci, vs, ss = _coupled()
    with pytest.raises(ValueError):
        assemble_coupled_block_system(
            ci, vs, ss, nu=1.0, j_fn=_zero_j, extra_vector_rhs=np.zeros(vs.ndofs + 1)
        )


def test_coupled_assembly_rejects_wrong_shape_applied_field() -> None:
    _, ci, vs, ss = _coupled()
    with pytest.raises(ValueError):
        assemble_coupled_block_system(
            ci, vs, ss, nu=1.0, j_fn=_zero_j, applied_field_h0=np.zeros(2)
        )


def test_coupled_assembly_rejects_wrong_shape_nu_array() -> None:
    mesh, ci, vs, ss = _coupled()
    with pytest.raises(ValueError):
        assemble_coupled_block_system(
            ci, vs, ss, nu=np.ones(mesh.n_cells + 1), j_fn=_zero_j
        )


# ---------------------------------------------------------------------------- #
# hybrid связанный Picard: чужая сетка, диапазон релаксации, формы, магнит.
# ---------------------------------------------------------------------------- #
def test_coupled_picard_rejects_space_on_other_mesh() -> None:
    _, ci, _, ss = _coupled()
    _, vs_b, _ = _fem()  # vs на другой сетке
    with pytest.raises(ValueError):
        solve_coupled_nonlinear_picard(
            ci, vs_b, ss, nu_of_B=lambda B: np.ones(ci.tetra_mesh.n_cells),
            nu_init=np.ones(ci.tetra_mesh.n_cells),
        )


def test_coupled_picard_rejects_bad_relaxation() -> None:
    _, ci, vs, ss = _coupled()
    nc = ci.tetra_mesh.n_cells
    with pytest.raises(ValueError):
        solve_coupled_nonlinear_picard(
            ci, vs, ss, nu_of_B=lambda B: np.ones(nc), nu_init=np.ones(nc), relaxation=2.0
        )


def test_coupled_picard_rejects_wrong_shape_static_magnetization() -> None:
    _, ci, vs, ss = _coupled()
    nc = ci.tetra_mesh.n_cells
    with pytest.raises(ValueError):
        solve_coupled_nonlinear_picard(
            ci, vs, ss, nu_of_B=lambda B: np.ones(nc), nu_init=np.ones(nc),
            magnetization=np.zeros((nc, 2)),  # должно быть (nc, 3)
        )


# ---------------------------------------------------------------------------- #
# magnet_demag: политика отвергает неверную маску/релаксацию.
# ---------------------------------------------------------------------------- #
def test_magnet_policy_rejects_wrong_shape_mask() -> None:
    magnet = n42sh_magnet(easy_axis=[0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        MagnetDemagPolicy(magnet, np.ones(5, dtype=bool), T=20.0, n_cells=10)


def test_magnet_policy_rejects_bad_relaxation() -> None:
    magnet = n42sh_magnet(easy_axis=[0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        MagnetDemagPolicy(magnet, np.ones(10, dtype=bool), T=20.0, n_cells=10, relaxation=0.0)
