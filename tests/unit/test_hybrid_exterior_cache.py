"""
Кэш линейного BEM-экстерьера (`prepare_coupled_exterior`) — это ТОЧНАЯ оптимизация
нелинейного Picard, а не приближение. Здесь доказываем: связанная система, собранная
С кэшем, бит-в-бит (до машинной точности) совпадает с собранной БЕЗ кэша, на всех
RHS-путях (ν поячеечно + намагниченность + приложенное поле). Иначе ускорение было бы
«костылём», меняющим физику.
"""
from __future__ import annotations

import numpy as np
import pytest

from magcore.femcore.assembly import assemble_magnetization_rhs
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import (
    assemble_coupled_block_system,
    prepare_coupled_exterior,
)
from magcore.hybrid.interface import CouplingInterface
from magcore.mesh.mesh_generators import build_ball_tetra_mesh


def _zero_j(_x):
    return np.zeros(3)


def _setup(n: int = 3):
    mesh = build_ball_tetra_mesh(n, radius=1.0)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    return mesh, ci, vs, ss


def test_cached_exterior_matches_full_assembly_bitwise() -> None:
    mesh, ci, vs, ss = _setup()
    nc = mesh.n_cells
    # все нелинейные RHS-пути сразу: поячеечная ν + магнит + приложенное поле.
    nu = np.linspace(0.8, 1.6, nc)
    f_br = assemble_magnetization_rhs(mesh, vs, np.tile([0.0, 0.0, 1.0], (nc, 1)))
    h0 = np.array([0.1, -0.2, 0.3])

    full = assemble_coupled_block_system(
        ci, vs, ss, nu=nu, j_fn=_zero_j, extra_vector_rhs=f_br,
        applied_field_h0=h0, mu0=1.0,
    )
    cache = prepare_coupled_exterior(ci, vs, mu0=1.0)
    cached = assemble_coupled_block_system(
        ci, vs, ss, nu=nu, j_fn=_zero_j, extra_vector_rhs=f_br,
        applied_field_h0=h0, mu0=1.0, exterior=cache,
    )

    # до машинной точности (одни и те же операции ⇒ практически бит-в-бит).
    assert np.allclose(full.matrix, cached.matrix, rtol=0.0, atol=1e-12)
    assert np.allclose(full.rhs, cached.rhs, rtol=0.0, atol=1e-12)
    assert (full.n_a, full.n_p, full.n_psi, full.n_lam) == (
        cached.n_a, cached.n_p, cached.n_psi, cached.n_lam,
    )


def test_cached_exterior_matches_full_without_magnet_or_field() -> None:
    # вырожденные пути (нет магнита, нет поля) тоже должны совпадать.
    mesh, ci, vs, ss = _setup()
    cache = prepare_coupled_exterior(ci, vs, mu0=1.0)
    full = assemble_coupled_block_system(ci, vs, ss, nu=1.2, j_fn=_zero_j, mu0=1.0)
    cached = assemble_coupled_block_system(
        ci, vs, ss, nu=1.2, j_fn=_zero_j, mu0=1.0, exterior=cache
    )
    assert np.allclose(full.matrix, cached.matrix, rtol=0.0, atol=1e-12)
    assert np.allclose(full.rhs, cached.rhs, rtol=0.0, atol=1e-12)


def test_cache_rejects_mu0_mismatch() -> None:
    # кэш собран при mu0=1; использование при другом mu0 ДОЛЖНО отвергаться
    # (W/S_ext внутри уже масштабированы на mu0 кэша).
    _, ci, vs, ss = _setup()
    cache = prepare_coupled_exterior(ci, vs, mu0=1.0)
    with pytest.raises(ValueError):
        assemble_coupled_block_system(
            ci, vs, ss, nu=1.0, j_fn=_zero_j, mu0=2.0, exterior=cache
        )
