from __future__ import annotations

import time

import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.bem.double_layer import assemble_double_layer_galerkin
from magcore.hybrid.interface import CouplingInterface
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _assemble(n: int, config: AdaptiveIntegrationConfig | None = None):
    ci = CouplingInterface.from_tetra_mesh(build_structured_unit_cube_tetra_mesh(n))
    K = assemble_double_layer_galerkin(
        ci.surface_mesh, ci.outward_normals, ci.phi_space, ci.flux_space, config
    )
    return ci, K


def test_double_layer_galerkin_shape_and_calderon_solid_angle() -> None:
    # Дискретный Кальдерон / тождество телесного угла на плоской Γ:
    #   Σ_m K[f,m] = ∫_{T_f} (K·1)(x) ds_x = ∫_{T_f} (-1/2) ds_x = -A_f/2.
    # С квадратурой Заутера–Шваба (касающиеся пары) + регулярным правилом высокого
    # порядка (дальние/near) тождество выполняется до ~1e-6 БЫСТРО (n=1: ~3.7e-6),
    # вместо ~1.3e-2 при наивном подразбиении.
    ci, K = _assemble(1)  # дефолтный конфиг: use_sauter_schwab=True, ss_order=6, regular_order=6

    assert K.shape == (ci.flux_space.ndofs, ci.phi_space.ndofs)
    assert np.isfinite(K).all()
    assert np.linalg.norm(K) > 0.0

    row_sums = K.sum(axis=1)
    # Все 12 треугольников единичного куба геометрически эквивалентны ⇒ одинаковые row-sums.
    assert float(np.std(row_sums)) < 1e-5
    # Двойной слой замкнутой поверхности отрицателен на Γ, сходится к -A_f/2 (телесный угол).
    assert np.all(row_sums < 0.0)
    assert np.allclose(row_sums, -0.5 * ci.face_areas, atol=1e-5)


def test_double_layer_galerkin_naive_path_still_available() -> None:
    # use_sauter_schwab=False восстанавливает прежний (корректный, но грубый) путь:
    # тождество Кальдерона выполняется лишь приближённо.
    cfg = AdaptiveIntegrationConfig(use_sauter_schwab=False, regular_order=0, max_depth=3)
    ci, K = _assemble(1, cfg)
    row_sums = K.sum(axis=1)
    assert np.all(row_sums < 0.0)
    assert np.allclose(row_sums, -0.5 * ci.face_areas, atol=5e-2)


def test_double_layer_galerkin_fast_path_timing() -> None:
    # Регрессионный сторож скорости: SS (касающиеся) + регулярное правило высокого
    # порядка (дальние/near) считают 48 граней за ~1 с и до ~1e-6, тогда как наивное
    # подразбиение было ~51 с. Порог 10 с — с большим запасом против флаки, но ловит
    # откат к медленному пути.
    ci = CouplingInterface.from_tetra_mesh(build_structured_unit_cube_tetra_mesh(2))
    t0 = time.perf_counter()
    K = assemble_double_layer_galerkin(
        ci.surface_mesh, ci.outward_normals, ci.phi_space, ci.flux_space, None
    )
    dt = time.perf_counter() - t0
    err = float(np.max(np.abs(K.sum(axis=1) + 0.5 * ci.face_areas)))
    assert err < 1e-5
    assert dt < 10.0
