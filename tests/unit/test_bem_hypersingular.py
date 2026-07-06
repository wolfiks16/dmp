from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.bem.hypersingular import assemble_hypersingular_maue, surface_curl_hats
from magcore.hybrid.interface import CouplingInterface
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def test_surface_curl_hats_signs_and_partition() -> None:
    # Эталонный треугольник, нормаль правила правой руки n=(0,0,1), 2A=1.
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    n = np.array([0.0, 0.0, 1.0])
    c = surface_curl_hats(tri, n)

    # Закрытая форма c_i = (p_{i+1} − p_{i+2}) / (2A): пиннинг знака curl_Γ.
    expected = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.allclose(c, expected, atol=1e-12)

    # Разбиение единицы: Σ_i curl_Γ λ_i = curl_Γ(1) = 0 (точно).
    assert np.allclose(c.sum(axis=0), 0.0, atol=1e-13)

    # Каждый c_i касателен грани (⟂ нормали).
    assert np.allclose(c @ n, 0.0, atol=1e-13)


def _assemble_W(n: int = 1):
    ci = CouplingInterface.from_tetra_mesh(build_structured_unit_cube_tetra_mesh(n))
    cfg = AdaptiveIntegrationConfig(quadrature_order=2, max_depth=3, self_max_depth=4)
    w = assemble_hypersingular_maue(
        ci.surface_mesh, ci.outward_normals, ci.phi_space, ci.flux_space, cfg
    )
    return ci, w


def test_hypersingular_maue_shape_symmetry_consistency_psd() -> None:
    ci, w = _assemble_W(1)

    nv = ci.phi_space.ndofs
    assert w.shape == (nv, nv)
    assert np.isfinite(w).all()
    assert np.linalg.norm(w) > 0.0

    # 1) Симметрия (свойство Costabel: S симметрична ⇒ W=RᵀSR симметрична).
    assert np.allclose(w, w.T, atol=1e-12)

    # 2) Тождество совместности (дискретный Кальдерон для W): W·1 = 0 — ТОЧНО (машинно),
    #    т.к. Σ_{n∈T'} c_{T'}^n = n×∇(Σλ) = 0 не зависит от квадратуры однослойного.
    ones = np.ones(nv)
    assert np.allclose(w @ ones, 0.0, atol=1e-9)

    # 3) Положительная полуопределённость с ядром = константы (1-мерное на связной Γ):
    #    S SPD ⇒ W=RᵀSR ⪰ 0; ровно одно собственное значение ≈ 0 (вектор констант).
    eig = np.linalg.eigvalsh(w)
    assert eig[0] > -1e-9                      # PSD
    assert abs(eig[0]) < 1e-8 * max(eig[-1], 1.0)  # ядро: наименьшее с.з. ≈ 0
    assert eig[1] > 1e-6 * eig[-1]             # положителен на дополнении к константам
