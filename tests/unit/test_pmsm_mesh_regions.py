import numpy as np
import pytest

from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)

# Посегментная сетка: пользователь задаёт характерный размер элемента ПО РЕГИОНУ. Оракул —
# не «красивая картинка», а проверяемые следствия: (1) локальное сгущение в зазоре/магнитах
# при грубых ярмах даёт число ячеек СТРОГО между равномерно-грубой и равномерно-тонкой;
# (2) средняя площадь ячейки в сгущённом регионе ~ h² и близка к равномерно-тонкой там же,
# и много меньше грубого ярма; (3) валидация отвергает мусор.

pytest.importorskip("gmsh")

H_COARSE = 0.006
H_FINE = 0.0018


def _mean_cell_area(g, region: Region) -> float:
    cells = np.where(g.region == int(region))[0]
    return float(np.mean([g.mesh.cell_area(int(c)) for c in cells]))


def test_per_region_refines_locally_not_globally():
    base = dict(n_slots=12, n_poles=14)
    g_coarse = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=H_COARSE, **base))
    g_fine = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=H_FINE, **base))
    g_reg = build_outrunner_spm_pmsm(OutrunnerPMSMParams(
        mesh_size=H_COARSE, mesh_size_by_region={"air_gap": H_FINE, "magnet": H_FINE}, **base))

    # (1) Сгущение только локально ⇒ ячеек больше, чем у грубой, но меньше, чем у сплошь тонкой.
    assert g_coarse.mesh.n_cells < g_reg.mesh.n_cells < g_fine.mesh.n_cells

    # (2a) В зазоре размер соответствует ЗАДАННОМУ тонкому: средняя площадь ~ как у равномерно
    # тонкой в том же регионе (в пределах ×2 — переходные зоны у границ).
    a_gap_reg = _mean_cell_area(g_reg, Region.AIR_GAP)
    a_gap_fine = _mean_cell_area(g_fine, Region.AIR_GAP)
    assert 0.5 < a_gap_reg / a_gap_fine < 2.0

    # (2b) Ярмо статора осталось грубым ⇒ его ячейки заметно крупнее зазорных (площадь ~ h²,
    # ожидаемо (H_COARSE/H_FINE)² ≈ 11×; берём безопасный порог ×3).
    a_yoke_reg = _mean_cell_area(g_reg, Region.STATOR_YOKE)
    assert a_gap_reg < a_yoke_reg / 3.0


def test_uniform_default_unchanged():
    # Без mesh_size_by_region путь ровно прежний: сетка строится, регионы полны.
    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=H_COARSE))
    assert g.mesh.n_cells > 0
    assert set(np.unique(g.region).tolist()) == {int(r) for r in Region}


def test_mesh_region_validation():
    with pytest.raises(ValueError, match="неизвестный регион"):
        OutrunnerPMSMParams(mesh_size_by_region={"nonsense": 0.001}).validate()
    with pytest.raises(ValueError, match="должен быть > 0"):
        OutrunnerPMSMParams(mesh_size_by_region={"magnet": -0.001}).validate()
