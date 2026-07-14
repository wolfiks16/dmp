import numpy as np
import pytest

from magcore.fem2d.machines.winding import (
    WindingLayout,
    star_of_slots_layout,
)


def test_star_of_slots_balanced_various():
    for n_slots, n_poles in [(12, 14), (12, 10), (9, 8), (12, 8), (6, 4), (24, 4), (18, 16)]:
        lay = star_of_slots_layout(n_slots, n_poles)
        counts = lay.phase_slot_counts()
        assert lay.is_balanced()                       # равное число пазов на фазу
        assert counts.sum() == n_slots
        assert np.all(counts == n_slots // 3)


def test_star_of_slots_12_14_signed_balanced():
    # Для 12/14 достижима знак-сбалансированная раскладка (нулевой суммарный знак фазы).
    lay = star_of_slots_layout(12, 14)
    for m in range(3):
        assert lay.sign_of_slot[lay.phase_of_slot == m].sum() == 0


def test_winding_layout_validation():
    with pytest.raises(ValueError):
        WindingLayout(4, np.array([0, 1, 2, 3]), np.array([1, 1, 1, 1]))  # фаза 3 недопустима
    with pytest.raises(ValueError):
        WindingLayout(3, np.array([0, 1, 2]), np.array([1, 0, -1]))       # знак 0 недопустим
    with pytest.raises(ValueError):
        WindingLayout(3, np.array([0, 1]), np.array([1, -1]))             # длина != n_slots


def test_star_of_slots_rejects_non_divisible():
    with pytest.raises(ValueError):
        star_of_slots_layout(10, 8)   # 10 не делится на 3


def test_cell_phase_sign_maps_slots_consistently():
    pytest.importorskip("gmsh")
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        cell_phase_sign,
    )

    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.0018))
    lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    phase, sign = cell_phase_sign(g, lay)

    in_slot = g.slot_id >= 0
    assert np.all(phase[~in_slot] == -1) and np.all(sign[~in_slot] == 0)   # вне пазов пусто
    assert np.all(phase[in_slot] >= 0)
    # все пазы размечены и все ячейки одного паза имеют одну фазу/знак.
    assert set(np.unique(g.slot_id[in_slot])) == set(range(g.params.n_slots))
    for s in range(g.params.n_slots):
        cells = g.slot_id == s
        assert len(set(phase[cells])) == 1
        assert phase[cells][0] == lay.phase_of_slot[s]
        assert sign[cells][0] == lay.sign_of_slot[s]
