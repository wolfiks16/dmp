import numpy as np
import pytest

from magcore.fem3d.mesh import TetMesh3D, _has_duplicate_rows, orient_cells, signed_volumes6, unique_rows
from magcore.mesh.mesh_generators import build_structured_box_tetra_mesh

# Векторный контейнер 3D-сетки (этап 3D-1): те же инварианты, что у TetraMesh старого ядра,
# но массивно. Оракулы: сумма объёмов = объём параллелепипеда; число граничных граней
# структурной сетки известно заранее; качество правильного тетраэдра = 1; негативные случаи
# ловятся; перевод в TetraMesh проходит его собственную поячеечную проверку.


def _box(n: int = 3):
    m = build_structured_box_tetra_mesh(n, n, n, xlim=(0.0, 2.0), ylim=(0.0, 1.0), zlim=(0.0, 3.0))
    return m.vertices, m.cells


def test_volumes_and_boundary_faces_of_structured_box():
    v, c = _box(3)
    m = TetMesh3D(v, c)
    assert abs(m.cell_volumes().sum() - 6.0) < 1e-12
    assert np.all(m.cell_volumes() > 0.0)
    # 6 граней куба × 9 квадратов × 2 треугольника
    assert m.boundary_faces().shape == (6 * 9 * 2, 3)
    assert np.allclose(m.cell_centroids().min(axis=0) > 0.0, True)


def test_quality_of_regular_tetrahedron_is_one():
    v = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    m = TetMesh3D(v, orient_cells(v, [[0, 1, 2, 3]]))
    assert abs(m.quality()[0] - 1.0) < 1e-12
    flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.3, 1e-6]])
    assert TetMesh3D(flat, orient_cells(flat, [[0, 1, 2, 3]])).quality()[0] < 1e-5


def test_orient_cells_fixes_negative_orientation():
    v, c = _box(2)
    bad = c.copy()
    bad[::2, [1, 2]] = bad[::2, [2, 1]]
    assert np.any(signed_volumes6(v, bad) < 0.0)
    with pytest.raises(ValueError):
        TetMesh3D(v, bad)
    fixed = orient_cells(v, bad)
    assert np.all(signed_volumes6(v, fixed) > 0.0)
    assert TetMesh3D(v, fixed).n_cells == c.shape[0]


def test_invalid_meshes_rejected():
    v, c = _box(2)
    with pytest.raises(ValueError):
        TetMesh3D(v, np.vstack([c, c[:1]]))                    # повтор тетраэдра
    with pytest.raises(ValueError):
        TetMesh3D(v, c + v.shape[0])                           # индексы вне диапазона
    rep = c.copy()
    rep[0, 1] = rep[0, 0]
    with pytest.raises(ValueError):
        TetMesh3D(v, rep)                                      # повтор вершины в ячейке
    vn = v.copy()
    vn[0, 0] = np.nan
    with pytest.raises(ValueError):
        TetMesh3D(vn, c)                                       # нечисловые координаты
    with pytest.raises(ValueError):
        TetMesh3D(v[:, :2], c)                                 # не та форма
    # грань (0,1,2) у трёх ячеек — неманифолдно
    w = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.2, 0.2, 0.5]])
    three = orient_cells(w, [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]])
    with pytest.raises(ValueError):
        TetMesh3D(w, three)


def test_unique_rows_is_exactly_numpy_unique_by_rows():
    # Поиск одинаковых граней через числовой ключ строки (ускорение открытия расчёта, этап 3D-9): ключ строго
    # растёт в лексикографическом порядке строк, поэтому результат обязан совпасть с np.unique(axis=0) ДО БИТА —
    # строки и их порядок, обратные номера, счётчики. Строки с повторами, 2–4 столбца, отсортированные тройки
    # (грани); ключ, не помещающийся в int64, идёт прежним путём по строкам.
    rng = np.random.default_rng(3)
    cases = [rng.integers(0, 50, size=(2000, 3)), rng.integers(0, 7, size=(500, 2)),
             rng.integers(0, 3000, size=(4000, 4)), np.sort(rng.integers(0, 40, size=(3000, 3)), axis=1),
             np.array([[2 ** 40, 1], [5, 2 ** 40], [2 ** 40, 1]])]
    for r in cases:
        for inv, cnt in ((False, False), (True, False), (False, True), (True, True)):
            got = unique_rows(r, return_inverse=inv, return_counts=cnt)
            ref = np.unique(r, axis=0, return_inverse=inv, return_counts=cnt)
            got, ref = (got, ref) if (inv or cnt) else ((got,), (ref,))
            assert len(got) == len(ref)
            for a, b in zip(got, ref):
                assert a.dtype == b.dtype and a.size == b.size
                assert np.array_equal(a.reshape(b.shape) if a.ndim != b.ndim else a, b)
    rows = rng.integers(0, 100, size=(1000, 4))
    rows = rows[np.unique(rows, axis=0, return_index=True)[1]]            # без повторов
    big = np.array([[2 ** 32, 1, 2, 3], [4, 5, 6, 7]])                    # ключ-пара не помещается — прежний путь
    assert not _has_duplicate_rows(rows) and _has_duplicate_rows(np.vstack([rows, rows[17:18]]))
    assert not _has_duplicate_rows(big) and _has_duplicate_rows(np.vstack([big, big[:1]]))


def test_conversion_to_old_core_mesh():
    v, c = _box(2)
    m = TetMesh3D(v, c)
    old = m.to_tetra_mesh()                   # поячеечная проверка старого ядра проходит
    assert old.n_cells == m.n_cells
    assert abs(sum(old.cell_volume(i) for i in range(old.n_cells)) - 6.0) < 1e-12
