"""Smoke-тесты пилота: viz-функции создают файлы, ядро импортируется корректно."""
import numpy as np

from magcore.fem2d.mesh_generators import build_disk_tri_mesh
from pilot import viz


def test_viz_functions_write_png(tmp_path):
    disk = build_disk_tri_mesh(1.0, 6, 16)
    mesh = disk.mesh
    nc = mesh.n_cells

    p1 = viz.plot_cell_scalar(mesh, np.ones(nc), title="t", label="l",
                              save_path=tmp_path / "cell.png")
    p2 = viz.plot_node_scalar(mesh, np.zeros(mesh.n_vertices),
                              save_path=tmp_path / "node.png")
    p3 = viz.plot_B_field(mesh, np.tile([0.1, 0.0], (nc, 1)),
                          save_path=tmp_path / "b.png")
    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 0


def test_material_comparison_png(tmp_path):
    from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet

    out = viz.plot_material_comparison(
        [("NdFeB", n42sh_magnet([1, 0, 0])), ("SmCo", sm2co17_magnet([1, 0, 0]))],
        H_op=-6.0e5, save_path=tmp_path / "cmp.png",
    )
    assert out.exists() and out.stat().st_size > 0
