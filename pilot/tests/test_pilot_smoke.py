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


def test_example_config_parses():
    from pathlib import Path

    from pilot.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "example.toml")
    assert cfg.material_name == "NdFeB"
    assert cfg.magnet_radius < cfg.box_half
    assert cfg.n >= 1


def test_build_custom_and_named_materials():
    from pilot.config import build_magnet

    nd, nd_name = build_magnet({"material": "ndfeb"})
    sm, sm_name = build_magnet({"material": "smco"})
    assert nd_name == "NdFeB" and sm_name == "SmCo"
    assert sm.Br(20.0) < nd.Br(20.0)              # SmCo — ниже остаточная индукция

    custom, name = build_magnet({
        "material": "custom", "name": "мой", "Br": 1.2, "Hcb": 9.0e5,
        "Hk": 1.0e6, "Hcj": 1.3e6, "alpha_Br": 0.1, "gamma_Hc": 0.5,
    })
    assert name == "мой"
    assert custom.Br(20.0) == 1.2
