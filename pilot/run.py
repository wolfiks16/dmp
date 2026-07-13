"""
CLI пилота: расчёт по ПОЛЬЗОВАТЕЛЬСКОМУ конфигу (вы сами задаёте параметры).

    python -m pilot.run pilot/configs/example.toml

Считает связку магнит-тепло-демаг по параметрам из конфига, сохраняет графики и отчёт.
Ничего в ядре `magcore` не меняет — только вызывает верифицированные решатели.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.fem2d.kelvin import solve_kelvin_magnetostatic
from magcore.fem2d.magneto_thermal import solve_magneto_thermal_demag
from magcore.fem2d.mesh_generators import (
    build_disk_tri_mesh,
    build_structured_rectangle_tri_mesh,
)
from magcore.fem2d.spaces import LagrangeP1Space2D
from pilot import viz
from pilot.config import RunConfig, load_config

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "example.toml"


def _oracle_cylinder() -> float:
    disk = build_disk_tri_mesh(3.0, 24, 96)
    mesh = disk.mesh
    nu_br = np.zeros((mesh.n_cells, 2))
    mag = [c for c in range(mesh.n_cells) if np.linalg.norm(mesh.cell_centroid(c)) < 1.0]
    nu_br[mag] = [1.0, 0.0]
    res = solve_kelvin_magnetostatic(disk, nu_real=np.ones(mesh.n_cells), magnetization=nu_br)
    return float((res.B_cells[mag] - np.array([1.0, 0.0]))[:, 0].mean())


def run(cfg: RunConfig) -> dict:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    mesh = build_structured_rectangle_tri_mesh(
        cfg.n, cfg.n, x0=-cfg.box_half, x1=cfg.box_half, y0=-cfg.box_half, y1=cfg.box_half
    )
    space = LagrangeP1Space2D(mesh)
    nc = mesh.n_cells
    mask = np.array(
        [np.linalg.norm(mesh.cell_centroid(c)) < cfg.magnet_radius for c in range(nc)],
        dtype=bool,
    )
    if not mask.any():
        raise ValueError("Магнит пуст: magnet_radius слишком мал для сетки — увеличьте n или радиус.")
    q = np.where(mask, cfg.heat_load, 0.0)

    res = solve_magneto_thermal_demag(
        space, cfg.magnet, mask, heat_source_cells=q, k_cells=np.full(nc, 1.0),
        h=cfg.cooling_h, T_amb=cfg.T_ambient, applied_B0=cfg.applied_B0, em_max_iter=200,
    )

    tag = cfg.material_name
    viz.plot_B_field(mesh, res.B_cells, title="Поле B (%s)" % tag,
                     save_path=out / "B_field.png")
    viz.plot_node_scalar(mesh, res.T_field, title="Температура, °C", label="T, °C",
                         save_path=out / "temperature.png")
    viz.plot_demag_risk(mesh, mask, res.risk,
                        title="%s: T=%.0f°C, за коленом %d/%d" % (
                            tag, res.T_magnet, res.risk.n_demagnetized, res.risk.cell_indices.size),
                        save_path=out / "demag_risk.png")
    viz.plot_material_comparison(
        [("NdFeB", n42sh_magnet([1, 0, 0])), ("SmCo", sm2co17_magnet([1, 0, 0]))],
        H_op=-6.0e5, save_path=out / "material_comparison.png",
    )

    h_in = _oracle_cylinder()
    lines = [
        "=" * 68,
        "ПИЛОТ: расчёт по конфигу — материал=%s, тепл.нагрузка=%.0f, поле B0=%s Тл"
        % (tag, cfg.heat_load, cfg.applied_B0),
        "=" * 68,
        "[ОРАКУЛ ДОВЕРИЯ] намагниченный цилиндр H_in = %.4f (аналитика -0.5000)" % h_in,
        "-" * 68,
        "%-8s T_магнита=%6.1f C  за_коленом=%2d/%d  худшая_маржа=%+9.2e А/м  сошлось=%s"
        % (tag, res.T_magnet, res.risk.n_demagnetized, res.risk.cell_indices.size,
           res.risk.worst_margin, res.em_converged),
        "-" * 68,
        "ИТОГ: магнит %s (%s)."
        % ("ЧАСТИЧНО РАЗМАГНИЧЕН" if res.risk.n_demagnetized else "ЦЕЛ",
           "снизьте нагрузку/усильте охлаждение или возьмите SmCo"
           if res.risk.n_demagnetized else "запас по демагу есть"),
        "Графики и отчёт: %s" % out,
    ]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(report)
    return {"T_magnet": res.T_magnet, "n_demagnetized": res.risk.n_demagnetized,
            "converged": res.em_converged}


def main(argv: list[str]) -> None:
    path = Path(argv[1]) if len(argv) > 1 else _DEFAULT_CONFIG
    if not path.exists():
        raise SystemExit("Конфиг не найден: %s" % path)
    run(load_config(path))


if __name__ == "__main__":
    main(sys.argv)
