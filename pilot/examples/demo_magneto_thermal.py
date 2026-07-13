"""
Демо-сценарий пилота: связка магнит-тепло-демаг + сравнение NdFeB vs SmCo.

Запуск:  python -m pilot.examples.demo_magneto_thermal
Результат: PNG-графики в pilot/output/ + числовой отчёт в консоль — для РУЧНОЙ проверки.

Считает на верифицированном ядре (magcore.fem2d), ничего в нём не меняя. Геометрия —
магнит-диск в прямоугольной области (реальное сечение мотора — следующий инкремент v1).
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

OUT = Path(__file__).resolve().parents[1] / "output"


def _oracle_cylinder() -> float:
    """Оракул доверия: намагниченный цилиндр H_in должен быть = −M/2 = −0.5 (аналитика)."""
    disk = build_disk_tri_mesh(3.0, 24, 96)
    mesh = disk.mesh
    nu_br = np.zeros((mesh.n_cells, 2))
    mag = [c for c in range(mesh.n_cells) if np.linalg.norm(mesh.cell_centroid(c)) < 1.0]
    nu_br[mag] = [1.0, 0.0]
    res = solve_kelvin_magnetostatic(disk, nu_real=np.ones(mesh.n_cells), magnetization=nu_br)
    return float((res.B_cells[mag] - np.array([1.0, 0.0]))[:, 0].mean())


def run(load: float = 110.0) -> None:
    OUT.mkdir(exist_ok=True)
    mesh = build_structured_rectangle_tri_mesh(24, 24, x0=-2, x1=2, y0=-2, y1=2)
    space = LagrangeP1Space2D(mesh)
    nc = mesh.n_cells
    mask = np.array([np.linalg.norm(mesh.cell_centroid(c)) < 0.8 for c in range(nc)], dtype=bool)
    q = np.where(mask, load, 0.0)
    k = np.full(nc, 1.0)

    nd_mag = n42sh_magnet([1.0, 0.0, 0.0])
    sm_mag = sm2co17_magnet([1.0, 0.0, 0.0])
    common = dict(heat_source_cells=q, k_cells=k, h=2.0, T_amb=20.0,
                  applied_B0=(-0.20, 0.0), em_max_iter=200)
    nd = solve_magneto_thermal_demag(space, nd_mag, mask, **common)
    sm = solve_magneto_thermal_demag(space, sm_mag, mask, **common)

    # --- графики ---
    viz.plot_B_field(mesh, nd.B_cells, title="Поле B (NdFeB)", save_path=OUT / "B_field.png")
    viz.plot_node_scalar(mesh, nd.T_field, title="Температура, °C", label="T, °C",
                         save_path=OUT / "temperature.png")
    viz.plot_demag_risk(mesh, mask, nd.risk,
                        title="NdFeB: T=%.0f°C, за коленом %d/%d" % (
                            nd.T_magnet, nd.risk.n_demagnetized, nd.risk.cell_indices.size),
                        save_path=OUT / "demag_ndfeb.png")
    viz.plot_demag_risk(mesh, mask, sm.risk,
                        title="SmCo: T=%.0f°C, за коленом %d/%d" % (
                            sm.T_magnet, sm.risk.n_demagnetized, sm.risk.cell_indices.size),
                        save_path=OUT / "demag_smco.png")
    viz.plot_material_comparison([("NdFeB", nd_mag), ("SmCo", sm_mag)], H_op=-6.0e5,
                                 save_path=OUT / "material_comparison.png")

    # --- числовой отчёт (для ручной проверки) ---
    h_in = _oracle_cylinder()
    lines = [
        "=" * 68,
        "ПИЛОТ: магнит-тепло-демаг (магнит-диск в области, тепл. нагрузка=%.0f)" % load,
        "=" * 68,
        "[ОРАКУЛ ДОВЕРИЯ] намагниченный цилиндр H_in = %.4f (аналитика -0.5000)" % h_in,
        "-" * 68,
    ]
    for name, r in (("NdFeB", nd), ("SmCo", sm)):
        lines.append(
            "%-6s  T_магнита=%6.1f C  за_коленом=%2d/%d  худшая_маржа=%+9.2e А/м  сошлось=%s"
            % (name, r.T_magnet, r.risk.n_demagnetized, r.risk.cell_indices.size,
               r.risk.worst_margin, r.em_converged)
        )
    lines += [
        "-" * 68,
        "ВЫВОД: при одной тепловой нагрузке (T_магнита ~%.0f C) материал решает судьбу:"
        % nd.T_magnet,
        "       NdFeB %s, SmCo %s."
        % ("за коленом" if nd.risk.n_demagnetized else "цел",
           "за коленом" if sm.risk.n_demagnetized else "цел"),
        "Графики сохранены в: %s" % OUT,
    ]
    report = "\n".join(lines)
    (OUT / "report.txt").write_text(report + "\n", encoding="utf-8")
    try:  # cp1251-консоль Windows не всегда печатает UTF-8 — не роняем расчёт из-за печати
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(report)


if __name__ == "__main__":
    run()
