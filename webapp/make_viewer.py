"""
Сборка вьюпорта геометрии (первый реальный кусок UI): экспортирует РЕАЛЬНУЮ сцену
(сетка + регионы) из Problem2D и внедряет её в HTML-шаблон вьюпорта. На выходе —
самодостаточный `geometry_viewer.html`, рисующий настоящую сетку решателя (зум/панорама,
регионы по материалам, каркас, оси намагничивания). Данные — из magcore, а не стилизация.

Запуск:  python -m webapp.make_viewer   (нужен gmsh для генерации геометрии)
"""
from __future__ import annotations

import json
from pathlib import Path

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines import (
    OutrunnerPMSMParams,
    build_outrunner_spm_pmsm,
    pmsm_to_problem,
)
from magcore.fem2d.model import problem_to_scene

HERE = Path(__file__).parent


def build_pmsm_scene(mesh_size: float = 0.0026) -> dict:
    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=mesh_size))
    problem = pmsm_to_problem(g, n42sh_magnet((1, 0, 0)), m270_35a_bh_curve(), T=20.0)
    return problem_to_scene(problem, ndigits=3)


def main() -> None:
    scene = build_pmsm_scene()
    scene_json = json.dumps(scene, ensure_ascii=False)
    if "</script" in scene_json.lower():
        raise ValueError("scene JSON содержит </script — небезопасно для инлайна.")
    template = (HERE / "geometry_viewer_template.html").read_text(encoding="utf-8")
    out = template.replace("__SCENE__", scene_json)
    (HERE / "geometry_viewer.html").write_text(out, encoding="utf-8")
    print(f"geometry_viewer.html: {len(scene['cells'])} ячеек, {round(len(out) / 1024, 1)} КБ")


if __name__ == "__main__":
    main()
