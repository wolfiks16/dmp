"""
Разбор пользовательского конфига пилота (TOML, читается стандартным tomllib) и сборка
объектов расчёта. Позволяет задавать параметры БЕЗ правки Python — только текстовый файл.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from magcore.domain.magnet_model import (
    AnisotropicBHTMagnet,
    magnet_from_datasheet,
    n42sh_magnet,
    sm2co17_magnet,
)


@dataclass(frozen=True)
class RunConfig:
    magnet: AnisotropicBHTMagnet
    material_name: str
    box_half: float
    magnet_radius: float
    applied_B0: tuple[float, float]
    heat_load: float
    cooling_h: float
    T_ambient: float
    n: int
    output_dir: Path


def build_magnet(cfg: dict) -> tuple[AnisotropicBHTMagnet, str]:
    """Собрать магнит из [magnet]: 'ndfeb' | 'smco' | 'custom' (даташит-параметры СИ)."""
    mat = str(cfg.get("material", "ndfeb")).lower()
    axis = [1.0, 0.0, 0.0]
    if mat == "ndfeb":
        return n42sh_magnet(axis), "NdFeB"
    if mat == "smco":
        return sm2co17_magnet(axis), "SmCo"
    if mat == "custom":
        req = ("Br", "Hcb", "Hk", "Hcj")
        missing = [k for k in req if k not in cfg]
        if missing:
            raise ValueError(f"custom-магнит требует параметры: {missing} (СИ: Тл, А/м).")
        m = magnet_from_datasheet(
            "custom", str(cfg.get("name", "custom magnet")), axis,
            Br=float(cfg["Br"]), Hcb=float(cfg["Hcb"]),
            Hk=float(cfg["Hk"]), Hcj=float(cfg["Hcj"]),
            alpha_Br=float(cfg.get("alpha_Br", 0.12)),
            gamma_Hc=float(cfg.get("gamma_Hc", 0.60)),
        )
        return m, str(cfg.get("name", "custom"))
    raise ValueError(f"неизвестный материал '{mat}' (ожидается ndfeb|smco|custom).")


def load_config(path: str | Path) -> RunConfig:
    """Прочитать TOML-конфиг и собрать RunConfig (с валидацией и разумными умолчаниями)."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    magnet, name = build_magnet(raw.get("magnet", {}))
    geom = raw.get("geometry", {})
    op = raw.get("operating", {})
    mesh = raw.get("mesh", {})
    out = raw.get("output", {})

    return RunConfig(
        magnet=magnet,
        material_name=name,
        box_half=float(geom.get("box_half", 2.0)),
        magnet_radius=float(geom.get("magnet_radius", 0.8)),
        applied_B0=(float(op.get("applied_B0_x", -0.20)), float(op.get("applied_B0_y", 0.0))),
        heat_load=float(op.get("heat_load", 110.0)),
        cooling_h=float(op.get("cooling_h", 2.0)),
        T_ambient=float(op.get("T_ambient", 20.0)),
        n=int(mesh.get("n", 24)),
        output_dir=Path(out.get("dir", "pilot/output")),
    )
