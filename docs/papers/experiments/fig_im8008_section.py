# -*- coding: utf-8 -*-
"""
СЕЧЕНИЕ РЕАЛЬНОГО ДВИГАТЕЛЯ Scorpion IM-8008-100kv (объект расчёта с 2026-09-09).

Три источника размеров, и они РАЗЛИЧАЮТСЯ ПО ДОСТОВЕРНОСТИ — на рисунке это подписано:
  ПАСПОРТ  — 36N40P, статор Ø80,0 × 8,0 мм, лист 0,2 мм (сайт Scorpion, сверено 2026-09-09);
  CAD      — колокол Ø87,20 мм (8 цилиндрических граней R=43,60 в STEP), Ø80,00 подтверждён
             девятью гранями, посадка Ø43 у основания;
  ИДЕНТИФ. — зазор, толщина магнита, ярмо ротора, ярмо статора, доля зубца, охват магнитом,
             витки: НЕ публикует никто, выведены из физики и паспорта (см. ниже).

Как идентифицировано:
  · Радиальный бюджет ротора ЖЁСТКО задан: 43,60 − 40,00 = 3,60 мм на зазор + магнит + ярмо.
    Требуемое ярмо по потоку полюса при B_ярма 1,4–1,6 Тл выходит 0,9–1,1 мм ⇒ на зазор с
    магнитом остаётся 2,5 мм. Принято 0,5 / 2,0 / 1,10 мм (зазор серийных аутраннеров
    0,4–0,6 мм).
  · Посадка статора выбрана ФИЗИКОЙ, а не на глаз: из двух кандидатов CAD (Ø43 и Ø55) при
    витках, подогнанных под паспортное kV = 100, плотность тока на ПАСПОРТНОМ длительном
    токе 24 А выходит 16,4 А/мм² (Ø43) против 23,9 (Ø55). Второй вариант нереален ⇒ Ø43.
  · Витки подобраны так, чтобы K_e совпал с паспортным kV = 100 в конвенции «от шины,
    шеститактный регулятор» (см. `machines/conventions.py`).

⚠ НЕ ПРОВЕРЕНО НЕЗАВИСИМО: марка магнита и стали приняты представительными (N42SH, сталь 10);
  измеренный КПД (макс. 87,2 %) в подгонке НЕ участвовал и остаётся независимым контролем.

Запуск: PYTHONPATH=<repo> python docs/papers/experiments/fig_im8008_section.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
from matplotlib.collections import PolyCollection                        # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402

import scenario_paper1 as cfg                                            # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet                     # noqa: E402
from magcore.fem2d.machines.conventions import ke_from_kv                # noqa: E402
from magcore.fem2d.machines.excitation import slot_areas                 # noqa: E402
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, Region  # noqa: E402
from magcore.fem2d.machines.scenario import machine_scenario             # noqa: E402

# ------------------------------------------------------------------ геометрия объекта
IM8008 = OutrunnerPMSMParams(
    n_slots=36, n_poles=40,          # ПАСПОРТ: 36N40P
    R_bore=21.5e-3,                  # CAD Ø43 у основания + отбор по плотности тока
    h_stator_yoke=3.0e-3,            # ИДЕНТИФ.
    h_tooth=15.5e-3,                 # ИДЕНТИФ. (дополняет до Ø80 паспорта)
    air_gap=0.5e-3,                  # ИДЕНТИФ. (практика 0,4–0,6 мм)
    h_magnet=2.0e-3,                 # ИДЕНТИФ. (бюджет 3,60 мм)
    h_rotor_yoke=1.1e-3,             # ИДЕНТИФ. (до CAD Ø87,20)
    tooth_width_frac=0.5,            # ИДЕНТИФ.
    magnet_embrace=0.85,             # ИДЕНТИФ.
    axial_length=8.0e-3,             # ПАСПОРТ: высота статора 8,0 мм
    mesh_size=1.2e-3,
    mesh_size_by_region={"magnet": 0.35e-3, "air_gap": 0.167e-3},
)
KV, U_DC, I_CONT, I_PEAK = 100.0, 44.4, 24.0, 45.0        # ПАСПОРТ
SLOT_FILL = 0.45                                           # ИДЕНТИФ.

COL = {int(Region.AIR_GAP): "#f4f6f8", int(Region.STATOR_YOKE): "#9aa7b4",
       int(Region.TOOTH): "#9aa7b4", int(Region.SLOT): "#d9a441",
       int(Region.MAGNET): "#c0504d", int(Region.ROTOR_YOKE): "#7f8c9b"}
LBL = {int(Region.STATOR_YOKE): "ярмо статора и зубцы (сталь)",
       int(Region.SLOT): "паз с обмоткой", int(Region.MAGNET): "магнит",
       int(Region.ROTOR_YOKE): "ярмо ротора (колокол)", int(Region.AIR_GAP): "воздух и зазор"}


def build():
    sc = machine_scenario(IM8008, n42sh_magnet((1.0, 0.0, 0.0)),
                          cfg.STEEL_KINDS["steel10"]["curve"]())
    sol = sc.solve(T=20.0, i_peak=0.0, gamma_elec=0.0, turns_per_slot=10.0, max_iter=400)
    ke_per_turn = sc.back_emf_constant(sol, turns_per_slot=10.0) / 10.0
    turns = ke_from_kv(KV, "bus_sixstep") / ke_per_turn
    op = sc.operating_point(sol)
    a_cu = SLOT_FILL * float(np.mean(slot_areas(sc.geometry)))
    j = lambda i_dc: turns * (math.pi / 3 * i_dc / math.sqrt(2)) / a_cu / 1e6
    return sc, sol, dict(
        turns=turns, ke=ke_per_turn * turns,
        pc=float(np.average(op.permeance, weights=op.cell_volume)),
        b_op=float(np.average(op.B_op, weights=op.cell_volume)),
        j_cont=j(I_CONT), j_peak=j(I_PEAK), a_slot=float(np.mean(slot_areas(sc.geometry))),
        cells=sc.geometry.mesh.n_cells)


def draw(sc, info):
    p, mesh = IM8008, sc.geometry.mesh
    v = np.asarray(mesh.vertices) * 1e3
    tri = v[np.asarray(mesh.cells)]
    reg = np.asarray(sc.geometry.region)

    fig = plt.figure(figsize=(13.0, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    for r, col in COL.items():
        sel = reg == r
        if sel.any():
            ax.add_collection(PolyCollection(tri[sel], facecolors=col,
                                             edgecolors="white", linewidths=0.12))
    R = p.R_out * 1e3
    ax.set_xlim(-R * 1.04, R * 1.04); ax.set_ylim(-R * 1.04, R * 1.04)
    ax.set_aspect("equal"); ax.set_xlabel("координата X, мм"); ax.set_ylabel("координата Y, мм")
    ax.tick_params(labelsize=9)
    ax.legend(handles=[Patch(facecolor=COL[k], edgecolor="#b6bfc8", label=LBL[k])
                       for k in (Region.STATOR_YOKE, Region.SLOT, Region.MAGNET,
                                 Region.ROTOR_YOKE, Region.AIR_GAP)],
              loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
    rows = [
        ("ПАСПОРТ (сайт Scorpion)", "", ""),
        ("пазов / полюсов", "36 / 40", ""),
        ("диаметр статора", "80,0 мм", ""),
        ("высота пакета статора", "8,0 мм", ""),
        ("толщина листа", "0,2 мм", ""),
        ("коэффициент по оборотам", "100 об/(мин·В)", ""),
        ("длительный ток и мощность", "24 А / 1065 Вт", ""),
        ("пиковый ток и мощность", "45 А / 2000 Вт", ""),
        ("", "", ""),
        ("ИЗ CAD-МОДЕЛИ (файл STEP)", "", ""),
        ("наружный диаметр колокола", "87,20 мм", "8 граней R = 43,60"),
        ("диаметр статора — сверка", "80,00 мм", "9 граней, совпал"),
        ("посадка статора", "43,0 мм", "выбрана по плотности тока"),
        ("", "", ""),
        ("ИДЕНТИФИЦИРОВАНО", "", "не публикует никто"),
        ("воздушный зазор", "0,50 мм", "практика 0,4–0,6"),
        ("толщина магнита", "2,00 мм", "бюджет 3,60 мм"),
        ("ярмо ротора", "1,10 мм", "по потоку 0,9–1,1"),
        ("ярмо статора", "3,00 мм", ""),
        ("витков на паз", "%.1f" % info["turns"], "подогнано под kV = 100"),
        ("", "", ""),
        ("ЧТО ПОЛУЧИЛОСЬ", "", ""),
        ("рабочая индукция магнита", "%.3f Тл" % info["b_op"], ""),
        ("коэффициент проницания", "%.2f" % info["pc"], ""),
        ("площадь паза", "%.1f мм²" % (info["a_slot"] * 1e6), ""),
        ("плотность тока, длительно", "%.1f А/мм²" % info["j_cont"], "практика 12–17"),
        ("плотность тока, пик", "%.1f А/мм²" % info["j_peak"], ""),
        ("ячеек сетки", "%d" % info["cells"], ""),
    ]
    y = 0.985
    for a, b, c in rows:
        if not a:
            y -= 0.014; continue
        head = a.isupper() or a.startswith(("ПАСПОРТ", "ИЗ CAD", "ИДЕНТИФ", "ЧТО"))
        ax2.text(0.0, y, a, fontsize=9.6, family="DejaVu Sans", va="top",
                 weight="bold" if head else "normal",
                 color="#1a1a1a" if head else "#333333")
        if b:
            ax2.text(0.58, y, b, fontsize=9.6, va="top", weight="bold", color="#1a1a1a")
        if c:
            ax2.text(0.79, y, c, fontsize=8.4, va="top", color="#6b7480")
        y -= 0.0345 if head else 0.0335

    fig.suptitle("Сечение Scorpion IM-8008-100kv: паспорт и CAD задают габариты, "
                 "внутренние размеры идентифицированы",
                 fontsize=13.5, y=0.975, weight="bold")
    out = HERE / "figs" / "fig_im8008_section.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print("сохранено: %s" % out)
    return out


if __name__ == "__main__":
    sc, sol, info = build()
    print("витков на паз %.2f -> K_e = %.5f В·с/рад (цель %.5f)"
          % (info["turns"], info["ke"], ke_from_kv(KV, "bus_sixstep")))
    print("плотность тока: длительно %.1f, пик %.1f А/мм²" % (info["j_cont"], info["j_peak"]))
    print("P_c = %.2f, B_раб = %.3f Тл, ячеек %d" % (info["pc"], info["b_op"], info["cells"]))
    draw(sc, info)
