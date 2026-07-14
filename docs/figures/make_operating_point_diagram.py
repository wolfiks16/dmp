import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from magcore.fem2d.machines import (
    OutrunnerPMSMParams, build_outrunner_spm_pmsm, star_of_slots_layout,
    solve_machine_static, magnet_operating_point,
)
from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve

OUT = sys.argv[1]
g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.004))
magnet = n42sh_magnet(easy_axis=(1, 0, 0)); steel = m270_35a_bh_curve()

# Рабочая точка (ХХ, 20°C) — среднее по объёму.
r = solve_machine_static(g, magnet, steel, T=20.0, max_iter=400)
op = magnet_operating_point(g, r, magnet, T=20.0)
Hd = op.volume_weighted_mean_H_op() / 1e3      # кА/м
Bd = float(np.average(op.B_op, weights=op.cell_volume))
Hd_worst = op.worst_H_op() / 1e3
Bd_worst = float(op.B_op.min())

# Кривая размагничивания при 20°C (2-й квадрант).
curve = magnet.curve_at(20.0)
H = curve.H_values / 1e3                        # кА/м
B = curve.B_values
sel = H <= 5
H, B = H[sel], B[sel]
H_knee = magnet.knee_field(20.0) / 1e3
Br = magnet.Br(20.0)

fig, ax = plt.subplots(figsize=(9, 6.2))
ax.plot(H, B, "b-", lw=2.5, label="кривая размагничивания B(H), 20°C")
ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)

# Линия нагрузки (проницаемости) от начала через рабочую точку.
ax.plot([0, 1.35 * Hd], [0, 1.35 * Bd], "g--", lw=1.8, label="линия нагрузки (наклон = P_c)")

# Рабочая точка (среднее по объёму) + проекции на обе оси.
ax.plot([Hd], [Bd], "ro", ms=11, zorder=5)
ax.plot([Hd, Hd], [0, Bd], "r:", lw=1.4); ax.plot([0, Hd], [Bd, Bd], "r:", lw=1.4)
ax.annotate(f"РАБОЧАЯ ТОЧКА\n(H_d={Hd:.0f} кА/м,  B_d={Bd:.2f} Тл)",
            (Hd, Bd), textcoords="offset points", xytext=(15, 12),
            fontsize=11, color="darkred", fontweight="bold")
ax.annotate(f"B_d={Bd:.2f} Тл\n(«твоя» рабочая точка)", (0, Bd),
            textcoords="offset points", xytext=(8, -4), fontsize=10, color="red")
ax.annotate(f"H_d={Hd:.0f}", (Hd, 0), textcoords="offset points",
            xytext=(-10, 8), fontsize=10, color="red", ha="right")

# Худшая (кромочная) рабочая точка по объёму.
ax.plot([Hd_worst], [Bd_worst], "mo", ms=8, zorder=5)
ax.annotate(f"худшая по объёму\n(кромка): B_d={Bd_worst:.2f} Тл",
            (Hd_worst, Bd_worst), textcoords="offset points", xytext=(12, -34),
            fontsize=9.5, color="purple")

# Колено и Br.
ax.plot([H_knee], [magnet.B_major_parallel(H_knee * 1e3, 20.0)], "ks", ms=8)
ax.annotate(f"КОЛЕНО H_knee={H_knee:.0f} кА/м\n(ниже — необратимая потеря)",
            (H_knee, magnet.B_major_parallel(H_knee * 1e3, 20.0)),
            textcoords="offset points", xytext=(10, -40), fontsize=9.5)
ax.plot([0], [Br], "b^", ms=8); ax.annotate(f"Br={Br:.2f} Тл", (0, Br),
            textcoords="offset points", xytext=(8, 4), fontsize=10, color="blue")

ax.set_xlabel("H — поле, кА/м (2-й квадрант, размагничивающее < 0)", fontsize=11)
ax.set_ylabel("B — индукция, Тл", fontsize=11)
ax.set_title("Рабочая точка магнита = точка (H_d, B_d) на кривой размагничивания\n"
             "B_d [Тл] и H_d [кА/м] — ДВЕ координаты ОДНОЙ точки (N42SH)", fontsize=12)
ax.set_xlim(min(H_knee * 1.15, -1450), 60); ax.set_ylim(-0.05, 1.45)
ax.legend(loc="lower right", fontsize=10); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=135)
print(f"Bd={Bd:.3f} Hd={Hd:.0f} Pc={op.volume_weighted_mean_permeance():.2f} saved {OUT}")
