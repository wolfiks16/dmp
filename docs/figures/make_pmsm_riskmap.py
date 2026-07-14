import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import TwoSlopeNorm

from magcore.fem2d.machines import (
    OutrunnerPMSMParams, build_outrunner_spm_pmsm, Region, star_of_slots_layout,
    solve_machine_static, evaluate_demag_impact,
)
from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve

OUT = sys.argv[1] if len(sys.argv) > 1 else "pmsm_riskmap.png"
T = 140.0

g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.0028))  # чуть плотнее для картинки
magnet = n42sh_magnet(easy_axis=(1, 0, 0)); steel = m270_35a_bh_curve()
layout = star_of_slots_layout(g.params.n_slots, g.params.n_poles)

# Поле |B| при worst-case нагрузке (T=140, d-ток) + машинный итог P5.
r = solve_machine_static(g, magnet, steel, T=T, layout=layout,
                         i_peak=30.0, gamma_elec=np.pi, turns_per_slot=40.0, max_iter=400)
imp = evaluate_demag_impact(g, magnet, steel, layout, i_peak=30.0, gamma_elec=np.pi,
                            turns_per_slot=40.0, T=T)
print(f"conv={r.converged} demag_area={imp.aggregate.demag_area_fraction*100:.1f}% "
      f"mean_loss={imp.aggregate.mean_loss_frac*100:.2f}% lam_drop={imp.flux_linkage_drop_frac*100:.2f}%")

V = g.mesh.vertices
tri = Triangulation(V[:, 0] * 1e3, V[:, 1] * 1e3, g.mesh.cells)  # мм
Bmag = np.hypot(r.B_cells[:, 0], r.B_cells[:, 1])

fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))

# --- (A) |B| по всей машине ---
axA = axes[0]
tpc = axA.tripcolor(tri, facecolors=Bmag, cmap="viridis", shading="flat")
axA.set_aspect("equal"); axA.set_title(f"(A) |B|, Тл — worst-case нагрузка, T={T:.0f}°C")
axA.set_xlabel("x, мм"); axA.set_ylabel("y, мм")
fig.colorbar(tpc, ax=axA, fraction=0.046, pad=0.04, label="|B|, Тл")

# --- (B) карта риска демага: только магнит, по марже к колену ---
axB = axes[1]
margin = np.full(g.mesh.n_cells, np.nan)
idx = imp.risk.cell_indices
margin[idx] = imp.risk.margin / 1e3        # кА/м
# фон машины бледным контуром
axB.tripcolor(tri, facecolors=np.ones(g.mesh.n_cells), cmap="Greys", vmin=0, vmax=6, shading="flat")
vmin, vmax = np.nanmin(margin), np.nanmax(margin)
norm = TwoSlopeNorm(vmin=min(vmin, -1e-6), vcenter=0.0, vmax=max(vmax, 1e-6))
mtri = axB.tripcolor(tri, facecolors=margin, cmap="RdBu", norm=norm, shading="flat")
axB.set_aspect("equal")
axB.set_title(f"(B) Запас до колена, кА/м — красное = за коленом (необратимо)")
axB.set_xlabel("x, мм"); axB.set_ylabel("y, мм")
cb = fig.colorbar(mtri, ax=axB, fraction=0.046, pad=0.04, label="H_par − H_knee, кА/м")

fig.suptitle(
    f"Outrunner SPM PMSM 12N14P (N42SH, M270) — P4/P5: связанный расчёт + необратимый демаг\n"
    f"За коленом {imp.aggregate.demag_area_fraction*100:.1f}% площади магнита · "
    f"средняя потеря Br {imp.aggregate.mean_loss_frac*100:.2f}% · "
    f"падение ЭДС/момента {imp.flux_linkage_drop_frac*100:.2f}%",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, dpi=130)
print("saved", OUT)
