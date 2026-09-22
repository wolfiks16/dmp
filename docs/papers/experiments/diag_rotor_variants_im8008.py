# -*- coding: utf-8 -*-
"""
ДИАГНОСТИКА-2 (2026-09-11): варианты формул потерь ротора по волнам прохода 4 (кэш waves/ сверки).

Волны — те, на которых считает сама сверка (точные витки, ток и угол). Для каждой — раскладка
потерь ротора по вариантам формул, чтобы видеть, какие неопределённости что стоят:
  fix          — как в коде после этапа 1: магнит по центральной разности, кольцо по гармоникам
                 (сшивка min(w,2δ)), гистерезис по малым петлям;
  fix_exact    — то же, но точная производная гармоник (без занижения разностью);
  fix_hm       — касательная составляющая поля в магните — с шириной пластины = толщина магнита;
  fix_hm_exact — оба предыдущих;
  fix_F        — кольцо по классике пластины при заданном потоке F(ξ).
Плюс доля зубцовых гармоник, шум пересетки (холостой ход) и КПД регулятора по вариантам
(медь и статор — как в проходе 3: этап 1 их не менял).
"""
from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "docs" / "papers" / "experiments"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(REPO))

import numpy as np                                                          # noqa: E402

import scenario_paper1 as cfg                                               # noqa: E402
from fig_im8008_section import IM8008                                       # noqa: E402
from magcore.fem2d.machines.iron_loss import (                              # noqa: E402
    SteinmetzCoefficients,
    hysteresis_density_minor_loops,
)
from magcore.fem2d.machines.magnet_loss import (                            # noqa: E402
    effective_eddy_thickness,
    magnet_segment_width,
    rotor_side_cells,
    skin_depth,
)
from magcore.fem2d.machines.pmsm_outrunner import build_outrunner_spm_pmsm  # noqa: E402


def P(*a) -> None:
    print(*a, flush=True)


PARAMS = dataclasses.replace(IM8008, mesh_size=1.6e-3, mesh_size_by_region=None)
CF = SteinmetzCoefficients.steel10_laminated(0.2e-3)
SIG_RING, MU_RING = 1.0 / 0.14e-6, 500.0
SIG_PM = cfg.magnet_conductivity("ndfeb", 20.0)
W_SEG = magnet_segment_width(PARAMS, 1)
D_RING, H_MAG = PARAMS.h_rotor_yoke, PARAMS.h_magnet
SPAN90 = 2.0 * math.pi / math.gcd(PARAMS.n_slots, PARAMS.n_poles)
SPAN10 = 2.0 * math.pi / PARAMS.n_slots
NCOPY = int(round(SPAN90 / SPAN10))

MOTOR = REPO / "docs" / "motors" / "scorpion_im8008"
CACHE = json.loads((MOTOR / "core_loss_cache.json").read_text(encoding="utf-8"))
ST = next(v for k, v in CACHE.items() if k.startswith("stator_loaded|"))
# медь (эффект близости — нижняя граница) и КПД изм. из прохода 3 в рабочих точках
COPPER = [1.2, 14.0, 37.0, 82.4]
ETA_MEAS = [80.7, 87.2, 85.8, 81.8]

geo0 = build_outrunner_spm_pmsm(PARAMS)
mag_idx, ry_idx = rotor_side_cells(geo0)
nm, nc = mag_idx.size, geo0.mesh.n_cells
vol = np.array([geo0.mesh.cell_area(c) for c in range(nc)], float) * PARAMS.axial_length
vm, vr = vol[mag_idx], vol[ry_idx]
cent = np.array([geo0.mesh.cell_centroid(int(c)) for c in mag_idx], float)
er = cent / np.linalg.norm(cent, axis=1)[:, None]


def f_ref(span, n):
    return (2.0 * math.pi * n / 60.0) / span


def msd_harm(Bw, span, n, kind):
    """Вклад гармоники m пролёта в ⟨(dB/dt)²⟩ по ячейкам: разность ('cd') или точная производная."""
    N = Bw.shape[0]
    X = np.fft.rfft(Bw, axis=0)
    A2 = (2.0 * np.abs(X) / N) ** 2
    A2[0] = 0.0
    if N % 2 == 0:
        A2[-1] = 0.0
    A2 = A2.sum(axis=2)
    m = np.arange(A2.shape[0])
    dphi = 2.0 * math.pi / N
    g = m.astype(float) if kind == "exact" else np.sin(m * dphi) / dphi
    fr = f_ref(span, n)
    return m, m * fr, (2.0 * math.pi * fr) ** 2 * (g ** 2)[:, None] * A2 / 2.0


def lamination_F(xi):
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    out = np.ones_like(xi)
    mid = (xi > 1e-3) & (xi <= 30.0)
    x = xi[mid]
    out[mid] = (3.0 / x) * (np.sinh(x) - np.sin(x)) / (np.cosh(x) - np.cos(x))
    big = xi > 30.0
    out[big] = 3.0 / xi[big]
    return out


def variants(Bw, span, n):
    r = {}
    Bm, Br = Bw[:, :nm], Bw[:, nm:]
    brad = (Bm * er[None]).sum(axis=2)
    btan = Bm[:, :, 1] * er[None, :, 0] - Bm[:, :, 0] * er[None, :, 1]
    for kind in ("cd", "exact"):
        m, fm, Sm = msd_harm(Bm, span, n, kind)
        by_m = SIG_PM * W_SEG ** 2 / 12.0 * (Sm * vm[None]).sum(axis=1)
        r["mag_" + kind] = float(by_m.sum())
        r["mag_by_m_" + kind] = by_m
        _, _, Srad = msd_harm(brad[:, :, None], span, n, kind)
        _, _, Stan = msd_harm(btan[:, :, None], span, n, kind)
        r["rad_" + kind] = float(SIG_PM * W_SEG ** 2 / 12.0 * (Srad.sum(axis=0) * vm).sum())
        r["tan_w_" + kind] = float(SIG_PM * W_SEG ** 2 / 12.0 * (Stan.sum(axis=0) * vm).sum())
        r["tan_h_" + kind] = float(SIG_PM * H_MAG ** 2 / 12.0 * (Stan.sum(axis=0) * vm).sum())
        _, fm, Sr = msd_harm(Br, span, n, kind)
        per_m = (Sr * vr[None]).sum(axis=1)
        weff = np.array([effective_eddy_thickness(D_RING, f, SIG_RING, MU_RING) for f in fm])
        xi = np.array([D_RING / skin_depth(f, SIG_RING, MU_RING) for f in fm])
        r["ring_by_m_" + kind] = SIG_RING / 12.0 * weff ** 2 * per_m
        r["ring_" + kind] = float(r["ring_by_m_" + kind].sum())
        r["ringF_" + kind] = float((SIG_RING * D_RING ** 2 / 12.0 * lamination_F(xi) * per_m).sum())
        r["m"], r["fm"] = m, fm
    q = hysteresis_density_minor_loops(Br, np.arange(Br.shape[1]), Br.shape[1], freq=f_ref(span, n),
                                       coeffs=CF)
    r["hyst"] = float((q * vr).sum())
    return r


TOTALS = (
    ("fix", "как в коде после этапа 1", lambda r: r["mag_cd"] + r["ring_cd"] + r["hyst"]),
    ("fix_exact", "  + точная производная", lambda r: r["mag_exact"] + r["ring_exact"] + r["hyst"]),
    ("fix_hm", "  + касат. по толщине магн.", lambda r: r["rad_cd"] + r["tan_h_cd"] + r["ring_cd"] + r["hyst"]),
    ("fix_hm_exact", "  + оба", lambda r: r["rad_exact"] + r["tan_h_exact"] + r["ring_exact"] + r["hyst"]),
    ("fix_F", "  + кольцо F(ξ)", lambda r: r["mag_cd"] + r["ringF_cd"] + r["hyst"]),
)


def slot_share(by_m, m):
    tot = float(by_m.sum())
    return float(by_m[(m > 0) & (m % NCOPY == 0)].sum()) / tot if tot > 0 else float("nan")


def show(r, title):
    P(title)
    P("   магнит: разность %.2f / точная %.2f Вт; радиальная %.2f, касательная %.2f (по толщине магнита %.2f);"
      " зубцовые гармоники %.0f %%" % (r["mag_cd"], r["mag_exact"], r["rad_cd"], r["tan_w_cd"], r["tan_h_cd"],
                                      100 * slot_share(r["mag_by_m_cd"], r["m"])))
    P("   кольцо: вихревые по гармоникам %.2f (точная %.2f), пластина F(ξ) %.2f (%.2f); гистерезис %.3f Вт;"
      " зубцовые %.0f %%" % (r["ring_cd"], r["ring_exact"], r["ringF_cd"], r["ringF_exact"], r["hyst"],
                            100 * slot_share(r["ring_by_m_cd"], r["m"])))
    for key, lab in (("mag_by_m_cd", "магнит"), ("ring_by_m_cd", "кольцо")):
        by = r[key]
        top = np.argsort(by)[::-1][:6]
        P("   %s, крупнейшие гармоники (m: частота, Вт): %s"
          % (lab, ", ".join("%d: %.0f Гц %.2f" % (r["m"][k], r["fm"][k], by[k]) for k in top)))


def load_waves():
    out = {}
    for path in sorted((MOTOR / "waves").glob("*.npz")):
        d = np.load(path, allow_pickle=False)
        key = str(d["key"])
        fields = dict(kv.split("=", 1) for kv in key.split("|")[1:])
        if int(fields["cells"]) != nc or not np.array_equal(d["idx"], np.concatenate([mag_idx, ry_idx])):
            raise RuntimeError("волна %s снята на другой сетке" % path.name)
        out[fields["i"]] = d["B"]
    return out


def main():
    waves = load_waves()
    n_top = ST["n"][-1]
    P("=" * 100)
    P("ВАРИАНТЫ ФОРМУЛ ПОТЕРЬ РОТОРА по волнам прохода 4 (кэш сверки: %d волн)" % len(waves))
    P("=" * 100)

    B0 = waves["0"]
    r0 = variants(B0, SPAN90, n_top)
    show(r0, "\nХОЛОСТОЙ ХОД при %.0f об/мин:" % n_top)
    K = B0.shape[0] // NCOPY
    B9 = B0.reshape(NCOPY, K, B0.shape[1], 2)
    Bmean = B9.mean(axis=0)
    rden = variants(np.tile(Bmean, (NCOPY, 1, 1)), SPAN90, n_top)
    P("   очищенная от шума волна (среднее %d зубцовых делений): %s"
      % (NCOPY, "; ".join("%s %.2f Вт" % (k, f(rden)) for k, _, f in TOTALS)))

    res = []
    for j, (n, i) in enumerate(zip(ST["n"], ST["i"])):
        key = repr(float(i))
        if key not in waves:
            P("\n%4.0f об/мин: волны нет — пропуск" % n)
            res.append(None)
            continue
        r = variants(waves[key], SPAN90, n)
        res.append(r)
        show(r, "\n%4.0f об/мин, ток %.1f А:" % (n, i))

    P("\n" + "=" * 100)
    P("ИТОГ: потери ротора [Вт] / подразумеваемый КПД регулятора (медь и статор — как в проходе 3)")
    P("=" * 100)
    P("%-30s" % "вариант" + "".join("%18s" % ("%4.0f об/мин" % n) for n in ST["n"]))
    for key, lab, fun in TOTALS:
        row = "%-30s" % lab
        for j, r in enumerate(res):
            if r is None:
                row += "%18s" % "—"
                continue
            n, M = ST["n"][j], ST["M_meas"][j]
            p_mech = M * 2.0 * math.pi * n / 60.0
            p_rot = fun(r)
            eta_mod = p_mech / (p_mech + COPPER[j] + ST["stator"][j] + p_rot)
            row += "%18s" % ("%.1f / %.3f" % (p_rot, ETA_MEAS[j] / 100.0 / eta_mod))
        P(row)


if __name__ == "__main__":
    main()
