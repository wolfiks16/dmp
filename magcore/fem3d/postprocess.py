from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from magcore.constants import MU0
from magcore.fem2d.model.materials import Air, LinearMaterial
from magcore.fem3d.nonlinear import solve_nonlinear3d
from magcore.fem3d.problem import Problem3D
from magcore.fem3d.scalar import ScalarField3D, assemble_scalar_system, p1_gradients

# ПОСТОБРАБОТКА 3D-РЕШЕНИЯ (этап 3D-4, план — docs/plan_3d_2026-09-11.md): сила и момент на тело,
# коэнергия и энергия поля, поток через сечение, сводка риска размагничивания по объектам.
#
# СИЛА И МОМЕНТ — виртуальная работа (метод Кулона) = тензор Максвелла, усреднённый по воздуху.
# Тело (объект или группа объектов) сдвигается на δ·d, узлы воздуха вокруг — на δ·θ·d, где θ = 1
# на узлах тела и 0 на узлах других тел и внешней границы. Сила — производная коэнергии при
# неизменном узловом потенциале (решение — стационарная точка функционала, поэтому это и полная
# производная коэнергии W' по положению тела — «при неизменных источниках»):
#     F·d = dW'/dδ = −∫_воздух d·T ∇θ dV,     T = μ₀(H⊗H − ½|H|² I) — тензор Максвелла в воздухе.
# Ячейки тела и других тел движутся жёстко (θ в них постоянна) и вклада не дают; вклад только от
# воздуха, где θ меняется. Для кусочно-линейных элементов это ТОЧНАЯ производная дискретной
# коэнергии (закреплено тестом). Поворот на δα вокруг оси a через точку x₀ — смещения узлов
# θ_i·a×(x_i − x₀), ось намагничивания тела поворачивается вместе с ним:
#     τ = −Σ_ячейки V Σ_i θ_i (x_i − x₀) × (T ∇λ_i).
# В непрерывной задаче любая такая θ даёт одну и ту же силу (∇·T = 0 в воздухе); на сетке —
# отличия в пределах погрешности сетки. Весовая функция:
#   'laplace' — гармоническая в воздухе (как «взвешенный тензор напряжений» FEMM): гладкая, вклад
#               собирается со всего воздуха;
#   'layer'   — один слой ячеек вокруг тела (классический метод Кулона);
#   массив θ  — своя (или заранее посчитанная `force_weight`) весовая функция по узлам.
# Тело должно быть отделено от других тел воздухом (пусть тонким зазором): сила «через контакт»
# твёрдых тел по полю не определена. «Воздух» — Air и линейный материал с μ_r = 1.


def _air_like(material) -> bool:
    return isinstance(material, Air) or (isinstance(material, LinearMaterial) and float(material.mu_r) == 1.0)


def _region_ids(problem: Problem3D, names) -> list[int]:
    names = [names] if isinstance(names, str) else list(names)
    if not names:
        raise ValueError("нужен хотя бы один объект.")
    by_name = {r.name: rid for rid, r in problem.regions.items()}
    missing = [nm for nm in names if nm not in by_name]
    if missing:
        raise ValueError(f"нет объектов {missing}; есть: {sorted(by_name)}.")
    return [by_name[nm] for nm in names]


def _cells_of(problem: Problem3D, objects) -> np.ndarray:
    """Маска ячеек объектов (None — вся область)."""
    reg = np.asarray(problem.cell_region)
    if objects is None:
        return np.ones(reg.shape[0], dtype=bool)
    return np.isin(reg, _region_ids(problem, objects))


# ----------------------------------------------------------------- сила и момент
def _weight_constraints(problem: Problem3D, body) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Узлы тела (там θ = 1), узлы, где θ = 0 (другие тела и внешняя граница), и ячейки воздуха.
    Проверки: у тела есть ячейки, тело не касается других тел и внешней границы.
    """
    mesh = problem.mesh
    reg = np.asarray(problem.cell_region)
    body_ids = _region_ids(problem, body)
    air_ids = [rid for rid, r in problem.regions.items() if _air_like(r.material) and rid not in body_ids]
    in_body = np.isin(reg, body_ids)
    in_air = np.isin(reg, air_ids)
    n = mesh.n_vertices
    body_nodes = np.zeros(n, dtype=bool)
    body_nodes[mesh.cells[in_body].ravel()] = True
    solid_nodes = np.zeros(n, dtype=bool)
    solid_nodes[mesh.cells[~in_body & ~in_air].ravel()] = True
    if not body_nodes.any():
        raise ValueError(f"у тела {body} нет ячеек (объект перекрыт или слишком мелкий).")
    if (body_nodes & solid_nodes).any():
        raise ValueError(f"тело {body} касается другого тела: нужен воздушный зазор (или включите "
                         "касающееся тело в состав тела).")
    outer = np.zeros(n, dtype=bool)
    outer[mesh.boundary_faces().ravel()] = True
    if (body_nodes & outer).any():
        raise ValueError(f"тело {body} касается внешней границы области: увеличьте запас домена.")
    return body_nodes, solid_nodes | outer, in_air


def force_weight(problem: Problem3D, body, *, kind: str = "laplace", solver: str = "direct",
                 rtol: float = 1.0e-12) -> np.ndarray:
    """
    Весовая функция θ по узлам (n_vertices,): 1 на узлах тела `body` (имя объекта или список имён),
    0 на узлах других тел (не воздуха) и внешней границы; в воздухе — гармоническая ('laplace') или
    0 ('layer' — меняется только в одном слое ячеек вокруг тела). Зависит только от геометрии —
    для нескольких решений на одной сетке её можно посчитать один раз. `solver` — 'direct' | 'cg'
    (сопряжённые градиенты с диагональным предобуславливателем — для больших сеток, Л-96).
    """
    if kind not in ("laplace", "layer"):
        raise ValueError("kind должен быть 'laplace' или 'layer'.")
    if solver not in ("direct", "cg"):
        raise ValueError("solver должен быть 'direct' или 'cg'.")
    mesh = problem.mesh
    body_nodes, zero_nodes, in_air = _weight_constraints(problem, body)
    theta = body_nodes.astype(float)
    if kind == "layer":
        return theta
    free = np.where(~(body_nodes | zero_nodes))[0]
    if free.size:
        lap = np.where(in_air[:, None, None], np.eye(3), 0.0)          # оператор Лапласа в воздухе
        K = assemble_scalar_system(mesh, lap, np.zeros((mesh.n_cells, 3)))[0]
        K_ff, rhs = K[free, :][:, free].tocsc(), -(K[free, :] @ theta)
        if solver == "direct":
            theta[free] = spla.spsolve(K_ff, rhs)
        else:
            x, info = spla.cg(K_ff, rhs, rtol=rtol, maxiter=20 * free.size,
                              M=sp.diags(1.0 / K_ff.diagonal()))
            if info != 0:
                raise RuntimeError(f"сопряжённые градиенты не сошлись (info={info}).")
            theta[free] = x
    return theta


@dataclass(frozen=True, slots=True)
class ForceTorque3D:
    """Сила [Н] и момент [Н·м] поля на тело; момент — относительно точки `point`."""

    body: tuple
    force: np.ndarray          # (3,) [Н]
    torque: np.ndarray         # (3,) [Н·м] относительно point
    point: np.ndarray          # (3,) [м]
    weight: str                # 'laplace' | 'layer' | 'custom'
    n_weight_cells: int        # ячеек воздуха, где весовая функция меняется


def magnetic_force_torque(field: ScalarField3D, body, *, point=None,
                          weight="laplace") -> ForceTorque3D:
    """
    Сила и момент магнитного поля на тело `body` (имя объекта или список имён) методом виртуальной
    работы (см. шапку модуля). `point` — точка, относительно которой считается момент [м]; None —
    центр объёма тела. `weight` — весовая функция: 'laplace' (по умолчанию), 'layer' или массив θ
    по узлам (например, из `force_weight`; проверяется: 1 на узлах тела, 0 на других телах и границе).
    """
    problem = field.problem
    mesh = problem.mesh
    if isinstance(weight, str):
        theta = force_weight(problem, body, kind=weight)
        label = weight
    else:
        theta = np.array(weight, dtype=float).reshape(-1)
        if theta.shape != (mesh.n_vertices,) or not np.isfinite(theta).all():
            raise ValueError("weight — 'laplace', 'layer' или массив θ по узлам формы (n_vertices,).")
        body_nodes, zero_nodes, _ = _weight_constraints(problem, body)
        if np.any(theta[body_nodes] != 1.0) or np.any(theta[zero_nodes] != 0.0):
            raise ValueError("θ должна быть 1 на узлах тела и 0 на узлах других тел и внешней границы.")
        label = "custom"
    body_cells = _cells_of(problem, body)
    if point is None:
        v_b = field.volumes[body_cells]
        x0 = (mesh.cell_centroids()[body_cells] * v_b[:, None]).sum(axis=0) / v_b.sum()
    else:
        x0 = np.asarray(point, dtype=float)
        if x0.shape != (3,) or not np.isfinite(x0).all():
            raise ValueError("point — три конечных числа [м].")
    grads, vol = p1_gradients(mesh)
    th = theta[mesh.cells]                                             # (M,4)
    act = th.max(axis=1) > th.min(axis=1)                              # воздух, где θ меняется
    H = field.H_cells[act]
    Tm = MU0 * (H[:, :, None] * H[:, None, :]
                - 0.5 * np.einsum("ck,ck->c", H, H)[:, None, None] * np.eye(3))
    Tg = np.einsum("ckl,cil->cik", Tm, grads[act])                     # T ∇λ_i  (M',4,3)
    w = vol[act][:, None] * th[act]                                    # V θ_i
    force = -np.einsum("ci,cik->k", w, Tg)
    r = mesh.vertices[mesh.cells[act]] - x0                            # x_i − x₀  (M',4,3)
    torque = -np.einsum("ci,cik->k", w, np.cross(r, Tg))
    names = (body,) if isinstance(body, str) else tuple(body)
    return ForceTorque3D(body=names, force=force, torque=torque, point=x0, weight=label,
                         n_weight_cells=int(act.sum()))


# ----------------------------------------------------------------- энергия
def coenergy(field: ScalarField3D, objects=None) -> float:
    """
    Коэнергия W' = ∫ w' dV [Дж] в ячейках объектов `objects` (None — вся область); w' = ∫₀^H B·dH
    того закона, по которому получено решение. На всей области (без внешнего поля) это значение
    функционала метода; производная по положению тела — сила.
    """
    if field.coenergy_density is None:
        raise ValueError("в решении нет плотности коэнергии.")
    m = _cells_of(field.problem, objects)
    return float((field.coenergy_density[m] * field.volumes[m]).sum())


def field_energy(field: ScalarField3D, objects=None) -> float:
    """
    Энергия поля W = ∫ (B·H − w') dV [Дж] в ячейках объектов `objects` (None — вся область): у
    воздуха ½μ₀H², у линейных тел ½B·H, у стали ∫₀^B H dB. В магните энергия зависит от выбора
    начального состояния — там не считается (используйте коэнергию).
    """
    if field.coenergy_density is None:
        raise ValueError("в решении нет плотности коэнергии.")
    problem = field.problem
    m = _cells_of(problem, objects)
    if (m & problem.magnet_mask()).any():
        raise ValueError("энергия поля в магните не определена однозначно — исключите магниты.")
    w = np.einsum("ck,ck->c", field.B_cells[m], field.H_cells[m]) - field.coenergy_density[m]
    return float((w * field.volumes[m]).sum())


# ----------------------------------------------------------------- сечение плоскостью и поток
_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def plane_polygons(mesh, point, normal, cells=None):
    """
    Сечение ячеек `cells` (номера; None — все) плоскостью: точка `point` [м], нормаль `normal`.
    У каждой пересечённой ячейки многоугольник сечения — треугольник или четырёхугольник, вершины
    против часовой стрелки, если смотреть навстречу нормали. Вершина на самой плоскости относится
    к стороне «+»: грань, лежащая в плоскости, попадает в сечение один раз.
    Возвращает (номера ячеек (m,), вершины (m, 4, 3) [у треугольника четвёртая = первой],
    число вершин (m,), единичная нормаль (3,)).
    """
    n = np.asarray(normal, dtype=float)
    p0 = np.asarray(point, dtype=float)
    if n.shape != (3,) or p0.shape != (3,) or not (np.isfinite(n).all() and np.isfinite(p0).all()):
        raise ValueError("point и normal — по три конечных числа.")
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        raise ValueError("normal не может быть нулевым.")
    n = n / nn
    sel = np.arange(mesh.n_cells) if cells is None else np.asarray(cells, dtype=np.int64)
    v = mesh.vertices[mesh.cells[sel]]                                  # (m,4,3)
    s = (v - p0) @ n                                                    # (m,4)
    pos = s >= 0.0
    cut = pos.any(axis=1) & ~pos.all(axis=1)
    sel, v, s, pos = sel[cut], v[cut], s[cut], pos[cut]
    if sel.size == 0:
        return sel, np.zeros((0, 4, 3)), np.zeros(0, dtype=np.int64), n
    a, b = _EDGES[:, 0], _EDGES[:, 1]
    crossed = pos[:, a] != pos[:, b]                                    # (m,6)
    den = np.where(crossed, s[:, a] - s[:, b], 1.0)
    t = np.where(crossed, s[:, a] / den, 0.0)
    q = v[:, a] + t[..., None] * (v[:, b] - v[:, a])                    # точки на рёбрах (m,6,3)
    u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)                                                  # u × w = n
    x2, y2 = (q - p0) @ u, (q - p0) @ w                                 # координаты в плоскости
    k = crossed.sum(axis=1)                                             # 3 или 4 вершины
    cx = np.where(crossed, x2, 0.0).sum(axis=1) / k
    cy = np.where(crossed, y2, 0.0).sum(axis=1) / k
    ang = np.where(crossed, np.arctan2(y2 - cy[:, None], x2 - cx[:, None]), np.inf)
    order = np.argsort(ang, axis=1)[:, :4]                              # обход многоугольника
    poly = np.take_along_axis(q, order[..., None], axis=1)
    tri = k == 3
    poly[tri, 3] = poly[tri, 0]
    return sel, poly, k, n


def section_triangles(poly: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Веер многоугольников сечения: (0,1,2) и у четырёхугольников (0,2,3) — вершины (t,3,3) и номер многоугольника (t,)."""
    quad = np.where(k == 4)[0]
    tris = np.concatenate([poly[:, [0, 1, 2]], poly[quad][:, [0, 2, 3]]])
    owner = np.concatenate([np.arange(poly.shape[0]), quad])
    return tris, owner


def flux_through_plane(field: ScalarField3D, point, normal, *, objects=None) -> float:
    """
    Магнитный поток Φ = ∫ B·n dS [Вб] через сечение плоскостью (точка `point` [м], нормаль `normal`)
    в ячейках объектов `objects` (None — вся область). B в ячейке постоянна, поэтому поток — точная
    сумма B·n по многоугольникам сечения ячеек (`plane_polygons` — тот же код рисует разрез).
    """
    sel = np.where(_cells_of(field.problem, objects))[0]
    cells, poly, k, n = plane_polygons(field.problem.mesh, point, normal, sel)
    if cells.size == 0:
        return 0.0
    tris, owner = section_triangles(poly, k)
    area = 0.5 * np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]) @ n
    return float((field.B_cells[cells[owner]] @ n * area).sum())


# ----------------------------------------------------------------- риск размагничивания
@dataclass(frozen=True, slots=True)
class DemagSummary3D:
    """Сводка риска размагничивания по одному магниту-объекту (доли — по объёму)."""

    name: str
    volume: float                  # [м³]
    past_knee_fraction: float      # за коленом сейчас (текущее поле ниже колена)
    damaged_fraction: float        # с необратимой потерей сейчас или раньше (сохранённая доля r < 1)
    beyond_hcj_fraction: float     # r = 0: поле хоть раз было ниже −H_cJ — модель не определена, потеря полная
    worst_margin: float            # min(H∥ − H_колена) по текущему полю [А/м]; < 0 — за коленом. Одна ячейка:
                                   # у краёв к сетке не сходится — не вердикт (вердикт — `flux_loss`, Л-104)
    max_loss: float                # наибольшая потеря ремнантности [Тл] — тоже одна ячейка, не вердикт
    retained: float                # средняя по объёму доля сохранённой ремнантности B_r,eff / B_r(T)


def demag_summary(field: ScalarField3D) -> dict[str, DemagSummary3D]:
    """Сводка карты риска размагничивания по магнитам-объектам: {имя объекта: DemagSummary3D}."""
    risk = field.risk
    if risk is None:
        raise ValueError("в решении нет карты риска — нужен нелинейный решатель и магнит в задаче.")
    problem = field.problem
    reg = np.asarray(problem.cell_region)[risk.cell_indices]
    vol = field.volumes[risk.cell_indices]
    damaged = (risk.H_par < risk.knee_field) if risk.retention is None else (risk.retention < 1.0)
    beyond = np.zeros(vol.size, dtype=bool) if risk.beyond_hcj is None else risk.beyond_hcj
    out: dict[str, DemagSummary3D] = {}
    for rid in np.unique(reg):
        s = reg == rid
        v = vol[s]
        V = float(v.sum())
        name = problem.regions[int(rid)].name
        out[name] = DemagSummary3D(
            name=name, volume=V,
            past_knee_fraction=float(v[risk.demagnetized[s]].sum() / V),
            damaged_fraction=float(v[damaged[s]].sum() / V),
            beyond_hcj_fraction=float(v[beyond[s]].sum() / V),
            worst_margin=float(risk.margin[s].min()),
            max_loss=float(risk.loss[s].max()),
            retained=float((risk.Br_eff[s] * v).sum() / (V * risk.Br_nominal)))
    return out


# ----------------------------------------------------------------- потеря потока — вердикт (Л-104)
# Вердикт о размагничивании — НЕОБРАТИМАЯ ПОТЕРЯ ПОТОКА магнита после события, как в опыте: поток
# до и после при 20 °C без внешнего поля, потеря (Φ₀ − Φ₁)/Φ₀ (docs/experiments/smco_validation_protocol.md,
# этап B; в сборке — падение K_e, этап D). Запас до колена в одной ячейке вердиктом не служит: у краёв
# магнита к сетке не сходится. Среднее поле по магниту — тоже: размагничивание местное и пороговое,
# среднее прячет повреждённый край. БПЛА32, нагрев без тока до 150–180 °C: худшая ячейка за коленом
# уже при 150 °C при потере потока 0,002 %; средний запас положителен до 180 °C, а магниты теряют до
# 8,7 % потока; потеря потока на сетках 1,0 и 0,5 мм различается не больше чем на 0,2 п.п. (Л-104).
# Поток магнита — ∫B·e dV (e — ось намагничивания ячейки) = ∫Φ(s) ds: поток через сечения поперёк
# намагничивания, проинтегрированный вдоль него; в отношении «после / до» длина сокращается.
FLUX_MEASURE_T = 20.0      # температура замера потока до и после события [°C] — как в опыте


def magnet_axial_flux(field: ScalarField3D) -> dict[str, float]:
    """∫B·e dV [Тл·м³] по каждому магниту-объекту решения `field` (e — ось намагничивания ячейки)."""
    problem = field.problem
    mags = problem.magnet_regions()
    if not mags:
        return {}
    reg = np.asarray(problem.cell_region)
    b_par = np.einsum("ck,ck->c", field.B_cells, np.asarray(problem.magnet_axis, dtype=float)) * field.volumes
    return {r.name: float(b_par[reg == r.region_id].sum()) for r in mags}


def new_magnet_flux(problem: Problem3D, *, bc: str = "neumann", T: float = FLUX_MEASURE_T,
                    solver: str = "cg") -> dict[str, float]:
    """Поток ∫B·e dV каждого магнита-объекта нового образца: замер при `T` без внешнего поля."""
    new = solve_nonlinear3d(replace(problem, T=float(T)), bc=bc, solver=solver)
    if not new.converged:
        raise RuntimeError("замер потока нового магнита не сошёлся — потеря потока не определена.")
    return magnet_axial_flux(new)


def flux_loss(problem: Problem3D, retention, *, bc: str = "neumann", T: float = FLUX_MEASURE_T,
              solver: str = "cg", new_flux: dict | None = None) -> dict[str, float]:
    """
    Необратимая потеря потока каждого магнита-объекта после события: доля 1 − Φ_после / Φ_новый.

    `retention` — сохранённая доля ремнантности r по ячейкам после события (`ScalarField3D.retention`).
    Оба замера — при температуре `T` без внешнего поля, с границей `bc` того же расчёта: новый магнит
    (или готовый его поток `new_flux` из `new_magnet_flux` — для повторных событий на той же модели)
    и магнит с долей r. Без повреждения (все r = 1) потеря ровно 0 и решать не нужно: после события
    состояние то же, что у нового магнита.
    Смысл: пока после события задача линейна (ячейки на линиях возврата, сталь не насыщена), по
    взаимности потеря одного магнита — средняя по нему потерянная доля 1 − r с весом индукции нового
    магнита B∥: каждый кусочек весит тот поток, который он нёс. Повреждение соседних магнитов тоже
    меняет поток магнита — это вклад сборки, он в потере учтён.
    """
    names = [r.name for r in problem.magnet_regions()]
    if not names:
        return {}
    if retention is None:
        raise ValueError("нет сохранённой доли r — нужен расчёт с коленом магнита (demag=True).")
    r = np.asarray(retention, dtype=float).reshape(-1)
    if r.shape != (problem.mesh.n_cells,):
        raise ValueError("retention — по числу на ячейку сетки.")
    if not (r[problem.magnet_mask()] < 1.0).any():
        return {n: 0.0 for n in names}
    if new_flux is None:
        new_flux = new_magnet_flux(problem, bc=bc, T=T, solver=solver)
    if set(new_flux) != set(names):
        raise ValueError(f"new_flux — поток каждого магнита модели: {sorted(names)}.")
    after = solve_nonlinear3d(replace(problem, T=float(T)), bc=bc, solver=solver, retention=r)
    if not after.converged:
        raise RuntimeError("замер потока после события не сошёлся — потеря потока не определена.")
    flux_after = magnet_axial_flux(after)
    out = {}
    for n in names:
        if not new_flux[n] > 0.0:
            raise ValueError(f"поток нового магнита {n!r} не положителен — потеря потока не определена.")
        out[n] = 1.0 - flux_after[n] / new_flux[n]
    return out
