from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from magcore.fem2d.mesh import TriangleMesh, signed_area2

# Параметрический генератор сечения OUTRUNNER SPM PMSM (магниты на роторе снаружи,
# слотованный статор внутри). Топология по радиусу изнутри наружу:
#   вал/расточка · ЯРМО СТАТОРА · зубья+пазы · ЗАЗОР · МАГНИТЫ+межполюс. воздух · ЯРМО РОТОРА
# Геометрия строится в gmsh, конформно мешится; области размечаются АНАЛИТИЧЕСКИ по
# центроиду ячейки (радиальные полосы + угловые секторы) — независимо от бухгалтерии gmsh.


class Region(IntEnum):
    AIR_GAP = 0        # зазор + межполюсный воздух (немагнитный, source-free)
    STATOR_YOKE = 1
    TOOTH = 2          # зубец статора (сталь)
    SLOT = 3           # паз (обмотка/воздух)
    MAGNET = 4
    ROTOR_YOKE = 5


REGION_NAMES = {int(r): r.name.lower() for r in Region}


@dataclass(frozen=True)
class OutrunnerPMSMParams:
    """Геометрические параметры сечения (СИ, метры/радианы). Толщины — по радиусу наружу."""

    n_slots: int = 12
    n_poles: int = 14
    R_bore: float = 0.010          # радиус расточки статора (вал), м
    h_stator_yoke: float = 0.004   # толщина ярма статора
    h_tooth: float = 0.008         # радиальная длина зубца/паза
    air_gap: float = 0.001         # зазор
    h_magnet: float = 0.003        # толщина магнита
    h_rotor_yoke: float = 0.003    # толщина ярма ротора
    tooth_width_frac: float = 0.5  # доля зубцового шага, занятая зубцом (0..1)
    magnet_embrace: float = 0.83   # охват полюса магнитом (доля полюсного шага, 0..1)
    axial_length: float = 0.030    # осевая длина (для масштаба момента; в 2D не в сетке)
    mesh_size: float | None = None # характерный размер элемента (по умолч. ~ зазор)

    # --- производные радиусы ---
    @property
    def R_sy(self) -> float:       # наружный радиус ярма статора = корень зубца
        return self.R_bore + self.h_stator_yoke

    @property
    def R_s_out(self) -> float:    # кончики зубьев
        return self.R_sy + self.h_tooth

    @property
    def R_mag_in(self) -> float:   # внутренняя поверхность магнитов
        return self.R_s_out + self.air_gap

    @property
    def R_mag_out(self) -> float:
        return self.R_mag_in + self.h_magnet

    @property
    def R_out(self) -> float:      # наружный радиус ротора
        return self.R_mag_out + self.h_rotor_yoke

    def validate(self) -> None:
        if self.n_slots < 3 or self.n_poles < 2 or self.n_poles % 2 != 0:
            raise ValueError("n_slots>=3 и n_poles>=2 чётное.")
        for nm in ("R_bore", "h_stator_yoke", "h_tooth", "air_gap", "h_magnet", "h_rotor_yoke"):
            if getattr(self, nm) <= 0:
                raise ValueError(f"{nm} must be positive.")
        if not (0.0 < self.tooth_width_frac < 1.0):
            raise ValueError("tooth_width_frac in (0,1).")
        if not (0.0 < self.magnet_embrace <= 1.0):
            raise ValueError("magnet_embrace in (0,1].")


@dataclass(frozen=True)
class MachineGeometry:
    mesh: TriangleMesh
    region: np.ndarray             # (n_cells,) коды Region
    magnet_easy_axis: np.ndarray   # (n_cells,2) радиальная ось·знак в магните, иначе 0
    slot_id: np.ndarray            # (n_cells,) номер паза 0..n_slots-1 для ячеек паза, иначе -1
    params: OutrunnerPMSMParams

    def mask(self, region: Region) -> np.ndarray:
        return self.region == int(region)

    def region_areas(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for code, name in REGION_NAMES.items():
            cells = np.where(self.region == code)[0]
            out[name] = float(sum(self.mesh.cell_area(int(c)) for c in cells))
        return out


def _nearest_center_dist(theta: float, pitch: float) -> float:
    """Угловое расстояние до ближайшего центра решётки с шагом pitch (центры в k·pitch)."""
    d = theta % pitch
    if d > pitch / 2:
        d -= pitch
    return abs(d)


def _classify(cx: float, cy: float, p: OutrunnerPMSMParams,
              slot_pitch: float, tooth_ang: float,
              pole_pitch: float, mag_ang: float) -> tuple[int, float]:
    """Регион и знак полярности магнита по центроиду ячейки."""
    r = math.hypot(cx, cy)
    th = math.atan2(cy, cx) % (2.0 * math.pi)
    if r < p.R_sy:
        return int(Region.STATOR_YOKE), 0.0
    if r < p.R_s_out:
        d = _nearest_center_dist(th, slot_pitch)
        return (int(Region.TOOTH) if d <= tooth_ang / 2 else int(Region.SLOT)), 0.0
    if r < p.R_mag_in:
        return int(Region.AIR_GAP), 0.0
    if r < p.R_mag_out:
        d = _nearest_center_dist(th, pole_pitch)
        if d <= mag_ang / 2:
            k = int(round(th / pole_pitch)) % p.n_poles
            return int(Region.MAGNET), (1.0 if k % 2 == 0 else -1.0)
        return int(Region.AIR_GAP), 0.0
    return int(Region.ROTOR_YOKE), 0.0


def build_outrunner_spm_pmsm(params: OutrunnerPMSMParams) -> MachineGeometry:
    """Построить сечение outrunner SPM PMSM через gmsh → TriangleMesh + теги + ось магнита."""
    import gmsh

    params.validate()
    p = params
    slot_pitch = 2.0 * math.pi / p.n_slots
    tooth_ang = p.tooth_width_frac * slot_pitch
    pole_pitch = 2.0 * math.pi / p.n_poles
    mag_ang = p.magnet_embrace * pole_pitch
    size = p.mesh_size if p.mesh_size is not None else p.air_gap

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ

        def ring(ra: float, rb: float) -> int:
            d_out = occ.addDisk(0, 0, 0, rb, rb)
            d_in = occ.addDisk(0, 0, 0, ra, ra)
            out, _ = occ.cut([(2, d_out)], [(2, d_in)])
            return out[0][1]

        def sector(ra: float, rb: float, a1: float, a2: float) -> int:
            o = occ.addPoint(0, 0, 0)
            a = occ.addPoint(ra * math.cos(a1), ra * math.sin(a1), 0)
            b = occ.addPoint(rb * math.cos(a1), rb * math.sin(a1), 0)
            c = occ.addPoint(rb * math.cos(a2), rb * math.sin(a2), 0)
            d = occ.addPoint(ra * math.cos(a2), ra * math.sin(a2), 0)
            loop = occ.addCurveLoop([
                occ.addLine(a, b),
                occ.addCircleArc(b, o, c),
                occ.addLine(c, d),
                occ.addCircleArc(d, o, a),
            ])
            return occ.addPlaneSurface([loop])

        surfs: list[int] = [ring(p.R_bore, p.R_sy)]                 # ярмо статора
        for k in range(p.n_slots):                                  # зубья + пазы
            c = k * slot_pitch
            surfs.append(sector(p.R_sy, p.R_s_out, c - tooth_ang / 2, c + tooth_ang / 2))
            surfs.append(sector(p.R_sy, p.R_s_out, c + tooth_ang / 2, c + slot_pitch - tooth_ang / 2))
        surfs.append(ring(p.R_s_out, p.R_mag_in))                   # зазор
        for k in range(p.n_poles):                                  # магниты + межполюс. воздух
            c = k * pole_pitch
            surfs.append(sector(p.R_mag_in, p.R_mag_out, c - mag_ang / 2, c + mag_ang / 2))
            surfs.append(sector(p.R_mag_in, p.R_mag_out, c + mag_ang / 2, c + pole_pitch - mag_ang / 2))
        surfs.append(ring(p.R_mag_out, p.R_out))                    # ярмо ротора

        dt = [(2, s) for s in surfs]
        occ.fragment(dt, dt)                                        # конформность (общие границы)
        occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)[:, :2]
        tag2idx = {int(t): i for i, t in enumerate(node_tags)}

        tri_type = 2  # 3-узловой треугольник
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tris = None
        for et, en in zip(etypes, enodes):
            if et == tri_type:
                tris = np.asarray(en, dtype=int).reshape(-1, 3)
                break
        if tris is None:
            raise RuntimeError("gmsh не вернул треугольных элементов.")
        cells = np.vectorize(tag2idx.get)(tris)
    finally:
        gmsh.finalize()

    verts = np.ascontiguousarray(coords)
    # Ориентация CCW (наш TriangleMesh требует положительной площади).
    fixed = []
    for tri in cells:
        if signed_area2(verts[tri]) < 0.0:
            tri = tri[[0, 2, 1]]
        fixed.append(tri)
    cells = np.asarray(fixed, dtype=int)
    # Убрать orphan-узлы: gmsh возвращает узлы вне треугольной сетки (точки/кривые OCC) —
    # они дают нулевые строки в матрице жёсткости ⇒ вырожденная система. Оставляем только
    # узлы, входящие хотя бы в один треугольник, и перенумеровываем ячейки.
    used = np.unique(cells.reshape(-1))
    if used.size != verts.shape[0]:
        remap = np.full(verts.shape[0], -1, dtype=int)
        remap[used] = np.arange(used.size)
        verts = np.ascontiguousarray(verts[used])
        cells = remap[cells]
    mesh = TriangleMesh(vertices=verts, cells=cells)

    region = np.empty(mesh.n_cells, dtype=int)
    axis = np.zeros((mesh.n_cells, 2), dtype=float)
    slot_id = np.full(mesh.n_cells, -1, dtype=int)
    for c in range(mesh.n_cells):
        cx, cy = mesh.cell_centroid(c)
        th = math.atan2(cy, cx) % (2.0 * math.pi)
        code, sign = _classify(cx, cy, p, slot_pitch, tooth_ang, pole_pitch, mag_ang)
        region[c] = code
        if code == int(Region.MAGNET):
            axis[c] = sign * np.array([math.cos(th), math.sin(th)])
        elif code == int(Region.SLOT):
            slot_id[c] = int(th // slot_pitch) % p.n_slots
    return MachineGeometry(
        mesh=mesh, region=region, magnet_easy_axis=axis, slot_id=slot_id, params=p
    )
