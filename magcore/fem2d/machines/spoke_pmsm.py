from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from magcore.fem2d.machines.pmsm_outrunner import (
    REGION_NAMES,
    MachineGeometry,
    Region,
)
from magcore.mesh.gmsh_session import close_gmsh, open_gmsh
from magcore.fem2d.mesh import TriangleMesh, signed_area2

# НОВЫЙ (реальный) генератор сечения OUTRUNNER PMSM по инженерной схеме заказчика (КОМПАС):
# СПИЦЕВОЙ статор («лучи» с полюсными башмаками-«топориками») + вращающаяся ОБЕЧАЙКА с
# магнитами. Топология по радиусу изнутри наружу:
#   вал · ЯРМО-ОСНОВАНИЕ · стержни ЛУЧЕЙ + ТОПОРИКИ (зубья) / ПАЗЫ · ЗАЗОР · МАГНИТЫ · ОБЕЧАЙКА.
#
# Отличие от прежнего pmsm_outrunner (трапецеидальные зубья + один «охват»): здесь зуб = луч
# с тонким стержнем постоянной толщины и широким башмаком, а магнит имеет ВЫБОР ФОРМЫ (сектор /
# усечённый сектор / призма) — как в исходной параметризации. Выдаёт тот же `MachineGeometry`
# (те же теги регионов + ось лёгкого намагничивания + номер паза), поэтому вся готовая физика
# (S1/S2/S3, момент, потери, прогонка ротора) подключается БЕЗ изменений.
#
# Параметры пользователь задаёт в ДИАМЕТРАХ (мм) и градусах — как в его чертеже; внутри всё
# переводится в радиусы в МЕТРАХ (СИ, консистентно с остальным ядром).


class MagnetShape(str, Enum):
    SECTOR = "sector"                     # дуговой сектор (радиальные стороны, дуги сверху/снизу)
    TRUNCATED_SECTOR = "truncated_sector" # усечённый: дуги сверху/снизу, ПРЯМЫЕ стороны (ширина)
    PRISM = "prism"                       # призма: прямоугольный блок у обечайки (ширина × толщина)


_MM = 1.0e-3
_DEG = math.pi / 180.0


@dataclass(frozen=True)
class SpokeMotorParams:
    """
    Геометрия спицевого outrunner PMSM в терминах чертежа заказчика.

    Диаметры — в ММ, углы — в ГРАДУСАХ, счётные — целые. Внутренние свойства (R_*) переводят
    их в РАДИУСЫ в МЕТРАХ. Имена полей повторяют смысл параметров таблицы (в комментариях —
    исходное название).
    """

    n_poles: int = 14                     # Магниты
    n_teeth: int = 12                     # Лучи
    stack_length_mm: float = 20.0         # Длина магнита (= осевая длина пакета; в 2D — множитель)

    D_shell_out_mm: float = 50.5          # Внешний диаметр обечайки
    D_shell_in_mm: float = 45.4           # Внутренний диаметр обечайки (= внешний магнита)
    D_magnet_in_mm: float = 41.26         # Внутренний диаметр магнита
    D_tooth_out_mm: float = 40.86         # Внешний диаметр лучей (верх топорика)
    D_shoe_in_mm: float = 38.5            # Внутренний диаметр луча (низ топорика)
    D_base_out_mm: float = 19.5           # Внешний диаметр основания луча
    D_base_in_mm: float = 17.0            # Внутренний диаметр основания луча (расточка)

    tooth_stem_mm: float = 2.5            # Толщина луча (стержня)
    shoe_width_mm: float = 8.55           # Ширина топорика (дуговая ширина башмака)

    magnet_shape: MagnetShape = MagnetShape.TRUNCATED_SECTOR
    sector_angle_deg: float = 25.5        # Угол сектора
    truncated_width_mm: float = 8.0       # Ширина усечённого сектора
    prism_width_mm: float = 8.0           # Ширина призмы
    prism_thickness_mm: float = 1.5       # Толщина призмы

    rotor_angle: float = 0.0              # механический поворот ротора [рад] (магниты + обечайка)
    mesh_size_mm: float | None = None     # глобальный размер элемента (по умолч. ~ зазор)
    mesh_size_by_region: dict = field(default_factory=dict)  # имя региона → размер (мм)

    # --- радиусы в метрах ---
    @property
    def R_bore(self) -> float:
        return 0.5 * self.D_base_in_mm * _MM

    @property
    def R_sy(self) -> float:              # внешний радиус ярма-основания (корень стержня)
        return 0.5 * self.D_base_out_mm * _MM

    @property
    def R_shoe_in(self) -> float:         # низ топорика (переход стержень→башмак)
        return 0.5 * self.D_shoe_in_mm * _MM

    @property
    def R_s_out(self) -> float:           # верх топорика = внешняя поверхность статора
        return 0.5 * self.D_tooth_out_mm * _MM

    @property
    def R_mag_in(self) -> float:
        return 0.5 * self.D_magnet_in_mm * _MM

    @property
    def R_mag_out(self) -> float:         # = внутренняя поверхность обечайки
        return 0.5 * self.D_shell_in_mm * _MM

    @property
    def R_out(self) -> float:             # внешний радиус ротора (обечайки)
        return 0.5 * self.D_shell_out_mm * _MM

    @property
    def axial_length(self) -> float:
        return self.stack_length_mm * _MM

    # --- совместимость с прежним API (physics читает params.n_slots) ---
    @property
    def n_slots(self) -> int:
        return self.n_teeth

    def validate(self) -> None:
        if self.n_teeth < 3 or self.n_poles < 2 or self.n_poles % 2 != 0:
            raise ValueError("n_teeth>=3 и n_poles>=2 чётное.")
        rs = [self.R_bore, self.R_sy, self.R_shoe_in, self.R_s_out,
              self.R_mag_in, self.R_mag_out, self.R_out]
        if any(np.diff(rs) <= 0.0):
            raise ValueError("радиусы должны строго возрастать: расточка < основание < низ "
                             "топорика < верх лучей < внутр. магнита < обечайка внутр < внешн.")
        stem_half = 0.5 * self.tooth_stem_mm * _MM
        if not (0.0 < stem_half < self.R_sy):
            raise ValueError("толщина луча вне допустимого.")
        shoe_half = 0.5 * self.shoe_width_mm * _MM
        if shoe_half >= self.R_shoe_in * math.sin(math.pi / self.n_teeth):
            raise ValueError("ширина топорика больше зубцового шага — башмаки перекрываются.")
        if self.magnet_shape == MagnetShape.SECTOR:
            if not (0.0 < self.sector_angle_deg < 360.0 / self.n_poles):
                raise ValueError("угол сектора должен быть в (0, полюсный шаг).")
        elif self.magnet_shape == MagnetShape.PRISM:
            if self.prism_thickness_mm * _MM >= (self.R_mag_out - self.R_mag_in):
                raise ValueError("толщина призмы больше радиального зазора под магнит.")


def _rotate(x: float, y: float, phi: float) -> tuple[float, float]:
    """Повернуть точку в ЛОКАЛЬНУЮ систему полюса/зуба (ось объекта → +x)."""
    c, s = math.cos(-phi), math.sin(-phi)
    return x * c - y * s, x * s + y * c


def _in_magnet(xr: float, yr: float, r: float, p: SpokeMotorParams) -> bool:
    """Принадлежит ли точка магниту (в ЛОКАЛЬНОЙ системе полюса: xr вдоль оси, yr поперёк)."""
    if p.magnet_shape == MagnetShape.SECTOR:
        if not (p.R_mag_in <= r <= p.R_mag_out):
            return False
        return abs(math.atan2(yr, xr)) <= 0.5 * p.sector_angle_deg * _DEG
    if p.magnet_shape == MagnetShape.TRUNCATED_SECTOR:
        if not (p.R_mag_in <= r <= p.R_mag_out):
            return False
        return abs(yr) <= 0.5 * p.truncated_width_mm * _MM         # прямые стороны
    # PRISM — прямоугольный блок у обечайки
    return (abs(yr) <= 0.5 * p.prism_width_mm * _MM
            and (p.R_mag_out - p.prism_thickness_mm * _MM) <= xr <= p.R_mag_out)


def _classify(x: float, y: float, p: SpokeMotorParams,
              tooth_pitch: float, stem_half: float, shoe_half: float,
              pole_pitch: float) -> tuple[int, float]:
    """Регион и знак полярности магнита по точке (центроиду ячейки)."""
    r = math.hypot(x, y)
    if r < p.R_sy:
        return int(Region.STATOR_YOKE), 0.0
    if r < p.R_s_out:
        # ближайшая ось зуба
        th = math.atan2(y, x)
        k = round(th / tooth_pitch)
        xr, yr = _rotate(x, y, k * tooth_pitch)
        if r < p.R_shoe_in:                          # стержень: постоянная толщина
            return (int(Region.TOOTH) if abs(yr) <= stem_half else int(Region.SLOT)), 0.0
        return (int(Region.TOOTH) if abs(yr) <= shoe_half else int(Region.SLOT)), 0.0
    if r < p.R_mag_in:
        return int(Region.AIR_GAP), 0.0
    if r < p.R_mag_out:
        th = math.atan2(y, x)
        j = round((th - p.rotor_angle) / pole_pitch)
        phi = p.rotor_angle + j * pole_pitch
        xr, yr = _rotate(x, y, phi)
        if _in_magnet(xr, yr, r, p):
            return int(Region.MAGNET), (1.0 if j % 2 == 0 else -1.0)
        return int(Region.AIR_GAP), 0.0              # межполюсный воздух
    return int(Region.ROTOR_YOKE), 0.0


def build_spoke_pmsm(params: SpokeMotorParams) -> MachineGeometry:
    """
    Построить сечение спицевого outrunner PMSM → TriangleMesh + теги регионов + ось магнита.

    Регион ставится ПО ПРИНАДЛЕЖНОСТИ ЯЧЕЙКИ К ПОСТРОЕННОЙ ПОВЕРХНОСТИ (провенанс через
    gmsh.fragment), а не по центроиду ⇒ сетка ТОЧНО КОНФОРМНА границам всех регионов, тег
    точный до ребра. На перекрытиях (фон-кольцо под зубом/магнитом) побеждает более
    приоритетная поверхность: магнит/зуб (2) > ярмо/зазор/обечайка (1) > фон-кольцо (0).
    """
    import gmsh

    params.validate()
    p = params
    tooth_pitch = 2.0 * math.pi / p.n_teeth
    pole_pitch = 2.0 * math.pi / p.n_poles
    stem_half = 0.5 * p.tooth_stem_mm * _MM
    shoe_half_ang = (0.5 * p.shoe_width_mm * _MM) / p.R_shoe_in     # дуговая полуширина башмака
    default_size = p.mesh_size_mm * _MM if p.mesh_size_mm else (p.R_mag_in - p.R_s_out)

    open_gmsh()                                   # сеанс gmsh — под общим замком процесса
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
            loop = occ.addCurveLoop([occ.addLine(a, b), occ.addCircleArc(b, o, c),
                                     occ.addLine(c, d), occ.addCircleArc(d, o, a)])
            return occ.addPlaneSurface([loop])

        def rect(r0: float, r1: float, half_w: float, phi: float) -> int:
            """ПРЯМОУГОЛЬНИК (плоские торцы): радиально [r0,r1] × поперёк, повёрнут на phi."""
            s = occ.addRectangle(r0, -half_w, 0, r1 - r0, 2.0 * half_w)
            occ.rotate([(2, s)], 0, 0, 0, 0, 0, 1, phi)
            return s

        def bar(r0: float, r1: float, half_w: float, phi: float) -> int:
            """
            ДУГИ сверху/снизу (радиусы r0,r1) + ПРЯМЫЕ боковые стороны (±half_w) — «усечённый
            сектор»/булка. Магнит не залезает в зазор (внутр. грань — дуга по R_mag_in).
            """
            xi, xo = math.sqrt(r0 * r0 - half_w * half_w), math.sqrt(r1 * r1 - half_w * half_w)
            def rp(x, y):
                return occ.addPoint(x * math.cos(phi) - y * math.sin(phi),
                                    x * math.sin(phi) + y * math.cos(phi), 0)
            o = occ.addPoint(0, 0, 0)
            a, b = rp(xi, -half_w), rp(xo, -half_w)
            c, d = rp(xo, half_w), rp(xi, half_w)
            loop = occ.addCurveLoop([occ.addLine(a, b), occ.addCircleArc(b, o, c),
                                     occ.addLine(c, d), occ.addCircleArc(d, o, a)])
            return occ.addPlaneSurface([loop])

        Y, T, S = int(Region.STATOR_YOKE), int(Region.TOOTH), int(Region.SLOT)
        G, M, R = int(Region.AIR_GAP), int(Region.MAGNET), int(Region.ROTOR_YOKE)
        surfs: list[int] = []
        info: list[tuple[int, int]] = []                           # (регион, приоритет)

        def add(tag: int, reg: int, prio: int):
            surfs.append(tag)
            info.append((reg, prio))

        add(ring(p.R_bore, p.R_sy), Y, 1)                          # ярмо-основание
        add(ring(p.R_sy, p.R_s_out), S, 0)                         # фон зубьев+пазов → паз
        for k in range(p.n_teeth):                                 # лучи: стержень + топорик
            phi = k * tooth_pitch
            add(rect(p.R_sy, p.R_shoe_in, stem_half, phi), T, 2)
            add(sector(p.R_shoe_in, p.R_s_out, phi - shoe_half_ang, phi + shoe_half_ang), T, 2)
        add(ring(p.R_s_out, p.R_mag_in), G, 1)                     # зазор
        add(ring(p.R_mag_in, p.R_mag_out), G, 0)                   # фон магнитов → межполюс. воздух
        for j in range(p.n_poles):                                 # магниты выбранной формы
            phi = j * pole_pitch + p.rotor_angle
            if p.magnet_shape == MagnetShape.SECTOR:
                a = 0.5 * p.sector_angle_deg * _DEG
                add(sector(p.R_mag_in, p.R_mag_out, phi - a, phi + a), M, 2)
            elif p.magnet_shape == MagnetShape.TRUNCATED_SECTOR:
                add(bar(p.R_mag_in, p.R_mag_out, 0.5 * p.truncated_width_mm * _MM, phi), M, 2)
            else:  # PRISM — прямоугольный блок у обечайки (плоские грани)
                add(rect(p.R_mag_out - p.prism_thickness_mm * _MM, p.R_mag_out,
                         0.5 * p.prism_width_mm * _MM, phi), M, 2)
        add(ring(p.R_mag_out, p.R_out), R, 1)                      # обечайка

        dt = [(2, s) for s in surfs]
        _, outmap = occ.fragment(dt, dt)                           # конформность + провенанс
        occ.synchronize()

        # выходная поверхность → (регион, приоритет), приоритетная поверхность побеждает
        out_reg: dict[int, tuple[int, int]] = {}
        for i in range(len(surfs)):
            reg, prio = info[i]
            for d, child in outmap[i]:
                if d != 2:
                    continue
                cur = out_reg.get(child)
                if cur is None or prio > cur[1]:
                    out_reg[child] = (reg, prio)

        if p.mesh_size_by_region:
            reg_sizes = {name: float(p.mesh_size_by_region.get(name, p.mesh_size_mm or 1.0)) * _MM
                         for name in REGION_NAMES.values()}

            def _size_cb(dim, tag, x, y, z, lc):
                code, _ = _classify(x, y, p, tooth_pitch, stem_half, shoe_half_ang, pole_pitch)
                return reg_sizes[REGION_NAMES[code]]

            gmsh.model.mesh.setSizeCallback(_size_cb)
            gmsh.option.setNumber("Mesh.MeshSizeMin", min(reg_sizes.values()) * 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", max(reg_sizes.values()))
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMin", default_size * 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", default_size)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)[:, :2]
        tag2idx = {int(t): i for i, t in enumerate(node_tags)}

        etypes, etags, enodes = gmsh.model.mesh.getElements(2)
        tris = etag_tris = None
        for et, tg, en in zip(etypes, etags, enodes):
            if et == 2:
                tris = np.asarray(en, dtype=int).reshape(-1, 3)
                etag_tris = np.asarray(tg, dtype=np.int64)
                break
        if tris is None:
            raise RuntimeError("gmsh не вернул треугольных элементов.")
        cells = np.vectorize(tag2idx.get)(tris)

        # тег региона по поверхности элемента (точный, конформный)
        tag_region: dict[int, int] = {}
        for child, (reg, _prio) in out_reg.items():
            st, stg, _ = gmsh.model.mesh.getElements(2, child)
            for typ, tags in zip(st, stg):
                if typ == 2:
                    for t in tags:
                        tag_region[int(t)] = reg
        cell_region = np.array([tag_region.get(int(t), int(Region.AIR_GAP))
                                for t in etag_tris], dtype=int)
    finally:
        close_gmsh()

    verts = np.ascontiguousarray(coords)
    fixed = [(tri[[0, 2, 1]] if signed_area2(verts[tri]) < 0.0 else tri) for tri in cells]
    cells = np.asarray(fixed, dtype=int)
    used = np.unique(cells.reshape(-1))
    if used.size != verts.shape[0]:
        remap = np.full(verts.shape[0], -1, dtype=int)
        remap[used] = np.arange(used.size)
        verts = np.ascontiguousarray(verts[used])
        cells = remap[cells]
    mesh = TriangleMesh(vertices=verts, cells=cells)

    region = cell_region
    axis = np.zeros((mesh.n_cells, 2), dtype=float)
    slot_id = np.full(mesh.n_cells, -1, dtype=int)
    for c in range(mesh.n_cells):
        cx, cy = mesh.cell_centroid(c)
        th = math.atan2(cy, cx) % (2.0 * math.pi)
        if region[c] == int(Region.MAGNET):
            j = round((math.atan2(cy, cx) - p.rotor_angle) / pole_pitch)
            sign = 1.0 if j % 2 == 0 else -1.0
            axis[c] = sign * np.array([math.cos(th), math.sin(th)])
        elif region[c] == int(Region.SLOT):
            slot_id[c] = int(th // tooth_pitch) % p.n_teeth
    return MachineGeometry(mesh=mesh, region=region, magnet_easy_axis=axis,
                           slot_id=slot_id, params=p)
