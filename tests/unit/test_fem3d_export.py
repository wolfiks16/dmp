import re

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.fem2d.model.materials import Air, MagnetMaterial
from magcore.fem3d import GeoObject3D, auto_domain3d, build_object_problem3d, solve_nonlinear3d
from magcore.fem3d.export import write_vtu
from magcore.fem3d.scene import cell_quantities

# Выгрузка в ParaView (.vtu): файл читается обратно по спецификации VTK XML — двоичный хвост
# AppendedData, перед каждым массивом заголовок UInt64 с числом байт, смещения от байта после «_» —
# и даёт ровно те же массивы, что в решении.

_DT = {"Float64": "<f8", "Int64": "<i8", "Int32": "<i4", "UInt8": "<u1"}
_ARRAY = re.compile(r'<DataArray type="(\w+)"(?: Name="([^"]+)")?(?: NumberOfComponents="(\d+)")?'
                    r' format="appended" offset="(\d+)"/>')


def _read_vtu(raw: bytes):
    head, tail = raw.split(b'<AppendedData encoding="raw">', 1)
    data = tail[tail.index(b"_") + 1:]
    xml = head.decode("ascii")
    arrays = {}
    for m in _ARRAY.finditer(xml):
        vtype, name, ncomp, off = m.group(1), m.group(2), int(m.group(3) or 1), int(m.group(4))
        nbytes = int(np.frombuffer(data[off:off + 8], dtype="<u8")[0])
        a = np.frombuffer(data[off + 8:off + 8 + nbytes], dtype=_DT[vtype])
        arrays[name] = a.reshape(-1, ncomp) if ncomp > 1 else a
    piece = re.search(r'NumberOfPoints="(\d+)" NumberOfCells="(\d+)"', xml)
    return xml, arrays, int(piece.group(1)), int(piece.group(2))


def test_vtu_file_reads_back_as_the_solution(tmp_path):
    pytest.importorskip("gmsh")
    mag = MagnetMaterial(n42sh_magnet((0.0, 0.0, 1.0)))
    cube = GeoObject3D("m", "box", {"lx": 10e-3, "ly": 10e-3, "lz": 10e-3}, mag, magnet_dir="axial",
                       mesh_size=2.5e-3)
    p = build_object_problem3d([cube], auto_domain3d([cube], material=Air(), margin_frac=1.0),
                               default_mesh_size=2.5e-3, T=155.0)
    f = solve_nonlinear3d(p)
    raw = write_vtu(tmp_path / "m.vtu", f).read_bytes()
    assert raw.endswith(b"</AppendedData>\n</VTKFile>\n")
    xml, a, npts, ncells = _read_vtu(raw)
    assert 'type="UnstructuredGrid"' in xml and 'header_type="UInt64"' in xml
    assert (npts, ncells) == (p.mesh.n_vertices, p.mesh.n_cells)
    assert np.array_equal(a["Points"], p.mesh.vertices)
    assert np.array_equal(a["connectivity"].reshape(-1, 4), p.mesh.cells)
    assert np.array_equal(a["offsets"], 4 * np.arange(1, ncells + 1))
    assert np.all(a["types"] == 10)                                              # тетраэдр VTK
    assert np.array_equal(a["region"], np.asarray(p.cell_region))
    assert np.array_equal(a["B_T"], f.B_cells) and np.array_equal(a["H_A_m"], f.H_cells)
    q = cell_quantities(f)
    for key, name in (("B", "B_abs_T"), ("margin", "margin_kA_m"), ("loss", "loss_T")):
        assert np.array_equal(a[name], q[key][0], equal_nan=True), name
