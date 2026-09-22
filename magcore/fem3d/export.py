from __future__ import annotations

from pathlib import Path

import numpy as np

from magcore.fem3d.scalar import ScalarField3D
from magcore.fem3d.scene import cell_quantities

# ВЫГРУЗКА В ParaView (этап 3D-5): VTK XML UnstructuredGrid (.vtu). Массивы — двоичные, в конце
# файла (AppendedData, encoding="raw"): перед каждым заголовок UInt64 с числом байт, смещения
# отсчитываются от первого байта после «_». Ячейки — тетраэдры (тип VTK 10). По ячейкам: регион,
# B и H векторами и скалярные величины объёмного вида (у величин магнита вне магнитов — NaN).

_VTK_TETRA = 10
_SCALAR_NAMES = {"B": "B_abs_T", "H": "H_abs_kA_m", "mu": "mu_r",
                 "Hpar": "Hpar_kA_m", "margin": "margin_kA_m", "loss": "loss_T"}


def write_vtu(path, field: ScalarField3D) -> Path:
    """Записать решение в .vtu для ParaView; возвращает путь."""
    mesh = field.problem.mesh
    nc = mesh.n_cells
    blocks: list[bytes] = []
    offset = 0

    def data_array(arr, vtk_type: str, name: str | None = None, ncomp: int = 1) -> str:
        nonlocal offset
        raw = np.ascontiguousarray(arr).tobytes()
        at = offset
        blocks.append(np.array([len(raw)], dtype="<u8").tobytes() + raw)
        offset += 8 + len(raw)
        nm = f' Name="{name}"' if name else ""
        nco = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
        return f'<DataArray type="{vtk_type}"{nm}{nco} format="appended" offset="{at}"/>'

    points = data_array(mesh.vertices.astype("<f8"), "Float64", "Points", 3)
    conn = data_array(mesh.cells.astype("<i8").ravel(), "Int64", "connectivity")
    offs = data_array(np.arange(1, nc + 1, dtype="<i8") * 4, "Int64", "offsets")
    types = data_array(np.full(nc, _VTK_TETRA, dtype="<u1"), "UInt8", "types")
    cell_data = [data_array(np.asarray(field.problem.cell_region, dtype="<i4"), "Int32", "region"),
                 data_array(field.B_cells.astype("<f8"), "Float64", "B_T", 3),
                 data_array(field.H_cells.astype("<f8"), "Float64", "H_A_m", 3)]
    for key, (vals, _unit) in cell_quantities(field).items():
        if key in _SCALAR_NAMES:
            cell_data.append(data_array(np.asarray(vals, dtype="<f8"), "Float64", _SCALAR_NAMES[key]))
    head = ('<?xml version="1.0"?>\n'
            '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
            '<UnstructuredGrid>\n'
            f'<Piece NumberOfPoints="{mesh.n_vertices}" NumberOfCells="{nc}">\n'
            f'<Points>\n{points}\n</Points>\n'
            f'<Cells>\n{conn}\n{offs}\n{types}\n</Cells>\n'
            '<CellData Scalars="region">\n' + "\n".join(cell_data) + '\n</CellData>\n'
            '</Piece>\n</UnstructuredGrid>\n<AppendedData encoding="raw">\n_')
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii"))
        for b in blocks:
            fh.write(b)
        fh.write(b"\n</AppendedData>\n</VTKFile>\n")
    return path
