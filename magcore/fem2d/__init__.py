"""
Планарный (2D) магнитостатический backend: скалярный вектор-потенциал A_z на
P1-треугольниках. Формулировка −div(ν∇A_z)=J_z+curl₂(νB_r); B=(∂_yA_z,−∂_xA_z).

Разделяет размерно-независимое физическое ядро (materials, demag, nonlinear/Picard)
с 3D-backend'ом; экстерьер замыкается Kelvin-трансформацией (не BEM).
См. docs/novelty/pivot_2D_thermal_SmCo_2026-07-07.md §1.6, §2.1.
"""
from magcore.fem2d.mesh import TriangleMesh
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.spaces import LagrangeP1Space2D

__all__ = [
    "TriangleMesh",
    "build_structured_rectangle_tri_mesh",
    "LagrangeP1Space2D",
]
