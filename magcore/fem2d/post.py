from __future__ import annotations

import numpy as np

from magcore.fem2d.mesh import p1_gradients, triangle_area
from magcore.fem2d.quadrature import triangle_quadrature
from magcore.fem2d.spaces import LagrangeP1Space2D


def reconstruct_B_on_cells(space: LagrangeP1Space2D, a: np.ndarray) -> np.ndarray:
    """
    Поячеечное поле B = (∂_y A_z, −∂_x A_z) из узловых коэффициентов A_z (P1).
    ∇A_z постоянен по ячейке ⇒ B тоже. Возвращает (n_cells, 2).
    """
    mesh = space.mesh
    a = np.asarray(a, dtype=float)
    B = np.empty((mesh.n_cells, 2), dtype=float)
    for c in range(mesh.n_cells):
        grads = p1_gradients(mesh.cell_vertices(c))       # (3,2)
        idx = mesh.cell_vertex_indices(c)
        grad_az = a[list(idx)] @ grads                    # (2,) = ∇A_z
        B[c, 0] = grad_az[1]
        B[c, 1] = -grad_az[0]
    return B


def l2_error(
    space: LagrangeP1Space2D, a: np.ndarray, exact_fn, *, quadrature_order: int = 5
) -> float:
    """‖A_h − A_exact‖_{L²(Ω)} численной квадратурой."""
    mesh = space.mesh
    a = np.asarray(a, dtype=float)
    bary, w = triangle_quadrature(quadrature_order)
    acc = 0.0
    for c in range(mesh.n_cells):
        verts = mesh.cell_vertices(c)
        area = triangle_area(verts)
        idx = mesh.cell_vertex_indices(c)
        av = a[list(idx)]
        for q in range(bary.shape[0]):
            lam = bary[q]
            x = lam @ verts
            ah = float(lam @ av)
            diff = ah - float(exact_fn(x))
            acc += w[q] * area * diff * diff
    return float(np.sqrt(acc))


def h1_seminorm_error(
    space: LagrangeP1Space2D, a: np.ndarray, grad_exact_fn, *, quadrature_order: int = 5
) -> float:
    """|A_h − A_exact|_{H¹(Ω)} = ‖∇A_h − ∇A_exact‖_{L²}. ∇A_h постоянен по ячейке."""
    mesh = space.mesh
    a = np.asarray(a, dtype=float)
    bary, w = triangle_quadrature(quadrature_order)
    acc = 0.0
    for c in range(mesh.n_cells):
        verts = mesh.cell_vertices(c)
        area = triangle_area(verts)
        grads = p1_gradients(verts)
        idx = mesh.cell_vertex_indices(c)
        grad_ah = a[list(idx)] @ grads                    # (2,)
        for q in range(bary.shape[0]):
            lam = bary[q]
            x = lam @ verts
            ge = np.asarray(grad_exact_fn(x), dtype=float)
            d = grad_ah - ge
            acc += w[q] * area * float(d @ d)
    return float(np.sqrt(acc))
