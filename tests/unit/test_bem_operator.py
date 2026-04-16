from __future__ import annotations

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.bem.spaces import FaceP0Space, VertexP1Space
from magcore.bem.operators import (
    OperatorKind,
    SingleLayerOperator,
    DoubleLayerOperator,
    AdjointDoubleLayerOperator,
    HypersingularOperator,
)


def make_spaces():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    mesh = SurfaceMesh(vertices=vertices, faces=faces)

    phi = VertexP1Space.from_faces(mesh, (0,))
    flux = FaceP0Space.from_faces(mesh, (0,))
    return phi, flux


def test_single_layer_operator_shape():
    phi, flux = make_spaces()
    op = SingleLayerOperator(
        kind=OperatorKind.SINGLE_LAYER,
        domain_space=flux,
        range_space=phi,
        dual_space=None,
        label="V",
    )

    assert op.shape == (phi.ndofs, flux.ndofs)
    assert op.kind == OperatorKind.SINGLE_LAYER
    assert op.label == "V"


def test_double_layer_operator_shape():
    phi, flux = make_spaces()
    op = DoubleLayerOperator(
        kind=OperatorKind.DOUBLE_LAYER,
        domain_space=phi,
        range_space=phi,
        dual_space=None,
        label="K",
    )

    assert op.shape == (phi.ndofs, phi.ndofs)
    assert op.kind == OperatorKind.DOUBLE_LAYER


def test_adjoint_double_layer_operator_shape():
    phi, flux = make_spaces()
    op = AdjointDoubleLayerOperator(
        kind=OperatorKind.ADJOINT_DOUBLE_LAYER,
        domain_space=flux,
        range_space=flux,
        dual_space=None,
        label="K'",
    )

    assert op.shape == (flux.ndofs, flux.ndofs)
    assert op.kind == OperatorKind.ADJOINT_DOUBLE_LAYER


def test_hypersingular_operator_shape():
    phi, flux = make_spaces()
    op = HypersingularOperator(
        kind=OperatorKind.HYPERSINGULAR,
        domain_space=phi,
        range_space=flux,
        dual_space=None,
        label="D",
    )

    assert op.shape == (flux.ndofs, phi.ndofs)
    assert op.kind == OperatorKind.HYPERSINGULAR