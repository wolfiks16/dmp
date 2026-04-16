from __future__ import annotations

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.bem.regular_single_layer import (
    build_regular_pair_mask,
    assemble_single_layer_p0p0_regular,
)


def make_separated_two_triangle_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # tri 0
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 5.0],   # tri 1
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def make_touching_two_triangle_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_regular_pair_mask_for_separated_triangles():
    mesh = make_separated_two_triangle_mesh()
    mask = build_regular_pair_mask(mesh, (0, 1), near_factor=0.5)

    assert (0, 1) in mask.regular_pairs
    assert (1, 0) in mask.regular_pairs
    assert mask.n_regular >= 2


def test_regular_pair_mask_marks_self_pairs_as_singular():
    mesh = make_separated_two_triangle_mesh()
    mask = build_regular_pair_mask(mesh, (0, 1), near_factor=0.5)

    assert (0, 0) in mask.singular_pairs
    assert (1, 1) in mask.singular_pairs


def test_assemble_single_layer_p0p0_regular_shape():
    mesh = make_separated_two_triangle_mesh()
    mat, mask = assemble_single_layer_p0p0_regular(
        mesh=mesh,
        face_indices=(0, 1),
        near_factor=0.5,
        strict=False,
    )

    assert mat.shape == (2, 2)


def test_regular_matrix_is_symmetric_on_separated_mesh():
    mesh = make_separated_two_triangle_mesh()
    mat, mask = assemble_single_layer_p0p0_regular(
        mesh=mesh,
        face_indices=(0, 1),
        near_factor=0.5,
        strict=False,
    )

    assert np.isclose(mat[0, 1], mat[1, 0])


def test_regular_matrix_has_nan_on_unsupported_pairs_in_non_strict_mode():
    mesh = make_separated_two_triangle_mesh()
    mat, mask = assemble_single_layer_p0p0_regular(
        mesh=mesh,
        face_indices=(0, 1),
        near_factor=0.5,
        strict=False,
    )

    assert np.isnan(mat[0, 0])
    assert np.isnan(mat[1, 1])


def test_strict_mode_raises_when_unsupported_pairs_exist():
    mesh = make_separated_two_triangle_mesh()

    try:
        assemble_single_layer_p0p0_regular(
            mesh=mesh,
            face_indices=(0, 1),
            near_factor=0.5,
            strict=True,
        )
    except NotImplementedError:
        assert True
        return

    assert False, "Expected NotImplementedError in strict mode."


def test_touching_mesh_is_not_regular_only():
    mesh = make_touching_two_triangle_mesh()
    mask = build_regular_pair_mask(mesh, (0, 1), near_factor=0.5)

    assert mask.n_singular > 0 or mask.n_near > 0