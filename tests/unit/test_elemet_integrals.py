from __future__ import annotations

import numpy as np

from magcore.bem.quadrature import get_triangle_quadrature
from magcore.bem.element_integrals import (
    FacePairClass,
    triangle_area,
    triangle_centroid,
    triangle_diameter,
    map_reference_triangle_to_physical,
    single_layer_point_potential_regular,
    single_layer_face_face_regular,
    classify_face_pair,
)
from magcore.mesh.surface_mesh import SurfaceMesh


def make_triangle() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def make_mesh_for_classification() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # tri 0
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0],   # tri 1
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [0.2, 0.2, 0.0],   # tri 2 near tri 0
            [1.2, 0.2, 0.0],
            [0.2, 1.2, 0.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_triangle_area_is_correct():
    tri = make_triangle()
    assert np.isclose(triangle_area(tri), 1.0)


def test_triangle_centroid_is_correct():
    tri = make_triangle()
    c = triangle_centroid(tri)

    assert np.allclose(c, [2.0 / 3.0, 1.0 / 3.0, 0.0])


def test_triangle_diameter_is_correct():
    tri = make_triangle()
    d = triangle_diameter(tri)

    assert np.isclose(d, np.sqrt(5.0))


def test_reference_triangle_mapping_vertex():
    tri = make_triangle()
    p = map_reference_triangle_to_physical(tri, np.array([0.0, 0.0], dtype=float))

    assert np.allclose(p, tri[0])


def test_reference_triangle_mapping_other_vertex():
    tri = make_triangle()
    p = map_reference_triangle_to_physical(tri, np.array([1.0, 0.0], dtype=float))

    assert np.allclose(p, tri[1])


def test_single_layer_point_potential_regular_is_positive():
    tri = make_triangle()
    x = np.array([0.0, 0.0, 5.0], dtype=float)
    q = get_triangle_quadrature(order=2)

    val = single_layer_point_potential_regular(x, tri, q)
    assert val > 0.0


def test_single_layer_face_face_regular_is_positive_for_separated_triangles():
    tri_1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    tri_2 = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ],
        dtype=float,
    )
    q = get_triangle_quadrature(order=2)

    val = single_layer_face_face_regular(
        target_tri=tri_1,
        source_tri=tri_2,
        target_quadrature=q,
        source_quadrature=q,
    )
    assert val > 0.0


def test_single_layer_face_face_regular_is_symmetric_for_separated_triangles():
    tri_1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    tri_2 = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ],
        dtype=float,
    )
    q = get_triangle_quadrature(order=2)

    v12 = single_layer_face_face_regular(
        target_tri=tri_1,
        source_tri=tri_2,
        target_quadrature=q,
        source_quadrature=q,
    )
    v21 = single_layer_face_face_regular(
        target_tri=tri_2,
        source_tri=tri_1,
        target_quadrature=q,
        source_quadrature=q,
    )

    assert np.isclose(v12, v21)


def test_classify_face_pair_regular():
    mesh = make_mesh_for_classification()
    cls = classify_face_pair(mesh, 0, 1, near_factor=1.0)

    assert cls == FacePairClass.REGULAR


def test_classify_face_pair_near():
    mesh = make_mesh_for_classification()
    cls = classify_face_pair(mesh, 0, 2, near_factor=10.0)

    assert cls == FacePairClass.NEAR


def test_classify_face_pair_singular_same_face():
    mesh = make_mesh_for_classification()
    cls = classify_face_pair(mesh, 0, 0, near_factor=1.0)

    assert cls == FacePairClass.SINGULAR