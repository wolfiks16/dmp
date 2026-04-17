from __future__ import annotations

import math
import numpy as np

from magcore.femcore.quadrature import get_tetra_quadrature


def _exact_reference_tetra_monomial_integral(px: int, py: int, pz: int) -> float:
    """
    Точный интеграл по эталонному тетраэдру:
        T = {(x,y,z): x>=0, y>=0, z>=0, x+y+z<=1}

    Для монома x^px y^py z^pz:
        ∫_T x^px y^py z^pz dV = px! py! pz! / (px + py + pz + 3)!
    """
    return (
        math.factorial(px)
        * math.factorial(py)
        * math.factorial(pz)
        / math.factorial(px + py + pz + 3)
    )


def _quadrature_monomial_integral(order: int, px: int, py: int, pz: int) -> float:
    q = get_tetra_quadrature(order=order)
    values = (
        q.points[:, 0] ** px
        * q.points[:, 1] ** py
        * q.points[:, 2] ** pz
    )
    return float(np.dot(q.weights, values))


def test_order_1_is_exact_for_monomials_up_to_degree_1() -> None:
    monomials = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]

    for px, py, pz in monomials:
        exact = _exact_reference_tetra_monomial_integral(px, py, pz)
        approx = _quadrature_monomial_integral(1, px, py, pz)
        assert np.isclose(approx, exact, atol=1e-14)


def test_order_2_is_exact_for_monomials_up_to_degree_2() -> None:
    monomials = [
        (0, 0, 0),
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 0, 0), (0, 2, 0), (0, 0, 2),
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
    ]

    for px, py, pz in monomials:
        exact = _exact_reference_tetra_monomial_integral(px, py, pz)
        approx = _quadrature_monomial_integral(2, px, py, pz)
        assert np.isclose(approx, exact, atol=1e-14)


def test_order_3_is_exact_for_monomials_up_to_degree_3() -> None:
    monomials = [
        (0, 0, 0),
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 0, 0), (0, 2, 0), (0, 0, 2),
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (3, 0, 0), (0, 3, 0), (0, 0, 3),
        (2, 1, 0), (2, 0, 1), (1, 2, 0),
        (0, 2, 1), (1, 0, 2), (0, 1, 2),
        (1, 1, 1),
    ]

    for px, py, pz in monomials:
        exact = _exact_reference_tetra_monomial_integral(px, py, pz)
        approx = _quadrature_monomial_integral(3, px, py, pz)
        assert np.isclose(approx, exact, atol=1e-14)


def test_order_2_is_not_exact_for_a_cubic_monomial() -> None:
    px, py, pz = (3, 0, 0)

    exact = _exact_reference_tetra_monomial_integral(px, py, pz)
    approx = _quadrature_monomial_integral(2, px, py, pz)

    assert not np.isclose(approx, exact, atol=1e-14)
    assert abs(approx - exact) > 1e-8


def test_order_3_fixes_the_cubic_monomial_case() -> None:
    px, py, pz = (3, 0, 0)

    exact = _exact_reference_tetra_monomial_integral(px, py, pz)
    approx = _quadrature_monomial_integral(3, px, py, pz)

    assert np.isclose(approx, exact, atol=1e-14)