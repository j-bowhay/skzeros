import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from skzeros import AAA, Rectangle, derivative, evaluate
from skzeros._AAA import _XS


class TestXS:
    def test_XS_against_MATLAB(self):
        res = _XS(np.arange(0, 6), 2)
        excepted = np.asarray(
            [
                0.3333,
                0.6667,
                1.3333,
                1.6667,
                2.3333,
                2.6667,
                3.3333,
                3.6667,
                4.3333,
                4.6667,
            ]
        )
        assert_allclose(np.sort(res), excepted, rtol=1e-3)

    def test_include_naught_one_end_point(self):
        res = _XS([0.25, 0.5, 0.75], 5)
        expected = _XS([0, 0.25, 0.5, 0.75, 1], 5)
        assert_allclose(res, expected)


class TestAAA:
    def test_only_on_boundary(self):
        d = Rectangle(complex(-10, -10), complex(10, 10))

        def f(z):
            assert np.all(
                (np.real(z) - 10 < 1e-9)
                | (np.real(z) + 10 < 1e-9)
                | (np.imag(z) - 10 < 1e-9)
                | (np.imag(z) + 10 < 1e-9)
            )
            return np.tan(z)

        z, _, _ = AAA(f, d)
        assert np.all(
            (np.real(z) == -10)
            | (np.real(z) == 10)
            | (np.imag(z) == -10)
            | (np.imag(z) == 10)
        )

    @pytest.mark.parametrize(
        ("func", "deriv"),
        [(np.cos, lambda z: -np.sin(z)), (np.sin, np.cos), (np.exp, np.exp)],
    )
    def test_func(self, func, deriv):
        rng = np.random.default_rng(3423525)
        d = Rectangle(0, complex(5, 5))
        z, f, w = AAA(func, d)
        zz = 5 * (rng.random(100) + rng.random(100) * 1j)

        # check function
        actual = evaluate(z, f, w, zz)
        expected = func(zz)
        assert_allclose(actual, expected)
        # and at support points
        actual = evaluate(z, f, w, z)
        expected = func(z)
        assert_allclose(actual, expected)

        # first derivative
        actual = derivative(z, f, w, zz, k=1)
        expected = deriv(zz)
        assert_allclose(actual, expected)
        # and at support points
        actual = derivative(z, f, w, z, k=1)
        expected = deriv(z)
        assert_allclose(actual, expected)

    def test_derivative_length_1(self):
        rng = np.random.default_rng(890184901)
        zz = 5 * (rng.random(100) + rng.random(100) * 1j)
        res = derivative(np.array([1.0]), np.array([1.0]), np.array([1.0]), zz)
        assert_equal(res, np.zeros_like(zz))
