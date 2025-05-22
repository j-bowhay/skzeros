import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from skzeros import Rectangle
from skzeros._domain import _subdivide_domain

from . import problems


class TestRectangle:
    @pytest.mark.parametrize(("bl", "tr"), [(0, 0), (0, 1), (0, 1j), (1 + 1j, 0)])
    def test_iv(self, bl, tr):
        with pytest.raises(ValueError, match="right and above bottom left"):
            Rectangle(bl, tr)

    def test_attributes(self):
        bl, tr = 1 + 2j, 12 + 10j
        r = Rectangle(bl, tr)
        assert r.bottom_left == bl
        assert r.top_right == tr
        assert r.corners == (bl, 12 + 2j, tr, 1 + 10j)

    def test_parent(self):
        r = Rectangle(0, complex(1, 1))
        r.subdivide()
        for child in r.children:
            assert child.parent is r

    @pytest.mark.parametrize("attr", ["top_right", "bottom_left", "corners"])
    def test_read_only(self, attr):
        d = Rectangle(0, complex(1, 1))
        with pytest.raises(AttributeError):
            setattr(d, attr, 1)

    @pytest.mark.parametrize("method", ["gk21", "tanhsinh"])
    @pytest.mark.parametrize(
        ("f", "bl", "tr", "expected"),
        [
            (lambda z: 1 / z, complex(-1, -1), complex(1, 1), 2j * np.pi),
            (lambda z: 1 / (z**2 + 1) ** 2, -1, complex(10, 10), np.pi / 2),
            (lambda z: np.sin(z), complex(-10, -10), complex(12, 3), 0),
        ],
    )
    def test_contour_integral(self, f, bl, tr, expected, method):
        d = Rectangle(bl, tr)
        res = d.contour_integral(f, method=method)
        assert_equal(res.success, np.ones_like(res.integral, dtype=np.bool))
        assert_equal(res.status, np.zeros_like(res.integral))
        assert_allclose(res.integral, expected, atol=1e-10)

    def test_contour_integral_args_pass_through(self):
        d = Rectangle(0, complex(1, 1))
        r = d.contour_integral(
            lambda z: z * 10, method="tanhsinh", quadrature_args={"maxlevel": 1}
        )
        assert np.all(~r.success)

    def test_contour_integral_invalid_method(self):
        d = Rectangle(0, complex(1, 1))
        with pytest.raises(ValueError, match="Invalid `method`"):
            d.contour_integral(lambda z: z, method="cheese")

    @pytest.mark.parametrize("problem", problems.all)
    @pytest.mark.parametrize("method", ["gk21", "tanhsinh"])
    def test_arg_principle(self, problem: problems.Problem, method):
        arg = {"maxlevel": 20} if method == "tanhsinh" else {}
        res = problem.domain.argument_principle(
            problem.f, problem.f_z, method=method, quadrature_args=arg
        )
        assert_allclose(
            res.integral,
            problem.expected_arg_principle(),
            atol=1e-12,
        )

    def test_invalid_offset(self):
        r = Rectangle(0, complex(1, 1))
        with pytest.raises(ValueError, match="Offset"):
            r.subdivide(offset=1)

    def test_sample_boundary_invalid_t(self):
        t = np.asarray([0, 0.5, 0.9, 1, 1.5])
        r = Rectangle(0, complex(1, 1))
        with pytest.raises(ValueError, match="must lie"):
            r.sample_boundary(t)

    def test_sample_boundary(self):
        r = Rectangle(0, complex(1, 1))
        assert_equal(r.sample_boundary([0, 0.25, 0.5, 0.75]), r.corners)


class TestSubdivideDomain:
    def test_existing_children(self):
        d = Rectangle(0, complex(1, 1))
        d.children.append(Rectangle(0, complex(0.5, 0.5)))

        with pytest.raises(ValueError, match="children"):
            _subdivide_domain(d, lambda _: 1, lambda _: 0, 10)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize(
        "f",
        [
            lambda z: z,
            lambda z: z - 0.5,
            lambda z: z - 1,
            lambda z: z - 0.5j,
            lambda z: z - 1j,
            lambda z: z - 1 - 1j,
        ],
    )
    def test_zero_on_boundary(self, f):
        r = Rectangle(0, complex(1, 1))
        with pytest.raises(RuntimeError, match="boundary"):
            _subdivide_domain(r, f, f_z=lambda _: 1.0, max_arg_principle=2)

    def test_non_integer_argument_principle(self):
        r = Rectangle(complex(-1, -1), complex(1, 1))
        with pytest.raises(RuntimeError, match="Non-integer"):
            _subdivide_domain(r, lambda z: z, lambda _: 1.5, max_arg_principle=2)
        with pytest.raises(RuntimeError, match="Non-integer"):
            _subdivide_domain(r, lambda z: z, lambda _: 1.5j, max_arg_principle=2)

    @pytest.mark.parametrize("problem", problems.with_known_roots_poles)
    def test_successful_subdivision(self, problem):
        r = problem.domain
        max_arg = max(
            max(problem.zeros_multiplicities, default=0),
            max(problem.poles_multiplicities, default=0),
        )
        leafs = _subdivide_domain(
            r,
            problem.f,
            problem.f_z,
            max_arg_principle=max_arg + 0.1,
        )
        leafs_expected = get_leaf_regions(r)
        assert len(leafs) == len(leafs_expected)
        for leaf in leafs_expected:
            assert leaf in leafs
            actual = 0
            for zero, multiplicity in zip(
                problem.zeros, problem.zeros_multiplicities, strict=True
            ):
                if in_rectangle(zero, leaf.bottom_left, leaf.top_right):
                    actual += multiplicity
            for pole, multiplicity in zip(
                problem.poles, problem.poles_multiplicities, strict=True
            ):
                if in_rectangle(pole, leaf.bottom_left, leaf.top_right):
                    actual -= multiplicity
            assert abs(actual) <= max_arg
            assert actual == leaf._arg_principle


def get_leaf_regions(r):
    if len(r.children) >= 1:
        leafs = []
        for child in r.children:
            if len(child.children) == 0:
                leafs.append(child)
            else:
                leafs.extend(get_leaf_regions(child))
        return leafs
    return [r]


def in_rectangle(z, bottom_left, top_right):
    return (
        bottom_left.real < z.real < top_right.real
        and bottom_left.imag < z.imag < top_right.imag
    )
