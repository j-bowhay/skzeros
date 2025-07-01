# test that ensure the examples given in the paper are working correctly
import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose
from scipy.stats import qmc

from skzeros import AAA, Rectangle, find_zeros, force_subdivide, poles_residues
from skzeros._domain import get_leaf_regions


def test_branch_cut():
    def f(z):
        return np.sin(np.sqrt(z**2 + 1)) - z

    def f_prime(z):
        return z * np.cos(np.sqrt(z**2 + 1)) / np.sqrt(z**2 + 1) - 1

    def F(z):
        return (np.sin(np.sqrt(z**2 + 1)) - z) * (-np.sin(np.sqrt(z**2 + 1)) - z)

    def F_prime(z):
        return (-z - np.sin(np.sqrt(z**2 + 1))) * (
            z * np.cos(np.sqrt(z**2 + 1)) / np.sqrt(z**2 + 1) - 1
        ) + (-z + np.sin(np.sqrt(z**2 + 1))) * (
            -z * np.cos(np.sqrt(z**2 + 1)) / np.sqrt(z**2 + 1) - 1
        )

    d = Rectangle(complex(-5, -5), complex(5, 5))
    res1 = AAA(lambda z: f_prime(z) / f(z), d, max_iter=50, err_on_max_iter=False)
    res2 = AAA(lambda z: F_prime(z) / F(z), d)
    poles_1, residue_1 = poles_residues(*res1, residue=True)
    poles_2, residue_2 = poles_residues(*res2, residue=True)
    mask_1 = (np.abs(poles_1.real) < 5) & (np.abs(poles_1.imag) < 5)
    mask_2 = (np.abs(poles_2.real) < 5) & (np.abs(poles_2.imag) < 5)
    assert np.sum(np.isclose(residue_1[mask_1], 1, atol=1e-2, rtol=0)) == 2
    assert np.sum(np.isclose(residue_2[mask_2], 1, atol=1e-2, rtol=0)) == 8


@pytest.mark.filterwarnings("ignore:.*balance properties of Sobol' points require*")
def test_example_1():
    N = 100
    rng = np.random.default_rng(12345)
    sampler = qmc.Sobol(d=2, rng=rng)
    tmp = sampler.random(N)
    zeros = tmp[:, 0] + tmp[:, 1] * 1j

    def f(z):
        return np.prod(np.subtract.outer(z, zeros), axis=-1)

    def f_z(z):
        tmp = np.subtract.outer(z, zeros)
        return np.prod(tmp, axis=-1) * np.sum(1 / tmp, axis=-1)

    d = Rectangle(0, complex(1, 1))
    res = find_zeros(d, f, f_z, max_arg_principle=7.1)
    assert_allclose(np.sort(res.zeros), np.sort(zeros))


def test_example_2():
    A = -0.19435
    B = 1000.41
    C = 522463
    T = 0.005

    def f(z):
        return z**2 + A * z + B * np.exp(-T * z) + C

    def f_z(z):
        return 2 * z + A - T * B * np.exp(-T * z)

    r = Rectangle(complex(-2500, -15000), complex(10, 15000))
    res = find_zeros(r, f, f_z, max_arg_principle=7)
    zz = np.add.outer(np.linspace(-2500, 10), np.linspace(-15000, 15000) * 1j)
    f_bar = np.abs(np.median(f(zz)))
    assert_allclose(np.abs(f(res.zeros)) / f_bar, 0, atol=1e-10)


# https://discourse.julialang.org/t/computing-the-adjugate-cofactors-of-a-matrix/125184
def adjugate(A):
    U, S, VH = np.linalg.svd(A)
    phase_det = np.linalg.det(VH @ U)
    return VH.T.conj() @ (_adjugate_diagonal(S) * phase_det) @ U.conj().T


def _adjugate_diagonal(d):
    n = len(d)
    if n < 2:
        return np.diag(np.ones_like(d))

    dadj = np.empty_like(d)
    prefix = 1
    for i in range(n):
        dadj[i] = prefix
        prefix *= d[i]

    suffix = 1
    for i in reversed(range(n)):
        dadj[i] *= suffix
        suffix *= d[i]

    return np.diag(dadj)


def test_example_3a():
    rng = np.random.default_rng(1212)
    n = 50
    x = -0.4 + 0.8 * rng.integers(0, 1, n, endpoint=True)
    A = scipy.linalg.circulant(x)
    e = np.linalg.eigvals(A)

    @np.vectorize
    def f_1(z):
        return np.linalg.det(A - z * np.eye(n))

    @np.vectorize
    def f_1_z(z):
        return -np.trace(adjugate(A - z * np.eye(n)))

    d = Rectangle(complex(-5.1, -4.9), complex(5, 4.7))
    r = find_zeros(d, f_1, f_1_z, 7.1)
    assert_allclose(np.min(np.abs(np.subtract.outer(e, r.zeros)), axis=1), 0, atol=1e-8)


def test_example_3b():
    rng = np.random.default_rng(1212)
    n = 50
    x = -0.4 + 0.8 * rng.integers(0, 1, n, endpoint=True)
    A = scipy.linalg.circulant(x)
    e = np.linalg.eigvals(A)
    rng = np.random.default_rng(122144241122)
    u, v = rng.random((2, n))

    @np.vectorize
    def f_2(z):
        return u.T @ scipy.linalg.solve(A - z * np.eye(n), v)

    @np.vectorize
    def f_2_z(z):
        return u.T @ scipy.linalg.solve(
            A - z * np.eye(n), scipy.linalg.solve(A - z * np.eye(n), v)
        )

    d = Rectangle(complex(-5.1, -4.9), complex(5, 4.7))
    force_subdivide(d, 6)
    regions = get_leaf_regions(d)
    poles = np.array([])
    for region in regions:
        res = AAA(lambda z: f_2_z(z) / f_2(z), region)
        p, r = poles_residues(*res, residue=True)
        bl, tr = region.bottom_left, region.top_right
        to_keep = (
            (p.real > bl.real)
            & (p.real < tr.real)
            & (p.imag > bl.imag)
            & (p.imag < tr.imag)
            & (r < 0)
            & (abs(r.real + 1) < 1e-3)
        )
        poles = np.concatenate((poles, p[to_keep]))
    assert_allclose(np.min(np.abs(np.subtract.outer(e, poles)), axis=1), 0, atol=1e-8)


@np.vectorize
def cauchy_integral_derivative(f, z, r=1, rtol=1e-15):
    def integrand(t):
        return np.exp(-t) * f(z + r * np.exp(t))

    m = 2
    s = integrand(2j * np.pi * np.arange(1, m + 1) / m)
    val1 = np.mean(s) / r
    err1 = np.nan
    while m < 1e6:
        m *= 2
        s = np.concatenate((s, integrand(2j * np.pi * np.arange(1, m + 1, step=2) / m)))
        val = np.mean(s) / r
        kappa = np.mean(np.abs(s)) / np.abs(val)
        err0 = np.abs(val - val1) / np.abs(val)
        err = (err0 / err1) ** 2 * err0
        if err <= kappa * rtol:
            return val
        val1 = val
        err1 = err0
    return np.nan


def test_log_deriv_accuracy():
    d = Rectangle(0, complex(1, 1))

    rng = np.random.default_rng(123456)
    orders = [1, 2, 4, 6]

    for order in orders:
        for _ in range(1000):
            # generate random zero
            zero = complex(rng.uniform(), rng.uniform())

            def f(z, order=order, zero=zero):
                return np.exp(z) * (z - zero) ** order

            def f_z(z, order=order, zero=zero):
                return (
                    order * (z - zero) ** (order - 1) + (z - zero) ** order
                ) * np.exp(z)

            res_log_deriv = AAA(lambda z: f_z(z) / f(z), d)

            assert_allclose(
                np.min(np.abs(poles_residues(*res_log_deriv) - zero)), 0, atol=1e-12
            )

            res_log_deriv_AAA = AAA(
                lambda z: cauchy_integral_derivative(f, z, r=1e-2) / f(z), d, rtol=1e-12
            )
            assert_allclose(
                np.min(np.abs(poles_residues(*res_log_deriv_AAA) - zero)), 0, atol=1e-12
            )
