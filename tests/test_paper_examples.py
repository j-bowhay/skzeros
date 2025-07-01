# test that ensure the examples given in the paper are working correctly
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import qmc

from skzeros import AAA, Rectangle, find_zeros, poles_residues


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
