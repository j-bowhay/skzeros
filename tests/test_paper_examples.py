# test that ensure the examples given in the paper are working correctly
import numpy as np

from skzeros import AAA, Rectangle, poles_residues


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
