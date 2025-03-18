import numpy as np
from numpy.testing import assert_allclose

from .problems import ExampleHolomorphic


class TestExampleHolomorphic:
    def expected_f_z(z):
        return (
            50 * z**49
            + 12 * z**11
            + 60 * np.sin(12 * z) * np.sin(20 * z)
            - 100 * np.cos(12 * z) * np.cos(20 * z)
        )

    def expected_f_zz(z):
        return 2 * (
            1225 * z**48
            + 66 * z**10
            + 1200 * np.sin(12 * z) * np.cos(20 * z)
            + 1360 * np.sin(20 * z) * np.cos(12 * z)
        )

    def test_example_holomorphic(self):
        rng = np.random.default_rng(9023584029578023580)
        z = rng.random(1000) + rng.random(1000) * 1j
        assert_allclose(ExampleHolomorphic.f_z(z), self.expected_f_z(z))
        assert_allclose(ExampleHolomorphic.f_zz(z), self.expected_f_zz(z))
