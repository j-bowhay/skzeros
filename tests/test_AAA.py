import numpy as np
from numpy.testing import assert_allclose

from scikit_poles_zeros._AAA import XS


class TestXS:
    def test_XS_against_MATLAB(self):
        res = XS(np.arange(0, 6), 2)
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
        res = XS([0.25, 0.5, 0.75], 5)
        expected = XS([0, 0.25, 0.5, 0.75, 1], 5)
        assert_allclose(res, expected)
