import numpy as np
from numpy.testing import assert_allclose

from scikit_poles_zeros._AAA import XS


def test_XS():
    res = XS(np.arange(1, 6), 2)
    excepted = np.asarray(
        [1.3333, 1.6667, 2.3333, 2.6667, 3.3333, 3.6667, 4.3333, 4.6667]
    )
    assert_allclose(np.sort(res), excepted, rtol=1e-3)
