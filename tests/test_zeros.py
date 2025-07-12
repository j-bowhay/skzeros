import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import qmc

from skzeros import Rectangle, find_zeros


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_max_principal_too_high():
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
    with pytest.warns(RuntimeWarning):
        res = find_zeros(d, f, f_z, max_arg_principle=50)
    assert_allclose(np.sort(res.zeros), np.sort(zeros), rtol=5e-7)
