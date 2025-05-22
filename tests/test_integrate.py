from scipy.integrate import tanhsinh

from skzeros._integrate import _quadvec


def test_quadvec_rich_result():
    def f(x):
        return x

    res = _quadvec(f, 0, 1)
    ref = tanhsinh(f, 0, 1)

    assert type(res) is type(ref)
    assert hasattr(res, "success")
    assert hasattr(res, "status")
    assert hasattr(res, "integral")
    assert hasattr(res, "error")
    assert hasattr(res, "nfev")
