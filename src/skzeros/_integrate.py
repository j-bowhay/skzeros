from scipy._lib._util import _RichResult
from scipy.integrate import quad_vec


def _quadvec(
    f,
    a,
    b,
    *,
    epsabs=1e-200,
    epsrel=1e-8,
    norm="2",
    cache_size=100e6,
    limit=10000,
    workers=1,
    points=None,
    quadrature=None,
    args=(),
):
    """Wrapper around quadvec to return in the same style as `tanhsinh`."""
    res, err, info = quad_vec(
        f,
        a,
        b,
        epsabs=epsabs,
        epsrel=epsrel,
        norm=norm,
        cache_size=cache_size,
        limit=limit,
        workers=workers,
        points=points,
        quadrature=quadrature,
        full_output=True,
        args=args,
    )
    return _RichResult(
        success=info.success,
        status=info.status,
        integral=res,
        error=err,
        nfev=info.neval,
    )
