import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from skzeros._AAA import AAA, poles_residues
from skzeros._domain import _subdivide_domain

__all__ = ["ZerosResult", "find_zeros"]


@dataclass
class ZerosResult:
    """Results class for the zeros found by the `find_zeros` function.

    Attributes
    ----------
    zeros : np.ndarray
        The computed zeros of the function.
    multiplicities : np.ndarray
        The multiplicities of the zeros.
    """

    zeros: np.ndarray
    multiplicities: np.ndarray


def find_zeros(
    domain, f, f_z, max_arg_principle, quadrature_args=None, maxiter=50, rng=None
):
    """Find all the zeros of a holomorphic function in a given domain.

    First the `domain` is subdivided into smaller regions until the value of the
    argument principle is less than `max_arg_principle`. Then, a AAA approximation
    of the logarithmic derivative is used to locate the zeros of `f`.

    Parameters
    ----------
    domain : Domain
        The domain in which to search for zeros.
    f : callable
        The holomorphic function for which to find zeros. It should accept a 1D array
        and return a 1D array of complex numbers.
    f_z : callable
        The derivative of the holomorphic function `f`. It should accept a 1D array
        and return a 1D array of complex numbers.
    max_arg_principle : float
        Subdivide `domain` until the value of the argument principle is less than this
        value.
    quadrature_args : dict, optional
        Additional arguments to pass to the quadrature method used in subdivision.
        See `scipy.integrate.quadvec` for the supported arguments, by default None.
    maxiter : int, optional
        The maximum number of attempts at subdivision allowed, by default 50
    rng : np.random.Generator, optional
        Random number generator to use for sampling points in the domain. If None,
        the default numpy random generator is used.

    Returns
    -------
    ZerosResult
        A `ZerosResult` object containing the computed zeros and their multiplicities.
    """
    regions = _subdivide_domain(
        domain=domain,
        f=f,
        f_z=f_z,
        max_arg_principle=max_arg_principle,
        quadrature_args=quadrature_args,
        maxiter=maxiter,
        rng=rng,
    )
    queue = deque(regions)
    zeros = np.array([])
    multiplicities = np.array([])
    while len(queue) > 0:
        region = queue.popleft()
        # 1. Get the expected number of poles
        expected = region._arg_principle
        # if there are no zeros then there is no work to be done!
        if np.isclose(expected, 0, atol=1e-3):
            continue
        # 2. Apply continuum AAA
        support_points, support_values, weights = AAA(
            lambda z, f=f, f_z=f_z: f_z(z) / f(z), region
        )
        # 3. Compute poles and residue
        poles, residue = poles_residues(
            support_points, support_values, weights, residue=True
        )
        # 4. Discard any out of the region
        bl, tr = region.bottom_left, region.top_right
        to_keep = (
            (poles.real >= bl.real)
            & (poles.real <= tr.real)
            & (poles.imag >= bl.imag)
            & (poles.imag <= tr.imag)
        )
        # 5. Discard not close to a positive int
        to_keep &= ~np.isclose(residue, 0)
        to_keep &= np.isclose(np.round(residue.real), residue)
        # 6. Compare against the argument principle and subdivide further if needed
        actual = np.sum(residue[to_keep])

        if np.isclose(actual, expected):
            zeros = np.concat((zeros, poles[to_keep]))
            multiplicities = np.concat(
                (multiplicities, np.round(residue[to_keep].real))
            )
        else:
            standard_split_failed = False
            while True:
                warnings.warn(
                    "After initial subdivision the argument principle does "
                    "not match the output of AAA, applying further "
                    "subdivision",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if not standard_split_failed:
                    # first we try just a standard subdivision
                    region.subdivide()
                else:
                    region.subdivide(
                        offset=((0.1 - 0.01) * rng.random(1)[0] + 0.01)
                        * (-1) ** rng.integers(1, 2, endpoint=True)
                    )
                for child in region.children:
                    arg_principle = child.argument_principle(
                        f, f_z, quadrature_args=quadrature_args
                    )
                    if np.any(~arg_principle.success):
                        region.children = []
                        standard_split_failed = True
                        break
                    child._arg_principle = (
                        round(arg_principle.integral.real)
                        + round(arg_principle.integral.imag) * 1j
                    )
                else:
                    queue.extend(region.children)
                    break

    return ZerosResult(zeros=zeros, multiplicities=multiplicities)
