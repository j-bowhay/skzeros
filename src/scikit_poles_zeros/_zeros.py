from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scikit_poles_zeros._AAA import AAA, poles_residues
from scikit_poles_zeros._domain import _subdivide_domain

__all__ = ["ZerosResult", "find_zeros"]


@dataclass
class ZerosResult:
    zeros: npt.ArrayLike
    multiplicities: npt.ArrayLike


def find_zeros(
    domain, f, f_z, max_arg_principle, quadrature_args=None, maxiter=50, rng=None
):
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
            msg = (
                "Actual not equal expected, subdivision required"
                " however this has not been implemented yet!"
            )
            raise NotImplementedError(msg)

    return ZerosResult(zeros=zeros, multiplicities=multiplicities)
