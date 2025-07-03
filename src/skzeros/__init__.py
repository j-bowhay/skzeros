"""
Copyright (c) 2025 Jake Bowhay. All rights reserved.

scikit-zeros: A package to compute all the zeros of a holomorphic function in a region
of the complex plane.
"""

from ._AAA import AAA, derivative, evaluate, poles_residues, zeros
from ._domain import Domain, Rectangle, force_subdivide
from ._version import version as __version__
from ._zeros import ZerosResult, find_zeros

__all__ = [
    "AAA",
    "Domain",
    "Rectangle",
    "ZerosResult",
    "__version__",
    "derivative",
    "evaluate",
    "find_zeros",
    "force_subdivide",
    "poles_residues",
    "zeros",
]
