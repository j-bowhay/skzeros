"""
Copyright (c) 2025 Jake Bowhay. All rights reserved.

scikit-poles-zeros: A package to compute all the poles and zeros of a meromorphic
function in a bounded subset of the complex plane.
"""

from ._AAA import AAA, evaluate, poles_residues, zeros
from ._domain import Rectangle
from ._version import version as __version__
from ._zeros import ZerosResult, find_zeros

__all__ = [
    "AAA",
    "Rectangle",
    "ZerosResult",
    "__version__",
    "evaluate",
    "find_zeros",
    "poles_residues",
    "zeros",
]
