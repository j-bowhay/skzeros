from abc import ABC, abstractmethod
from collections import deque
from functools import cached_property

import numpy as np
from scipy.integrate import tanhsinh


class Domain(ABC):
    """Abstract Domain"""

    __slots__ = ()

    @property
    @abstractmethod
    def children(self): ...

    @abstractmethod
    def contour_integral(self, f): ...

    @abstractmethod
    def subdivide(self): ...


class Rectangle(Domain):
    """Rectangle region in the complex plane."""

    __slots__ = "__dict__", "_bottom_left", "_children", "_top_right"

    def __init__(self, bottom_left, top_right, /):
        # check that top_right is to the right and above bottom left in the complex
        # plane
        if bottom_left.real >= top_right.real or bottom_left.imag >= top_right.imag:
            msg = (
                "`top_right` must be to the right and above bottom left in the complex "
                "plane"
            )
            raise ValueError(msg)
        self._bottom_left = bottom_left
        self._top_right = top_right

        # children created if region is subdivided
        # 0th entry left/top, 1st entry right/bottom
        self._children = None

    @property
    def bottom_left(self):
        return self._bottom_left

    @property
    def top_right(self):
        return self._top_right

    @property
    def children(self):
        return self._children

    @cached_property
    def corners(self):
        """Returns the corners of the rectangle in a counter clockwise order
        starting from the bottom left."""
        bl = self._bottom_left
        tr = self._top_right
        return (bl, complex(tr.real, bl.imag), tr, complex(bl.real, tr.imag))

    def contour_integral(self, f, tanhsinh_args=None):
        """Compute the contour integral of `f` around the region."""
        tanhsinh_args = {} if tanhsinh_args is None else tanhsinh_args

        def f_wrapped(t, _a, _b):
            return f(_a * (1 - t) + _b * t)

        a, b = self.corners, np.roll(self.corners, -1)
        res = tanhsinh(f_wrapped, 0, 1, args=(a, b))
        res.integral *= b - a
        return res

    def subdivide(self): ...


def _subdivide_domain(domain, f, f_z, f_zz, max_poles_zeros):
    queue = deque(domain)
    while len(queue) > 0:
        current_domain = queue.popleft()

        # 1. Compute the combined number of poles and zeros in the domain
        def second_log_derivative(z):
            return (f_zz(z) / f(z)) - (f_z(z) / f(z)) ** 2

        n_poles_zeros = current_domain.contour_integral(second_log_derivative)
        # 2. Subdivide and repeat if this number is too high
        if n_poles_zeros > max_poles_zeros:
            current_domain.subdivide()
            queue.append(current_domain.children)
