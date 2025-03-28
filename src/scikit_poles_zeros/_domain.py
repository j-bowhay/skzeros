from abc import ABC, abstractmethod
from collections import deque
from math import pi
from typing import Literal

import numpy as np
from matplotlib import patches
from scipy._lib._util import _RichResult
from scipy.integrate import tanhsinh

from scikit_poles_zeros._integrate import _quadvec


class Domain(ABC):
    """Abstract Domain"""

    __slots__ = ("parent",)

    def __init__(self, *, parent=None):
        self.parent = parent

    @property
    @abstractmethod
    def children(self): ...

    @abstractmethod
    def contour_integral(
        self, f, *, method: Literal["gk21", "tanhsinh"] = "gk21", quadrature_args=None
    ): ...

    @abstractmethod
    def subdivide(self): ...

    @abstractmethod
    def plot(self, ax): ...

    def argument_principle(
        self,
        f,
        f_z,
        *,
        method: Literal["gk21", "tanhsinh"] = "gk21",
        quadrature_args=None,
    ):
        with np.errstate(invalid="ignore", divide="ignore"):
            res = self.contour_integral(
                lambda z: f_z(z) / f(z), method=method, quadrature_args=quadrature_args
            )
        res.integral /= complex(0, 2 * pi)
        return res


class Rectangle(Domain):
    """Rectangle region in the complex plane."""

    __slots__ = "_bottom_left", "_corners", "_top_right", "children"

    def __init__(self, bottom_left, top_right, /, *, parent=None):
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
        self._corners = (
            bottom_left,
            complex(top_right.real, bottom_left.imag),
            top_right,
            complex(bottom_left.real, top_right.imag),
        )

        # children created if region is subdivided
        # 0th entry left/top, 1st entry right/bottom
        self.children = []

        super().__init__(parent=parent)

    @property
    def bottom_left(self):
        return self._bottom_left

    @property
    def top_right(self):
        return self._top_right

    @property
    def corners(self):
        """Returns the corners of the rectangle in a counter clockwise order
        starting from the bottom left."""
        return self._corners

    def contour_integral(
        self, f, *, method: Literal["gk21", "tanhsinh"] = "gk21", quadrature_args=None
    ):
        """Compute the contour integral of `f` around the region."""
        quadrature_args = {} if quadrature_args is None else quadrature_args

        def f_wrapped(t, _a, _b):
            return f(_a * (1 - t) + _b * t)

        a, b = np.asarray(self.corners), np.roll(self.corners, -1)
        if method == "tanhsinh":
            res = tanhsinh(f_wrapped, 0, 1, args=(a, b), **quadrature_args)
        elif method == "gk21":
            success = []
            status = []
            integral = []
            error = []
            nfev = []
            for args in zip(a, b, strict=False):
                res_i = _quadvec(f_wrapped, 0, 1, args=args, **quadrature_args)
                success.append(res_i.success)
                status.append(res_i.status)
                integral.append(res_i.integral)
                error.append(res_i.error)
                nfev.append(res_i.nfev)
            res = _RichResult(
                success=np.asarray(success),
                status=np.asarray(status),
                integral=np.asarray(integral),
                error=np.asarray(error),
                nfev=np.asarray(nfev),
            )
        else:
            msg = "Invalid `method`"
            raise ValueError(msg)

        # multiply by the Jacobian
        res.integral *= b - a
        res.integral = np.sum(res.integral)
        return res

    def subdivide(self, offset=0):
        if not abs(offset) < 0.5:
            msg = "Offset must be between 0 and 1"
            raise ValueError(msg)
        diag = self.top_right - self.bottom_left

        if diag.real >= diag.imag:  # split vertically
            self.children.append(
                Rectangle(
                    self.bottom_left,
                    self.top_right - diag.real * (0.5 + offset),
                    parent=self,
                )
            )
            self.children.append(
                Rectangle(
                    self.bottom_left + diag.real * (0.5 - offset),
                    self.top_right,
                    parent=self,
                )
            )
        else:  # split horizontally
            self.children.append(
                Rectangle(
                    self.bottom_left,
                    self.top_right - 1j * diag.imag * (0.5 + offset),
                    parent=self,
                )
            )
            self.children.append(
                Rectangle(
                    self.bottom_left + 1j * diag.imag * (0.5 - offset),
                    self.top_right,
                    parent=self,
                )
            )

    def plot(self, ax):
        diff = self.top_right - self.bottom_left
        ax.add_patch(
            patches.Rectangle(
                (self.bottom_left.real, self.bottom_left.imag),
                diff.real,
                diff.imag,
                fc="none",
                edgecolor="r",
                lw=2,
            )
        )
        for child in self.children:
            child.plot(ax)


def _subdivide_domain(
    domain, f, f_z, max_arg_principle, quadrature_args=None, maxiter=50
):
    queue = deque([domain])
    i = 0
    while len(queue) > 0:
        i += 1
        if i > maxiter:
            msg = "Max number of iterations reached"
            raise RuntimeError(msg)
        current_domain = queue.popleft()

        # 1. Compute the combined number of poles and zeros in the domain
        arg_principle = current_domain.argument_principle(
            f, f_z, quadrature_args=quadrature_args
        )
        if np.any(~arg_principle.success):
            if i == 1:
                # if integration fails on the first iteration then we know that there
                # is a zero or pole on the boundary of the region, so we reject this
                msg = (
                    "Zero/Pole detected on the boundary of the provided region. Please "
                    "adjust region."
                )
                raise RuntimeError(msg)
            # otherwise we know the edge that we previously inserted passes through a
            # zero or pole so we need to try moving that edge
            # 1. Find the parent region
            parent = current_domain.parent
            # 2. Delete its children and remove them from the queue
            for child in parent.children:
                if child in queue:
                    queue.remove(child)
            parent.children = []
            # 3. Resplit the region and add to the front of the queue
            parent.subdivide(
                offset=0.111111
            )  # hardcode for now, possibly choose randomly later
            queue.extend(parent.children)
            continue

        if not (
            abs(arg_principle.integral.real - round(arg_principle.integral.real)) < 1e-3
            and abs(arg_principle.integral.imag) < 1e-3
        ):
            msg = (
                "Non-integer value of the argument principle computed. Try "
                "tightening quadrature tolerance and check that `f` is holomorphic."
            )
            raise RuntimeError(msg)

        # 2. Subdivide and repeat if this number is too high
        if abs(arg_principle.integral) > max_arg_principle:
            current_domain.subdivide()
            queue.extend(current_domain.children)
