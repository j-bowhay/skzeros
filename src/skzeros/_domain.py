from abc import ABC, abstractmethod
from collections import deque
from math import pi
from typing import Literal

import numpy as np
from matplotlib import patches
from scipy._lib._util import _RichResult
from scipy.integrate import tanhsinh

from skzeros._integrate import _quadvec

__all__ = ["Domain", "Rectangle", "force_subdivide"]


class Domain(ABC):
    """Abstract Domain"""

    __slots__ = ("_arg_principle", "parent")

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

    @abstractmethod
    def sample_boundary(self, t):
        t = np.asarray(t)
        if not np.all((t >= 0.0) & (t < 1.0)):
            msg = "t must lie in the interval [0,1)"
            raise ValueError(msg)
        return t


class Rectangle(Domain):
    """Rectangle region in the complex plane.

    Parameters
    ----------
    bottom_left : complex
        Bottom left corner of the rectangle.
    top_right : complex
        Top right corner of the rectangle.
    parent : Domain, optional
        Parent domain, by default None. This is used to keep track of the hierarchy of
        domains and is useful for subdividing the domain.
    """

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
        """Bottom left corner of the rectangle."""
        return self._bottom_left

    @property
    def top_right(self):
        """Top right corner of the rectangle."""
        return self._top_right

    @property
    def corners(self):
        """Returns the corners of the rectangle in a counter clockwise order
        starting from the bottom left."""
        return self._corners

    def contour_integral(
        self, f, *, method: Literal["gk21", "tanhsinh"] = "gk21", quadrature_args=None
    ):
        """Compute the contour integral of `f` around the region.

        Parameters
        ----------
        f : callable
            Function to integrate. It should accept a complex number and return a
            complex number.
        method : {'gk21', 'tanhsinh'}, optional
            Method to use for the contour integral. 'gk21' uses the GK21 method
            (Gauss-Kronrod 21-point rule) and 'tanhsinh' uses the tanhsinh method.
            By default 'gk21'.
        quadrature_args : dict, optional
            Additional arguments to pass to the quadrature method.

        Returns
        -------
        _RichResult
            Result of the contour integral, which includes the integral value, error,
            number of function evaluations, and success status.
        """
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

    def subdivide(self, *, offset=0):
        """Subdivide the rectangle into two smaller rectangles.

        The rectangles are split either vertically or horizontally depending on the
        aspect ratio of the rectangle. The `offset` parameter controls how much the
        split is offset from the center of the rectangle. The two child rectangles
        are stored in the `children` attribute of the rectangle.

        Parameters
        ----------
        offset : float, optional
            Offset for the split, by default 0.0. Must be between -0.5 and 0.5.
            A positive offset will move the split towards the right or top, while a
            negative offset will move it towards the left or bottom. If the offset is
            0.0, the rectangle is split exactly in the middle.
        """
        if not abs(offset) < 0.5:
            msg = "Offset must be between -0.5 and 0.5"
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
        """Plot the rectangle on a given matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis to plot on.
        """
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

    def sample_boundary(self, t):
        """Sample points on the boundary of the rectangle.

        Parameters
        ----------
        t : array_like
            Array of values in the interval [0, 1) which parametrise where the boundary
            points will be sampled. The values are interpreted as follows:
            - [0, 0.25) corresponds to the left edge
            - [0.25, 0.5) corresponds to the top edge
            - [0.5, 0.75) corresponds to the right edge
            - [0.75, 1) corresponds to the bottom edge

        Returns
        -------
        ndarray
            An array of complex numbers representing the sampled points on the boundary
            of the rectangle.
        """
        t = super().sample_boundary(t)
        out = np.empty_like(t, dtype=np.complex128)
        corners = self.corners
        edge1 = (t >= 0.0) & (t < 0.25)
        edge2 = (t >= 0.25) & (t < 0.5)
        edge3 = (t >= 0.5) & (t < 0.75)
        edge4 = (t >= 0.75) & (t < 1.0)
        t_1 = t[edge1] / 0.25
        t_2 = (t[edge2] - 0.25) / 0.25
        t_3 = (t[edge3] - 0.5) / 0.25
        t_4 = (t[edge4] - 0.75) / 0.25
        out[edge1] = corners[0] * (1 - t_1) + corners[1] * t_1
        out[edge2] = corners[1] * (1 - t_2) + corners[2] * t_2
        out[edge3] = corners[2] * (1 - t_3) + corners[3] * t_3
        out[edge4] = corners[3] * (1 - t_4) + corners[0] * t_4
        return out


def _subdivide_domain(
    domain, f, f_z, max_arg_principle, quadrature_args=None, maxiter=50, rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    if domain.children != []:
        msg = "`domain` must not already have children."
        raise ValueError(msg)

    queue = deque([domain])
    leafs = []
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
                if child in leafs:
                    leafs.remove(child)
            parent.children = []
            # 3. Resplit the region and add to the front of the queue
            parent.subdivide(
                offset=((0.1 - 0.01) * rng.random(1)[0] + 0.01)
                * (-1) ** rng.integers(1, 2, endpoint=True)
            )
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
        else:  # Accept!
            leafs.append(current_domain)
            current_domain._arg_principle = (
                round(arg_principle.integral.real)
                + round(arg_principle.integral.imag) * 1j
            )
    return leafs


def force_subdivide(region, n_times):
    """Forcefully subdivide a region a specified number of times.

    Parameters
    ----------
    region : Domain
        Domain to subdivide.
    n_times : int
        Number of times to subdivide the domain.
    """

    def inner(region, level):
        if level < n_times:
            region.subdivide()
            for child in region.children:
                inner(child, level + 1)

    inner(region, 0)


def get_leaf_regions(r):
    if len(r.children) >= 1:
        leafs = []
        for child in r.children:
            if len(child.children) == 0:
                leafs.append(child)
            else:
                leafs.extend(get_leaf_regions(child))
        return leafs
    return [r]
