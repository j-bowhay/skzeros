from collections import deque
from functools import cached_property


class Rectangle:
    """Rectangle region in the complex plane."""

    def __init__(self, bottom_left, top_right, /):
        # check that top_right is to the right and above bottom left in the complex
        # plane
        if bottom_left.real < top_right.real or bottom_left.imag > top_right.imag:
            msg = (
                "`top_right` must be to the right and above bottom left in the complex "
                "plane"
            )
            raise ValueError(msg)
        self.bottom_left = bottom_left
        self.top_right = top_right

        # children created if region is subdivided
        # 0th entry left/top, 1st entry right/bottom
        self.children = None

    @cached_property
    def corners(self):
        """Returns the corners of the rectangle in a counter clockwise order
        starting from the bottom left."""
        bl = self.bottom_left
        tr = self.top_right
        return (bl, tr.real + bl.imag, tr, bl.real + tr.imag)

    def contour_integral(self, _):
        """Compute the contour integral of `f` around the region."""

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
