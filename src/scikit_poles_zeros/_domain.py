from __future__ import annotations


class Rectangle:
    """Rectangle region in the complex plane"""

    def __init__(self, bottom_left, top_right, /):
        self.bottom_left = bottom_left
        self.top_right = top_right

        # children created if region is subdivided
        self.child_a = None  # left/top region
        self.child_b = None  # right/bottom region
