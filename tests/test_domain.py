from scikit_poles_zeros._domain import Rectangle


class TestRectangle:
    def test_attributes(self):
        bl, tr = 0, 10 + 10j
        r = Rectangle(bl, tr)
        assert r.bottom_left == bl
        assert r.top_right == tr
