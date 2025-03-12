from scikit_poles_zeros._domain import Rectangle


class TestRectangle:
    def test_attributes(self):
        bl, tr = 1 + 2j, 12 + 10j
        r = Rectangle(bl, tr)
        assert r.bottom_left == bl
        assert r.top_right == tr
        assert r.corners == (bl, 12 + 2j, tr, 1 + 10j)
