import pytest

from scikit_poles_zeros._domain import Rectangle


class TestRectangle:
    @pytest.mark.parametrize(("bl", "tr"), [(0, 0), (0, 1), (0, 1j), (1 + 1j, 0)])
    def test_iv(self, bl, tr):
        with pytest.raises(ValueError, match="right and above bottom left"):
            Rectangle(bl, tr)

    def test_attributes(self):
        bl, tr = 1 + 2j, 12 + 10j
        r = Rectangle(bl, tr)
        assert r.bottom_left == bl
        assert r.top_right == tr
        assert r.corners == (bl, 12 + 2j, tr, 1 + 10j)
