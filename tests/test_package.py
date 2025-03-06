from __future__ import annotations

import importlib.metadata

import scikit_poles_zeros as m


def test_version():
    assert importlib.metadata.version("scikit_poles_zeros") == m.__version__
