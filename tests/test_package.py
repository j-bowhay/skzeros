from __future__ import annotations

import importlib.metadata

import skzeros as m


def test_version():
    assert importlib.metadata.version("skzeros") == m.__version__
