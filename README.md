# scikit-zeros

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

## [Documentation](https://scikit-zeros.readthedocs.io/en/stable/)

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/j-bowhay/scikit-zeros/workflows/CI/badge.svg
[actions-link]:             https://github.com/j-bowhay/scikit-zeros/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/scikit-zeros
[conda-link]:               https://github.com/conda-forge/scikit-zeros-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/j-bowhay/scikit-zeros/discussions
[pypi-link]:                https://pypi.org/project/skzeros/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/skzeros
[pypi-version]:             https://img.shields.io/pypi/v/skzeros
[rtd-badge]:                https://readthedocs.org/projects/scikit-zeros/badge/?version=latest
[rtd-link]:                 https://scikit-zeros.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

`scikit-zeros` is a Python package for finding the all the roots of a
holomorphic function in a given region of the complex plane. It is based on
subdivision using the argument principle combined with AAA rational
approximation of the logarithm derivative.

## Installation

You can install `scikit-zeros` using pip:

```bash
pip install skzeros
```

## Example

```python
import numpy as np
import skzeros

A = -0.19435
B = 1000.41
C = 522463
T = 0.005


def f(z):
    return z**2 + A * z + B * np.exp(-T * z) + C


def f_z(z):
    return 2 * z + A - T * B * np.exp(-T * z)


r = skzeros.Rectangle(complex(-2500, -15000), complex(10, 15000))
res = skzeros.find_zeros(r, f, f_z, max_arg_principle=7)
```

## See Also

- [cxroots](https://github.com/rparini/cxroots)
- [POLZE](https://github.com/nennigb/polze)
