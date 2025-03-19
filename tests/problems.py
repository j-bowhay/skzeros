from abc import ABC, abstractmethod
from math import sqrt

import numpy as np

from scikit_poles_zeros._domain import Rectangle


class Problem(ABC):
    @staticmethod
    @abstractmethod
    def f(z): ...

    @staticmethod
    @abstractmethod
    def f_z(z): ...

    @staticmethod
    @abstractmethod
    def f_zz(z): ...

    @property
    @abstractmethod
    def domain(self): ...

    @classmethod
    def expected_arg_principle(cls):
        return sum(cls.zeros_multiplicities) - sum(cls.poles_multiplicities)

    zeros = []
    zeros_multiplicities = []
    poles = []
    poles_multiplicities = []


class NoRootPole(Problem):
    def f(z):
        return (z - 1) ** 3

    def f_z(z):
        return 3 * (z - 1) ** 2

    def f_zz(z):
        return 6 * (z - 1)

    domain = Rectangle(complex(2, 2), complex(5, 5))


class Polynomial1(Problem):
    def f(z):
        return z**3 * (z - 1.2) ** 2

    def f_z(z):
        return z**2 * (5.0 * z**2 - 9.6 * z + 4.32)

    def f_zz(z):
        return z * (20.0 * z**2 - 28.8 * z + 8.64)

    domain = Rectangle(complex(-2, -2), complex(2, 2))

    zeros = [0, 1.2]
    zeros_multiplicities = [3, 2]


class KVB141(Problem):
    def f(z):
        return (z - 10e-2) * (1 + (z - sqrt(3)) ** 2)

    def f_z(z):
        return 2 * (z - 0.1) * (z - np.sqrt(3)) + (z - np.sqrt(3)) ** 2 + 1

    def f_zz(z):
        return 6 * z - 4 * np.sqrt(3) - 0.2

    zeros = [10e-2, sqrt(3) + 1j, sqrt(3) - 1j]
    zeros_multiplicities = [1, 1, 1]

    domain = Rectangle(complex(0, -1.2), complex(2, 1.2))


class KVB142(Problem):
    def f(z):
        return np.exp(3 * z) + 2 * z * np.cos(z) - 1

    def f_z(z):
        return -2 * z * np.sin(z) + 3 * np.exp(3 * z) + 2 * np.cos(z)

    def f_zz(z):
        return -2 * z * np.cos(z) + 9 * np.exp(3 * z) - 4 * np.sin(z)

    zeros = [
        0,
        -1.844233953262213,
        0.5308949302929305 + 1.33179187675112098j,
        0.5308949302929305 - 1.33179187675112098j,
    ]
    zeros_multiplicities = [1, 1, 1, 1]

    domain = Rectangle(complex(-2, -2), complex(2, 2))


class KVB143(Problem):
    def f(z):
        return z**2 * (z - 1) * (z - 2) * (z - 3) * (z - 4) + z * np.sin(z)

    def f_z(z):
        return (
            6 * z**5
            - 50 * z**4
            + 140 * z**3
            - 150 * z**2
            + z * np.cos(z)
            + 48 * z
            + np.sin(z)
        )

    def f_zz(z):
        return (
            30 * z**4
            - 200 * z**3
            + 420 * z**2
            - z * np.sin(z)
            - 300 * z
            + 2 * np.cos(z)
            + 48
        )

    zeros = [
        0,
        1.18906588973011365517521756,
        1.72843498616506284043592924,
        3.01990732809571222812005354,
        4.03038191606046844562845941,
    ]
    zeros_multiplicities = [2, 1, 1, 1, 1]
    domain = Rectangle(complex(-5, -5), complex(5, 5))


class ExampleHolomorphic(Problem):
    """From Locating all the zeros of an analytic function in one complex variable,
    Michael Dellnitza, Oliver Sch,utzea, Qinghua Zheng, Section 4.1"""

    def f(z):
        return z**50 + z**12 - 5 * np.sin(20 * z) * np.cos(12 * z) - 1

    def f_z(z):
        return (
            50 * z**49
            + 12 * z**11
            + 60 * np.sin(12 * z) * np.sin(20 * z)
            - 100 * np.cos(12 * z) * np.cos(20 * z)
        )

    def f_zz(z):
        return 2 * (
            1225 * z**48
            + 66 * z**10
            + 1200 * np.sin(12 * z) * np.cos(20 * z)
            + 1360 * np.sin(20 * z) * np.cos(12 * z)
        )

    domain = Rectangle(complex(-20.3, -20.3), complex(20.7, 20.7))

    @staticmethod
    def expected_arg_principle():
        return 424


class SimpleRational(Problem):
    """https://github.com/fgasdia/RootsAndPoles.jl/blob/master/test/SimpleRationalFunction.jl"""

    def f(z):
        return (z - 1) * (z - 1j) ** 2 * (z + 1) ** 3 / (z + 1j)

    def f_z(z):
        return (
            (z + 1) ** 2
            * (z - 1j)
            * (
                -(z - 1) * (z + 1) * (z - 1j)
                + (z + 1j)
                * (2 * (z - 1) * (z + 1) + 3 * (z - 1) * (z - 1j) + (z + 1) * (z - 1j))
            )
            / (z + 1j) ** 2
        )

    def f_zz(z):
        return (
            20 * z**6
            + z**5 * (24 + 24 * 1j)
            + z**4 * (24 + 36 * 1j)
            + z**3 * (16 + 24 * 1j)
            + z**2 * (12 + 24 * 1j)
            + 24 * z
            + 8
            - 12 * 1j
        ) / (z**3 + 3 * 1j * z**2 - 3 * z - 1j)

    domain = Rectangle(complex(-2, -2), complex(2, 2))

    zeros = [1, 1j, -1]
    zeros_multiplicities = [1, 2, 3]
    poles = [-1j]
    poles_multiplicities = [1]


with_known_roots_poles = (
    NoRootPole,
    SimpleRational,
    Polynomial1,
    KVB141,
    KVB142,
    KVB143,
)
without_known_roots_poles = (ExampleHolomorphic,)
all = with_known_roots_poles + without_known_roots_poles
