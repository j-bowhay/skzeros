from abc import ABC, abstractmethod
from math import sqrt

import numpy as np

from skzeros._domain import Rectangle


class Problem(ABC):
    @staticmethod
    @abstractmethod
    def f(z): ...

    @staticmethod
    @abstractmethod
    def f_z(z): ...

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

    domain = Rectangle(complex(2, 2), complex(5, 5))


class Polynomial1(Problem):
    def f(z):
        return z**3 * (z - 1.2) ** 2

    def f_z(z):
        return z**2 * (5.0 * z**2 - 9.6 * z + 4.32)

    domain = Rectangle(complex(-2, -2), complex(2, 2))

    zeros = [0, 1.2]
    zeros_multiplicities = [3, 2]


class KVB141(Problem):
    def f(z):
        return (z - 10e-2) * (1 + (z - sqrt(3)) ** 2)

    def f_z(z):
        return 2 * (z - 0.1) * (z - np.sqrt(3)) + (z - np.sqrt(3)) ** 2 + 1

    zeros = [10e-2, sqrt(3) + 1j, sqrt(3) - 1j]
    zeros_multiplicities = [1, 1, 1]

    domain = Rectangle(complex(0, -1.2), complex(2, 1.2))


class KVB142(Problem):
    def f(z):
        return np.exp(3 * z) + 2 * z * np.cos(z) - 1

    def f_z(z):
        return -2 * z * np.sin(z) + 3 * np.exp(3 * z) + 2 * np.cos(z)

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

    zeros = [
        0,
        1.18906588973011365517521756,
        1.72843498616506284043592924,
        3.01990732809571222812005354,
        4.03038191606046844562845941,
    ]
    zeros_multiplicities = [2, 1, 1, 1, 1]
    domain = Rectangle(complex(-5, -5), complex(5, 5))


class KVB144(Problem):
    def f(z):
        return z**2 * (z - 2) ** 2 * (z**3 + np.exp(2 * z) * np.cos(z) - np.sin(z) - 1)

    def f_z(z):
        return (
            z
            * (z - 2)
            * (
                z
                * (z - 2)
                * (
                    3 * z**2
                    - np.exp(2 * z) * np.sin(z)
                    + 2 * np.exp(2 * z) * np.cos(z)
                    - np.cos(z)
                )
                + 2 * z * (z**3 + np.exp(2 * z) * np.cos(z) - np.sin(z) - 1)
                + (2 * z - 4) * (z**3 + np.exp(2 * z) * np.cos(z) - np.sin(z) - 1)
            )
        )

    zeros = [
        -0.4607141197289707542294459477 - 0.6254277693477682516688207854j,
        -0.4607141197289707542294459477 + 0.6254277693477682516688207854j,
        0,
        2,
        1.66468286974551654134568653,
    ]
    zeros_multiplicities = [1, 1, 3, 2, 1]
    domain = Rectangle(complex(-3, -3), complex(3, 3))


class KVB145(Problem):
    def f(z):
        return (
            (z - 10)
            * (z - 9)
            * (z - 8)
            * (z - 7)
            * (z - 6)
            * (z - 5)
            * (z - 4)
            * (z - 3)
            * (z - 2)
            * (z - 1)
        )

    def f_z(z):
        return (
            10 * z**9
            - 495 * z**8
            + 10560 * z**7
            - 127050 * z**6
            + 946638 * z**5
            - 4510275 * z**4
            + 13667720 * z**3
            - 25228500 * z**2
            + 25507152 * z
            - 10628640
        )

    zeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    zeros_multiplicities = np.ones_like(zeros)
    domain = Rectangle(complex(-1, -1), complex(11, 1))


class KVB151(Problem):
    def f(z):
        return 2 * z * np.cos(z) + np.exp(3 * z) - 1

    def f_z(z):
        return -2 * z * np.sin(z) + 3 * np.exp(3 * z) + 2 * np.cos(z)

    zeros = [
        -1.84423395326221337491592440,
        0,
        0.5308949302929305274642203840 - 1.331791876751120981651544228j,
        0.5308949302929305274642203840 + 1.331791876751120981651544228j,
    ]
    zeros_multiplicities = [1, 1, 1, 1]
    domain = Rectangle(complex(-2, -2), complex(2, 3))


class KVB152(Problem):
    def f(z):
        return z * (z * (z - 4) * (z - 3) * (z - 2) * (z - 1) + np.sin(z))

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

    zeros = [
        0,
        1.18906588973011365517521756,
        1.72843498616506284043592924,
        3.01990732809571222812005354,
        4.03038191606046844562845941,
    ]
    zeros_multiplicities = [2, 1, 1, 1, 1]
    domain = Rectangle(complex(-0.5, -0.5), complex(5.5, 1.5))


class KVB153(Problem):
    def f(z):
        return z**2 * (z - 2) ** 2 * (z**3 + np.exp(2 * z) * np.cos(z) - np.sin(z) - 1)

    def f_z(z):
        return (
            z
            * (z - 2)
            * (
                z
                * (z - 2)
                * (
                    3 * z**2
                    - np.exp(2 * z) * np.sin(z)
                    + 2 * np.exp(2 * z) * np.cos(z)
                    - np.cos(z)
                )
                + 2 * z * (z**3 + np.exp(2 * z) * np.cos(z) - np.sin(z) - 1)
                + (2 * z - 4) * (z**3 + np.exp(2 * z) * np.cos(z) - np.sin(z) - 1)
            )
        )

    zeros = [
        0,
        2,
        1.66468286974551654134568653,
        -0.4607141197289707542294459477 - 0.6254277693477682516688207854j,
        -0.4607141197289707542294459477 + 0.6254277693477682516688207854j,
    ]
    zeros_multiplicities = [3, 2, 1, 1, 1]
    domain = Rectangle(complex(-1, -1), complex(3, 1))


class KVB331(Problem):
    def f(z):
        return z * np.sin(z) + 4 + np.exp(-3 * z) + 1 / (z**2 * (z - 1) * (z**2 + 9))

    def f_z(z):
        return (
            z * np.cos(z)
            + np.sin(z)
            - 3 * np.exp(-3 * z)
            - 5 / ((z - 1) ** 2 * (z**2 + 9) ** 2)
            + 4 / (z * (z - 1) ** 2 * (z**2 + 9) ** 2)
            - 27 / (z**2 * (z - 1) ** 2 * (z**2 + 9) ** 2)
            + 18 / (z**3 * (z - 1) ** 2 * (z**2 + 9) ** 2)
        )

    domain = Rectangle(complex(-1, -1), complex(1.1, 1.1))

    zeros = [0.97843635600921, 0.16974891913248, -0.13327146070751]
    zeros_multiplicities = [1, 1, 1]
    poles = [1, 0]
    poles_multiplicities = [1, 2]


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
    KVB144,
    KVB145,
    KVB151,
    KVB152,
    KVB153,
    KVB331,
)
without_known_roots_poles = (ExampleHolomorphic,)
all = with_known_roots_poles + without_known_roots_poles
