from abc import ABC, abstractmethod

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
