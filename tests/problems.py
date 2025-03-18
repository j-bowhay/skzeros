from abc import ABC, abstractmethod

from autograd import elementwise_grad
from autograd import numpy as np

from scikit_poles_zeros._domain import Rectangle


class Problem(ABC):
    @abstractmethod
    def f(self, z): ...

    f_z = elementwise_grad(f)
    f_zz = elementwise_grad(f_z)

    @property
    @abstractmethod
    def domain(self): ...


class ExampleHolomorphic(Problem):
    """From Locating all the zeros of an analytic function in one complex variable,
    Michael Dellnitza, Oliver Sch,utzea, Qinghua Zheng, Section 4.1"""

    def f(self, z):
        return z**50 + z**12 - 5 * np.sin(20 * z) * np.cos(12 * z) - 1

    domain = Rectangle(complex(-20.3, -20.3), complex(20.7, 20, 7))
