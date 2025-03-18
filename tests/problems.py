from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from scikit_poles_zeros._domain import Rectangle

jax.config.update("jax_enable_x64", True)


class Problem(ABC):
    def __init__(self):
        self.jac = jax.jacobian(self._f, holomorphic=True)
        self.hess = jax.hessian(self._f, holomorphic=True)

    @abstractmethod
    def _f(self, z): ...

    def f(self, z):
        return np.asarray(self._f(z))

    def f_z(self, z):
        return np.diag(self.jac(jnp.atleast_1d(jnp.asarray(z, dtype=jnp.complex128))))

    @property
    @abstractmethod
    def domain(self): ...


class ExampleHolomorphic(Problem):
    """From Locating all the zeros of an analytic function in one complex variable,
    Michael Dellnitza, Oliver Sch,utzea, Qinghua Zheng, Section 4.1"""

    def _f(self, z):
        return z**50 + z**12 - 5 * jnp.sin(20 * z) * jnp.cos(12 * z) - 1

    domain = Rectangle(complex(-20.3, -20.3), complex(20.7, 20.7))
