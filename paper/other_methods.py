import numpy as np
import scipy


def delves_lyness(f, f_z, domain, N=None, quad_kwargs=None):
    if N is None:
        N = round(
            domain.argument_principle(f, f_z, quadrature_args=quad_kwargs).integral.real
        )
    print(N)
    if N == 0:
        return np.array([])
    s = np.empty(N + 1, dtype=np.complex128)
    s[0] = N

    for i in range(1, N + 1):
        s[i] = domain.argument_principle(
            f, lambda z, i=i: (z**i) * f_z(z), quadrature_args=quad_kwargs
        ).integral

    coeffs = np.ones(N + 1, dtype=np.complex128)

    # Newton coefficients
    coeffs = np.zeros(N + 1, dtype=np.complex128)
    coeffs[0] = 1.0

    for k in range(1, N + 1):
        acc = s[k]
        for j in range(1, k):
            acc += coeffs[j] * s[k - j]
        coeffs[k] = -acc / k

    return np.roots(coeffs)


def kravanja_van_barel(f, f_z, domain, N=None, quad_kwargs=None):
    if N is None:
        N = round(
            domain.argument_principle(f, f_z, quadrature_args=quad_kwargs).integral.real
        )

    s = np.empty(2 * N, dtype=np.complex128)
    s[0] = N

    for i in range(1, 2 * N):
        s[i] = domain.argument_principle(
            f, lambda z, i=i: (z**i) * f_z(z), quadrature_args=quad_kwargs
        ).integral

    H_N = scipy.linalg.hankel(s[:N], s[N - 1 : -1])
    n = np.linalg.matrix_rank(H_N)

    H_N = scipy.linalg.hankel(s[:n], s[n - 1 : 2 * n - 1])
    H_N_reduced = scipy.linalg.hankel(s[1 : n + 1], s[n : 2 * n])

    zeros = scipy.linalg.eigvals(H_N_reduced, H_N)
    multiplicities = np.linalg.solve(np.vander(zeros, increasing=True).T, s[:n])
    return zeros, multiplicities
