import math

import numpy as np
import scipy

__all__ = ["AAA", "derivative", "evaluate", "poles_residues", "zeros"]


def AAA(f, r, *, rtol=1e-13, max_iter=100, err_on_max_iter=True, initial_samples=16):
    """Compute a rational approximation of a function using the continuum AAA algorithm.

    Parameters
    ----------
    f : callable
        Function to approximate, should accept a 1D array and return a 1D array.
    r : Domain
        Domain on which to approximate the function.
    rtol : float, optional
        Relative tolerance of the approximation, by default 1e-13
    max_iter : int, optional
        Maximum number of iteration (degree of the rational approximation),
        by default 100.
    err_on_max_iter : bool, optional
        Whether to raise an error if `max_iter` is reached and but a relative accuracy
        of `rtol` has not been achieved, by default True.
    initial_samples : int, optional
        The initial number of samples of the function to take, by default 16. Subsequent
        initerations will take ``max(3, initial_samples - m)`` samples, where `m` is the
        number of iterations already performed.

    Returns
    -------
    z :
        Array of support points of the rational approximation.
    f :
        Values of the function at the support points.
    w :
        Barycentric weights of the rational approximation.
    """
    # Initial support points
    t = np.array([])
    S = np.array([])
    while (m := S.size) < max_iter:
        tm = _XS(t, max(3, initial_samples - m))
        X = r.sample_boundary(tm)
        fX = f(X)
        to_keep = (np.isfinite(fX)) & (~np.isnan(fX))
        X = X[to_keep]
        fX = fX[to_keep]

        if m == 0:
            R = np.mean(fX)
            fS = np.array([])
        else:
            fS = f(S)
            C = 1 / np.subtract.outer(X, S)
            A = np.subtract.outer(fX, fS) * C
            _, _, V = scipy.linalg.svd(
                A, full_matrices=(A.shape[0] <= A.shape[1]), check_finite=False
            )
            w = V.conj()[-1, :]
            R = (C @ (w * fS)) / (C @ w)

        err = np.linalg.norm(fX - R, ord=np.inf)
        fmax = np.linalg.norm(np.concat((fS, fX)), ord=np.inf)
        if err < rtol * fmax:
            return S, fS, w

        j = np.argmax(np.abs(fX - R))
        t = np.append(t, tm[j])
        S = np.append(S, X[j])
    if err_on_max_iter:
        msg = f"Iteration limit reached, current error {err}, require err {rtol * fmax}"
        raise RuntimeError(msg)
    return S[:-1], fS, w


def _XS(S, p):
    # make sure the end points 0 and 1 are included so we sample to the left of the
    # first support point and to the right of the last support point
    if 0 not in S:
        S = np.append(S, 0)
    if 1 not in S:
        S = np.append(S, 1)
    S = np.sort(S)
    d = np.arange(1, p + 1) / (p + 1)
    return (S[:-1] + np.multiply.outer(d, np.diff(S))).ravel()


def poles_residues(z, f, w, residue=False):
    """Compute the poles and residues of the rational function in barycentric form.

    Parameters
    ----------
    z : 1D array
        Support points of the rational function.
    f : 1D array
        Values of the rational function at the support points.
    w : 1D array
        Barycentric weights of the rational function.
    residue : bool, optional
        Whether to return the residue of the computed poles, by default False

    Returns
    -------
    poles : 1D array
        Poles of the rational function.
    residues : 1D array, optional
        Residues of the poles, if `residue` is True.
    """
    # poles
    m = w.size
    B = np.eye(m + 1, dtype=w.dtype)
    B[0, 0] = 0

    E = np.zeros_like(B, dtype=np.result_type(w, z))
    E[0, 1:] = w
    E[1:, 0] = 1
    np.fill_diagonal(E[1:, 1:], z)

    pol = scipy.linalg.eigvals(E, B)
    poles = pol[np.isfinite(pol)]

    if residue:
        # residue
        with np.errstate(divide="ignore", invalid="ignore"):
            N = (1 / (np.subtract.outer(poles, z))) @ (f * w)
            Ddiff = -((1 / np.subtract.outer(poles, z)) ** 2) @ w
            return poles, N / Ddiff
    return poles


def zeros(z, f, w):
    """Compute the zeros of the rational function in barycentric form.

    Parameters
    ----------
    z : 1D array
        Support points of the rational function.
    f : 1D array
        Values of the rational function at the support points.
    w : 1D array
        Barycentric weights of the rational function.

    Returns
    -------
    zeros : 1D array
        Zeros of the rational function.
    """
    # zeros
    m = w.size
    B = np.eye(m + 1, dtype=w.dtype)
    B[0, 0] = 0

    E = np.zeros_like(B, dtype=np.result_type(w, z, f))
    E[0, 1:] = w * f
    E[1:, 0] = 1
    np.fill_diagonal(E[1:, 1:], z)

    zeros = scipy.linalg.eigvals(E, B)
    return zeros[np.isfinite(zeros)]


def evaluate(z, f, w, Z):
    """Evaluate the rational function in barycentric form at points `Z`.

    Parameters
    ----------
    z : 1D array
        Support points of the rational function.
    f : 1D array
        Values of the rational function at the support points.
    w : 1D array
        Barycentric weights of the rational function.
    Z : ndarray
        Points at which to evaluate the rational function.

    Returns
    -------
    r : ndarray
        Values of the rational function at the points `Z`.
    """
    # evaluate rational function in barycentric form.
    Z = np.asarray(Z)
    zv = np.ravel(Z)

    # Cauchy matrix
    # Ignore errors due to inf/inf at support points, these will be fixed later
    with np.errstate(invalid="ignore", divide="ignore"):
        CC = 1 / np.subtract.outer(zv, z)
        # Vector of values
        r = CC @ (w * f) / (CC @ w)

    # Deal with input inf: `r(inf) = lim r(z) = sum(w*f) / sum(w)`
    if np.any(np.isinf(zv)):
        r[np.isinf(zv)] = np.sum(w * f) / np.sum(w)

    # Deal with NaN
    ii = np.nonzero(np.isnan(r))[0]
    for jj in ii:
        if np.isnan(zv[jj]) or not np.any(zv[jj] == z):
            # r(NaN) = NaN is fine.
            # The second case may happen if `r(zv[ii]) = 0/0` at some point.
            pass
        else:
            # Clean up values `NaN = inf/inf` at support points.
            # Find the corresponding node and set entry to correct value:
            r[jj] = f[zv[jj] == z].squeeze()

    return np.reshape(r, Z.shape)


def derivative(z, f, w, Z, *, k=1):
    """Evaluate the k-th derivative of the rational function in barycentric form at
    points `Z`.

    Parameters
    ----------
    z : 1D array
        Support points of the rational function.
    f : 1D array
        Values of the rational function at the support points.
    w : 1D array
        Barycentric weights of the rational function.
    Z : ndarray
        Points at which to evaluate the rational function.
    k : int, optional
        The order of the derivative to calculate, by default 1

    Returns
    -------
    out : ndarray
        Values of the k-th derivative of the rational function at the points `Z`.
    """
    Z = np.asarray(Z)

    if z.size <= 1:
        return np.zeros_like(Z)

    out = np.empty_like(Z)
    for i in np.ndindex(out.shape):
        out[i] = _derivative_scalar(z, f, w, Z[i], k)
    return out


def _derivative_scalar(zj, fj, wj, Z, k=1):
    diff = np.abs(Z - zj)
    if np.min(diff) > 0:
        V = wj / (Z - zj)
        gamma = V / np.sum(V)
        delta = fj
        phi = 0
        for _ in range(k + 1):
            phi = gamma @ delta
            delta = (delta - phi) / (zj - Z)
    else:
        j = np.argmin(diff)
        gamma = np.delete(-wj / wj[j], j)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.delete((fj - fj[j]) / (zj - zj[j]), j)
        zz = np.delete(zj, j)
        for _ in range(k):
            phi = gamma @ delta
            delta = (delta - phi) / (zz - zj[j])
    return math.factorial(k) * phi
