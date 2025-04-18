import numpy as np
import scipy


def AAA(f, r, rtol=1e-13):
    # Initial support points
    S = r.sample_boundary(
        [0, 0.25, 0.5, 0.75]
    )  # Corners of the square, TODO generalise
    while True:
        m = S.size
        X = XS(S, max(3, 32 - m))
        C = 1 / np.subtract.outer(X, S)
        fS = f(S)
        fX = f(X)
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
        S = np.append(S, X[j])


def XS(S, p):
    S = np.sort(S)
    d = np.arange(1, p + 1) / (p + 1)
    return (S[:-1] + np.multiply.outer(d, np.diff(S))).ravel()


def poles_residues(z, f, w, residue=False):
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
    # evaluate rational function in barycentric form.
    Z = np.asarray(Z)
    zv = np.ravel(Z)

    weights = w[..., np.newaxis]

    # Cauchy matrix
    # Ignore errors due to inf/inf at support points, these will be fixed later
    with np.errstate(invalid="ignore", divide="ignore"):
        CC = 1 / np.subtract.outer(zv, z)
        # Vector of values
        r = CC @ (weights * f) / (CC @ weights)

    # Deal with input inf: `r(inf) = lim r(z) = sum(w*f) / sum(w)`
    if np.any(np.isinf(zv)):
        r[np.isinf(zv)] = np.sum(weights * f) / np.sum(weights)

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
