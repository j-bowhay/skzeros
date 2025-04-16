import numpy as np
import scipy


def AAA(r, rtol=1e-12, max_terms=100, initial_points=25): ...


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
