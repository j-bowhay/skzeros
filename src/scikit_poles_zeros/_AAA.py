import numpy as np
import scipy


def AAA(r, rtol=1e-12, max_terms=100): ...


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
