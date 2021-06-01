import numpy as np


def compute_cumulants(G, mus, R=None, return_R=False):
    """
    Compute the cumulants of a Hawkes process given the integrated kernel
    matrix `G` and the baseline rate vector `mus`

    Arguments
    ---------
    G : np.ndarray
        The integrated kernel matrix of shape shape dim x dim
    mus : np.ndarray
        The baseline rate vector of shape dim
    R : np.ndarray (optional)
        Precomputed matrix R
    return_R : bool (optional)
        Return the matrix R if set to `True`

    Return
    ------
    L : np.ndarray
        Mean intensity matrix
    C : np.ndarray
        Covariance matrix
    Kc : np.ndarray
        Skewness matrix
    R : np.ndarray (returned only if `return_R` is True)
        Internal matrix to compute the cumulants
    """
    if not len(G.shape) == 2:
        raise ValueError("Matrix `G` should be 2-dimensional")
    if not len(mus.shape) == 1:
        raise ValueError("Vector `mus` should be 1-dimensional")
    if not G.shape[0] == G.shape[1]:
        raise ValueError("Matrix `G` should be a squared matrix")
    if not G.shape[0] == mus.shape[0]:
        raise ValueError("Vector `mus` should have the same dinension as `G`")
    R = np.linalg.inv(np.eye(len(G)) - G)
    L = np.diag(R @ mus)
    C = R @ L @ R.T
    Kc = (R**2) @ C.T + 2 * R * (C - R @ L) @ R.T
    if return_R:
        return L, C, Kc, R
    return L, C, Kc
