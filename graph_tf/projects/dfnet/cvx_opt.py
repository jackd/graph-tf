import sys
from typing import Callable

import cvxpy
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def dfnets_coefficients_optimizer(
    mu: np.ndarray,
    response: Callable,
    kb: int,
    ka: int,
    radius: float = 0.85,
    verbose: bool = True,
):
    """
    Find polynomial coefficients (b, a).

    This function finds polynomial coefficients (b,a) such that
    the ARMA model "rARMA = polyval(wrev(b),mu)./polyval(wrev(a), mu)"
    approximates the function response() at the points mu.

    Arguments:
        mu: the points where the response function is evaluated.
        response: the desired response function
        kb, ka: the orders of the numerator and denominator respectively
        radius: allows to control the tradeoff between convergence speed (small)
            and accuracy (large). Should be below 1 if the standard iterative
            implementation is used. With the conj. gradient implementation any
            radius is allowed.
        verbose: controls verbosity

    Returns:
        b, a, rARMA, error
    """

    if mu.shape[0] == 1:
        mu = mu.conj().transpose()

    # -------------------------------------------------------------------------
    # Construct various utility matrices
    # -------------------------------------------------------------------------

    # N is the Vandermonde that will be used to evaluate the numerator.
    NM = np.zeros((len(mu), kb + 1))

    NM[:, 0] = 1

    for k in range(1, kb + 1):
        NM[:, k] = NM[:, k - 1] * mu

    # M is the Vandermonde that will be used to evaluate the denominator.
    MM = np.zeros((len(mu), ka))

    MM[:, 0] = mu

    for k in range(1, ka):
        MM[:, k] = MM[:, k - 1] * mu

    V = np.zeros((np.size(mu), ka))

    for k in range(0, ka):
        V[:, k] = mu ** k

    ia = cvxpy.Variable(shape=(ka, 1))
    ib = cvxpy.Variable(shape=(kb + 1, 1))

    response_mu = response(mu)
    reshape_mu = response_mu.reshape(-1, 1)

    objective = cvxpy.Minimize(
        cvxpy.norm(reshape_mu + np.diag(response_mu) @ MM @ ia - NM @ ib)
    )
    constraints = [cvxpy.norm((V @ ia), "inf") <= radius]

    prob = cvxpy.Problem(objective, constraints)
    result = prob.solve(verbose=verbose)
    del result

    a = ia.value
    b = ib.value

    # this is the achieved response.
    rARMA = np.polyval(np.flipud(b), mu) / np.polyval(np.flipud(a), mu)

    # error
    error = cvxpy.norm(rARMA - response_mu) / cvxpy.norm(mu)

    return b, a, rARMA, error
