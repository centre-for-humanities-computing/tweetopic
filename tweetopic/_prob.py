import numpy as np
from numba import njit


@njit(fastmath=True)
def sample_categorical(pvals: np.ndarray) -> int:
    """Samples from a categorical distribution given its parameters.

    Parameters
    ----------
    pvals: array of shape (n_clusters, )
        Parameters of the categorical distribution.

    Returns
    -------
    int
        Sample.
    """
    # NOTE: This function was needed as numba's implementation
    # of numpy's multinomial sampling function has some floating point shenanigans going on.
    # Rejection sampling with cummulutative probabilities :)
    cum_prob = 0
    u = np.random.uniform(0.0, 1.0)
    for i in range(len(pvals)):
        cum_prob += pvals[i]
        if u < cum_prob:
            return i
    else:
        # This shouldn't ever happen, but floating point errors can
        # cause such behaviour ever so often.
        return 0


@njit
def norm_prob(prob: np.ndarray) -> None:
    """Normalizes probabilities in place.

    Parameters
    ----------
    prob: ndarray
        Improper probability distribution.
    """
    (n,) = prob.shape
    total = np.sum(prob)
    if total == 0:
        prob[:] = 1 / n
        return
    for i in range(n):
        prob[i] = prob[i] / total
