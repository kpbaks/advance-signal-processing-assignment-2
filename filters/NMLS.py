"""
Implementation of the NMLS algorithm.

Classes:
    Result: a named tuple containing the output signal, the error signal and the final weights

Functions:
    nmls(x, d, mu, ntaps, w0=None) -> Result
"""

import numpy as np
from dataclasses import dataclass

from filters.utils import window_iter, is_row_vector

@dataclass
class Result:
    y: np.ndarray
    err: np.ndarray
    weights: np.ndarray
    mse: np.ndarray # mean squared error



def nlms(x: np.ndarray, d: np.ndarray, mu: float, n_taps: int, w0: np.ndarray = None, psi: float = 0.1) -> Result:
    """NLMS filter.

    x: input signal (row vector), shape (N,)
    d: desired signal (row vector), shape (N,)
    mu: step size (mu > 0) (hyperparameter)
    n_taps: number of taps (n_taps > 0)
    w0: initial weights (optional)
    psi: regularization factor (psi > 0) (optional) (hyperparameter)
    """

    assert is_row_vector(x), 'x must be a row vector'
    assert is_row_vector(d), 'd must be a row vector'
    assert x.shape == d.shape, 'x and d must have the same shape'
    assert n_taps > 0, 'ntaps must be positive'
    assert mu > 0, 'mu must be positive'

    # initialize weights
    if w0 is None:
        w = np.zeros(n_taps)
    else:
        assert w0.shape == (n_taps,), 'w0 must have shape (ntaps,)'
        w = w0

    n_samples: int = x.shape[0]

    # a window of size 2 will result in n_samples - 2 + 1 iterations
    iterations: int = n_samples - n_taps + 1
    # initialize output signal and error signal
    y = np.zeros(iterations, dtype=np.float64)
    err = np.zeros(iterations, dtype=np.float64)
    # weights are stored and returned, so the caller can inspect and plot them
    ws = np.zeros((iterations, n_taps), dtype=np.float64)

    # mse (mean squared error) is the mean of the squared error signal
    mse = np.zeros(iterations, dtype=np.float64)

    for i, x_window in enumerate(window_iter(x, n_taps)):
        # compute output
        y[i] = np.dot(w, x_window)
        # compute error and save it
        err[i] = d[i] - y[i]
        # compute mean squared error and save it
        mse[i] = np.mean(err[:i] ** 2)
        # update weights
        # NOTE: this is where nlms differs from lms
        # the step size is divided by the norm of the input signal
        # this is to prevent the step size from growing too large
        # psi is a small constant to prevent division by zero
        normalization_factor: float = mu / (np.linalg.norm(x_window) ** 2 + psi)

        w = w + normalization_factor * err[i] * x_window
        ws[i] = w

    return Result(y, err, ws, mse)