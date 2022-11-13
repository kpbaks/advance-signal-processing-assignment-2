import numpy as np
from dataclasses import dataclass
import itertools as it
import functools as ft

from filters.utils import window_iter, is_row_vector

@dataclass
class Result:
    y: np.ndarray
    err: np.ndarray
    weights: np.ndarray
    mse: np.ndarray


def rls(x: np.ndarray, d: np.ndarray, n_taps: int, w0: np.ndarray = None, forgetting_factor: float = 0.9) -> Result:
    
    delta = 0.1 # is a small positive constant
    # initial value of the inverse correlation matrix
    R0 = np.eye(n_taps) / delta

    R = R0


    assert is_row_vector(x), 'x must be a row vector'
    assert is_row_vector(d), 'd must be a row vector'
    assert x.shape == d.shape, 'x and d must have the same shape'
    assert n_taps > 0, 'ntaps must be positive'
    assert 0 < forgetting_factor <= 1, 'forgetting factor must be in (0, 1]'

    # initialize weights
    if w0 is None:
        w = np.zeros(n_taps)
    else:
        assert w0.shape == (n_taps,), 'w0 must have shape (ntaps,)'
        w = w0

    mu = forgetting_factor

    # Stability and Optimality Conditions for Recursive Least Squares
    # see: https://matousc89.github.io/padasip/sources/filters/rls.html#stability-and-optimal-performance

    # a window of size 2 will result in x.shape[0] - 2 + 1 iterations
    iterations: int = x.shape[0] - n_taps + 1
    # initialize output signal and error signal
    y = np.zeros(iterations)
    err = np.zeros(iterations)
    # weights are stored and returned, so the caller can inspect and plot them
    ws = np.zeros((iterations, n_taps))

    mse = np.zeros(iterations)


    for i, x_window in enumerate(window_iter(x, n_taps)):
        # 1. Compute gain vector

        # 2. filter
        y[i] = np.inner(w, x_window)

        # 3. error estimation
        err[i] = d[i] - y[i]

        mse[i] = np.mean(err[:i] ** 2 if i > 0 else err[0] ** 2)

        # 4. tap weight vector adaptation aka. weight update

        # 5. covariance matrix_window adaptation
        # \Psi_{\lambda}^{-1}(n) update
        R = 1/mu * R - (R @ np.outer(x_window,x_window) @ R) / (mu + np.inner(x_window, R @ x_window.T))

        delta_w = err[i] * R @ x_window.T
        w = w + delta_w

        ws[i] = w

    return Result(y, err, ws, mse)

# gamma # forgetting factor [0, 1)

# rls has a rapid rate of convergence compared to lms
# rls is computationally more expensive than lms
# RLS is less sensitive to eigenvalue spread than LMS


# use np.polyfit to fit a second order polynomial to the loss function
# to highlight that the error/misadjustment fluctuates around a mean value


def misadjustment(forgetting_factor: float, N: int) -> float:
    """Compute the misadjustment of the RLS filter."""
    return (1 - forgetting_factor) / (1 + forgetting_factor) * N
