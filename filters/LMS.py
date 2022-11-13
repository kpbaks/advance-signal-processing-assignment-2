import numpy as np
from dataclasses import dataclass

from filters.utils import window_iter, is_row_vector

@dataclass
class Result:
    y: np.ndarray
    err: np.ndarray
    w: np.ndarray
    mse: np.ndarray # mean squared error




# desired signal is ecg
def lms(x: np.ndarray, d: np.ndarray, mu: float, ntaps: int, w0: np.ndarray = None) -> Result:
    """LMS filter.

    x: input signal, shape (N,)
    d: desired signal, shape (N,)
    mu: step size, mu > 0
    ntaps: number of taps, ntaps > 0
    w0: initial weights (optional), shape (ntaps,)
    """

    assert is_row_vector(x), 'x must be a row vector'
    assert is_row_vector(d), 'd must be a row vector'
    assert x.shape == d.shape, 'x and d must have the same shape'
    assert ntaps > 0, 'ntaps must be positive'
    assert mu > 0, 'mu must be positive'

    # initialize weights
    if w0 is None:
        w = np.zeros(ntaps)
    else:
        assert w0.shape == (ntaps,), 'w0 must have shape (ntaps,)'
        w = w0

    # a window of size 2 will result in x.shape[0] - 2 + 1 iterations
    iterations: int = x.shape[0] - ntaps + 1
    # initialize output signal and error signal
    y = np.zeros(iterations)
    err = np.zeros(iterations)
    # weights are stored and returned, so the caller can inspect and plot them
    ws = np.zeros((iterations, ntaps))

    mse = np.zeros(iterations)

    for i, x_window in enumerate(window_iter(x, ntaps)):
        # compute output
        y[i] = np.dot(w, x_window)
        # compute error and save it
        err[i] = d[i] - y[i] # when y is equal to x, then we will get 

        # compute mean squared error and save it
        mse[i] = np.mean(err[:i] ** 2)

        # update weights        
        w += mu * 2 * err[i] * x_window
        ws[i] = w
        
        
    return Result(y, err, ws, mse)
    