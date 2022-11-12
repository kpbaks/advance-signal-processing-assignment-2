import numpy as np
from dataclasses import dataclass
import itertools as it
import functools as ft

from utils import window_iter, is_row_vector

@dataclass
class Result:
    y: np.ndarray
    err: np.ndarray
    weights: np.ndarray
    mse: np.ndarray


def rls(x: np.ndarray, d: np.ndarray, mu: float, n_taps: int, w0: np.ndarray = None, psi: float = 0.1, forgetting_factor: float = 0.9) -> Result:
    
    delta = 0.1 # is a small positive constant
    # initial value of the inverse correlation matrix
    R0 = np.eye(n_taps) / delta


    assert 0 < forgetting_factor <= 1, 'forgetting factor must be in (0, 1]'

    mu = forgetting_factor

    # Stability and Optimality Conditions for Recursive Least Squares
    # see: https://matousc89.github.io/padasip/sources/filters/rls.html#stability-and-optimal-performance


    # 1. Compute gain vector

    # 2. filter

    y = np.inner(w, x)


    # 3. error estimation
    e = d - y

    # 4. tap weight vector adaptation aka. weight update

    # 5. covariance matrix adaptation
    # \Psi_{\lambda}^{-1}(n) update
    R = 1/mu * R - (R @ np.outer(x,x) @ R) / (mu + np.inner(x, R * x))

    delta_w = e * R * x
    w = w + delta_w



# gamma # forgetting factor [0, 1)

# rls has a rapid rate of convergence compared to lms
# rls is computationally more expensive than lms
# RLS is less sensitive to eigenvalue spread than LMS


# use np.polyfit to fit a second order polynomial to the loss function
# to highlight that the error/misadjustment fluctuates around a mean value


def misadjustment(forgetting_factor: float, N: int) -> float:
    """Compute the misadjustment of the RLS filter."""
    return (1 - forgetting_factor) / (1 + forgetting_factor) * N
