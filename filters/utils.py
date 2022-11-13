import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def signaltonoise(a, axis=0, ddof=0):
    """
    Signal to noise ratio. (equivalent to scipy.stats.signaltonoise)
    :param a: array_like
    :param axis: axis along which the standard deviation is computed, default is 0
    :param ddof: degrees of freedom
    :return: the signal to noise ratio
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def SNR(signal, noise) -> float:
    """
    Compute the signal to noise ratio (SNR) of a signal.
    :param signal: the signal
    :param noise: the noise
    :return: the SNR in dB
    """
    return 10 * np.log10(np.sum(signal**2) / np.sum(noise**2))


def autocorr(x: np.ndarray, biased: bool = True) -> np.ndarray:
    """
    Compute the autocorrelation of a signal.
    :param x: signal
    :return: autocorrelation
    """

    

    return [
        np.sum(x[:len(x) - k] * x[k:]) / len(x) if biased else np.sum(x[:len(x) - k] * x[k:]) / (len(x) - k)
        for k in range(len(x))
    ]



# def autocorr(x: np.ndarray, normalize=True) -> np.ndarray:
#     if normalize:
#         x = (x - np.mean(x)) / np.std(x)

#     result = np.correlate(x, x, mode='full')
#     return result[result.size // 2:]
    


def is_row_vector(x: np.ndarray) -> bool:
    return x.ndim == 1 and x.shape[0] > 1

def is_column_vector(x: np.ndarray) -> bool:
    return x.ndim == 2 and x.shape[0] > 1 and x.shape[1] == 1



# numpy window iterator
def window_iter(x: np.ndarray, window_size: int, step_size: int = 1) -> np.ndarray:
    """
    Iterate over a numpy array with a sliding window of a given size.
    :param x: numpy array
    :param window_size: size of the window
    :param step_size: step size (default: 1)
    :return: iterator over the array

    example:
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> for window in window_iter(x, 3, 2):
    >>>     print(window)
    [1 2 3]
    [3 4 5]
    [5 6 7]
    [7 8 9]
    """
    for i in range(0, len(x) - window_size + 1, step_size):
        yield x[i:i + window_size]