import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


# signal to noise ratio
def snr(x: np.ndarray, x_ref: np.ndarray) -> float:
    """
    Compute the signal to noise ratio of a signal.
    :param x: signal
    :param x_ref: reference signal
    :return: signal to noise ratio
    """
    return 10 * np.log10(np.sum(x_ref ** 2) / np.sum((x - x_ref) ** 2))

# snr = 10 * np.log10(np.sum(ecg**2) / np.sum(err**2))



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