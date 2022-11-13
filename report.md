---
author: "Kristoffer Plagborg Bak Sørensen"
date: "2022-11-13"
title: "Advanced Signal Processing - Assignment 2"
geometry: "margin=1in"
---

## Introduction

This report covers the topics of adaptive filtering and the LMS algorithm. The report is structured as follows: First, a brief overview of the theory behind the adaptive filtering is presented Then, the algorithms LMS, NMLS and RLS are presented. These algorihms are implemented in python and tested and compared on a sampled electrocardiogram (ECG) signal that contains colored noise from the powerline. Finally, the results are presented and discussed.

## Theory

Adaptive filters are FIR filters that are able to reconfigure themself based on the input signal. This is done by changing the filter coefficients. The filter coefficients are changed by an online quadratic programming algorithm that uses a known desired signal, $d[n]$, and the input signal, $x[n]$, to update the filter coefficients. The filter coefficients are updated in such a way that the filter output, $y[n]$, is as close as possible to the desired signal. This is done by minimizing the mean squared error (MSE) between the desired signal and the filter output. Adaptive filters differentiate themselves based on how they compute the update of the filter coefficients. This leads to different characteristics in terms of computational complexity, convergence speed and robustness to noise. The three most common algorithms are the LMS, NMLS and RLS algorithms. The LMS algorithm is the simplest of the three and is the most computationally efficient. The NMLS algorithm is a more advanced version of the LMS algorithm that is able to adapt to a changing input signal. The RLS algorithm is the most robust of the three and is able to adapt to a changing input signal and noise. The RLS algorithm is also the most computationally expensive of the three. 

The general structure of an adaptive filter is illustrated in Figure 1.  

![General block diagram for an adaptive filter algorithm](./img/adaptive-filter.excalidraw.png)




```python

def adaptive_filter(input_signal, desired_signal, n_taps, step_size):
    # Initialize filter coefficients
    # ...

    # loop over all samples

    
    output_signal = np.zeros(len(input_signal))
    for i in range(len(input_signal)):
        output_signal[i] = np.dot(filter_coefficients, input_signal[i])
        error = desired_signal[i] - output_signal[i]
        filter_coefficients = filter_coefficients + step_size * error * input_signal[i]
    return output_signal



# filter

# error

# update weights

```

### LMS (Least Mean Squares) Algorithm

$y(k) = w^T \cdot x(k)$

$e(k) = d(k) - y(k)$

$\Delta w(k) = \mu e(k) x(k)$

$w(k+1) = w(k) + \Delta w(k)$


suggest 2 taps, but with only 2 some noice is left, so 3 or more taps is better.
show zoomed in graph of a ECG peak, and demomstrates this.

5 is the smoothest

gradient descent


### Adaptive Line Enhancer


use a delayed version of the input signal as the input signal and the input signal as the desired signal.


### NMLS (Normalized LMS) Algorithm


$y(k) = w^T \cdot x(k)$

$e(k) = d(k) - y(k)$

$\Delta w(k) = \mu e(k) x(k) / \left\| x(k) \right\|^2$

$w(k+1) = w(k) + \Delta w(k)$




### RLS (Recursive Least Squares) Algorithm

Incorporates a forgetting factor, $\lambda$, that makes the algorithm more robust to noise. The forgetting factor is a value between 0 and 1 that determines how much the algorithm should forget about previous samples. A value of 1 means that the algorithm should not forget anything, while a value of 0 means that the algorithm should forget everything. The forgetting factor is used to calculate the inverse correlation matrix, $P^{-1}$, which is used to calculate the filter coefficients. The inverse correlation matrix is updated by the following equation:
 

$y(k) = w^T \cdot x(k)$

$e(k) = d(k) - y(k)$

$P(k) = \frac{1}{\lambda} P(k-1) - \frac{P(k-1) x(k) x(k)^T P(k-1)}{\lambda + x(k)^T P(k-1) x(k)}$

$\Delta w(k) = P(k) x(k) e(k)$



### Notch Filter

A notch filter is FIR filter that is used to remove a specific frequency $w_n$ from a signal. The filter is designed to have a frequency response that is 0 at the frequency that is to be removed. It can be thought of as a
combination of a low pass filter and a high pass filter, where the low pass filter has a cutoff frequency $w_l$ that is lower than the frequency that is to be removed and the high pass filter has a cutoff frequency $w_h$ that is higher than the frequency that is to be removed i.e. $w_l < w_n < w_h$.


---

## Problem Statement

> An ECG signal is sampled at a sampling frequency $Fs= 500\mathrm{Hz}$. The signals contains colored noise
at a frequency of circa $50\mathrm{Hz}$, which is the frequency of the powerline the equipment is connected to. The frequency of
the powergrid is not completely constant over time, which means that the noise is not guaranteed to be stationary. An adaptive filter
can be used to filter out the noise. The filter should be able to adapt to the changing noise.


## Hypothesis

LMS is sufficient, given that the signal is mostly stationary

RLS will be slower to compute, but will converge faster.


assume that 2 taps is sufficient to remove the noise, as the noise is colored and not white, and
only contain one frequency component.


## Methodology and Results

The sampled ECG signal is plotted in Figure 2. The ECG signal is not discernable from the noise.

<!-- 2 -->
![Sampled ECG signal with colored noise](./img/ecg_signal.png)


To use an adaptive algorithm to filter out the noise from the ECG signal, a desired signal of the noise is needed. The desired signal
is modeled as a sine wave with frequency $f_{noise}$

$$d = cos(2\pi \cdot f_{noise} / Fs \cdot t)$$

<!-- $$f_{noise} = 49.56\mathrm{Hz}$$ -->

The powerline frequency $f_{noise}$ is estimated using the power spectral density (PSD) of the sampled signal. The PSD is calculated using Welch's method. A plot of the PSD is shown in Figure 3. Most of the signals energy is contained in the noise component, so the single peak at $49.56\mathrm{Hz}$ corresponds to the frequency of the noise $f_{noise}$.

<!-- 3 -->
![Power spectral density of the ECG signal](./img/ecg_power_spectrum.png)


<!-- How many filter coeffcients are needed in the adaptive filter? -->

### How many filter coeffcients are needed in the adaptive filter?

Since the noise in the signal is colored at a single frequency $f_{noise}$ that varies slightly over time, it should be sufficient to have
**2** taps in the FIR filter. With 1 tap only the gain of the signal can be affected, as the filters system equation will only contain a scalar $H(z) = a$.With 2 taps both the gain and phase of the signal can be affected, as H(z) = a + bz^-1, which together is enough to attenuate the noise frequency $f_{noise}$.



### Select an appropriate value for the step-size $\mu$.

Equation 6.74 from the book, gives an analytical inequality that describes the interval for the LMS algorithm, given the parameter $mu$, where the algorithm remains stable. The inequality is given by

$$
0 < \mu < \frac{1}{3 \mathrm{tr}[R]}
$$

where $R$ is the autocorrelation matrix of the input signal $x(k)$.

Using python the upper bound of the inequaity was found to be $0.31$. The step-size $\mu$ was set to $0.3$.

### Using LMS with 2 taps, and $\mu = 0.3$.


![TODO finish ECG signal with LMS filter](./img/lms_filtered_ecg_signal_taps_2_mu_0.3.png)

![TODO](./img/lms_weights_over_time_and_against_each_other_taps_2_mu_0.3.png)

### Using SNR to determine the best combination of taps and $\mu$.

The signal to noise ratio (SNR) is a measure of the signal quality. The SNR is defined as the ratio between the power of the signal and the power of the noise. The SNR is given by the following equation:

$$
SNR = 10 \log_{10} \left( \frac{P_{signal}]}{P_{noise}} \right)
$$


![best snr](./img/lms_filtered_ecg_signal_best.png)



###  Adjust your parameters μ and the number of lter coecients and see how good results you can
acheive. If possible, estimate the improvement in signal-to-noise ratio.


### Computational Complexity of the Algorithms


| Algorithm | Computational Complexity |
|-----------|--------------------------|
| LMS       |                          |
| NMLS      |                          |
| RLS       |                          |




## Conclusion



footnotes