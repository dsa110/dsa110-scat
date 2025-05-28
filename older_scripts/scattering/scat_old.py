# Perform FRB burst profile fitting using nested sampling.

import re
import bilby
import csv
import math
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy.signal as ss
from scipy.signal import convolve
from scipy.optimize import curve_fit
from lmfit.models import ExponentialGaussianModel, ExponentialModel, GaussianModel
from astropy import modeling
from astropy.modeling import models, fitting

MIN_FLOAT = sys.float_info[3]

# Formatting
font = {'family': 'serif', 'weight': 'normal', 'size': 14}
plt.rc('font', **font)

def boxcar_kernel(width):
    """
    Returns the boxcar kernel of given width normalized by sqrt(width) for S/N reasons.

    Parameters:
    width (int): Width of the boxcar.

    Returns:
    array: Boxcar of width `width` normalized by sqrt(width).
    """
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)

def find_burst(ts, width_factor=4, min_width=1, max_width=2048):
    """
    Find burst peak and width using boxcar convolution.

    Parameters:
    ts (array): Time-series.
    width_factor (int, optional): Windowing factor for on and off-pulse determination.
    min_width (int, optional): Minimum width to search from, in number of time samples.
    max_width (int, optional): Maximum width to search up to, in number of time samples.

    Returns:
    tuple: (peak, width, snr) where peak is the index of the burst peak, width is the width of the burst, and snr is the signal-to-noise ratio.
    """
    min_width = int(min_width)
    max_width = int(max_width)
    widths = list(range(min_width, min(max_width + 1, int((len(ts) - 50) // 6))))
    snrs = np.empty_like(widths, dtype=float)
    peaks = np.empty_like(widths, dtype=int)
    outer = 3 * width_factor // 2
    inner = width_factor // 2

    for i in range(len(widths)):
        convolved = ss.convolve(ts, boxcar_kernel(widths[i]))
        peaks[i] = np.nanargmax(convolved)
        if (peaks[i] > 0.999 * ts.shape[0]) or (peaks[i] < 0.001 * ts.shape[0]):
            snrs[i] = np.nan
        else:
            baseline = np.concatenate([convolved[0: max(0, peaks[i] - 3 * widths[i])],
                                       convolved[peaks[i] + 3 * widths[i]:]])
            if baseline.shape[0] > 50:
                rms = np.std(baseline)
            else:
                rms = np.nan
            snrs[i] = convolved[peaks[i]] / rms

    best_idx = np.nanargmax(snrs)
    return peaks[best_idx] - widths[best_idx] // 2, widths[best_idx], snrs[best_idx]

def Gaussian1D(x, sig, x0):
    """
    Returns 1D Gaussian curve.

    Parameters:
    x (array): Input data points.
    sig (float): Standard deviation of the Gaussian.
    x0 (float): Mean of the Gaussian.

    Returns:
    array: Gaussian curve.
    """
    return np.exp(-(x - x0) * (x - x0) / (2 * sig * sig + MIN_FLOAT))

def exp_decay(x, tau, x0):
    """
    Returns 1D one-sided exponential curve.

    Parameters:
    x (array): Input data points.
    tau (float): Decay constant.
    x0 (float): Starting point of the exponential decay.

    Returns:
    array: One-sided exponential curve.
    """
    res = np.zeros(len(x)) + MIN_FLOAT
    res[x > x0] = np.exp(-(x[x > x0] - x0) / (tau + MIN_FLOAT))
    return res

def exp_gauss(x, x0, amp, sig, tau):
    """
    Returns Gaussian convolved with a one-sided exponential.

    Parameters:
    x (array): Input data points.
    x0 (float): Mean of the Gaussian.
    amp (float): Amplitude of the Gaussian.
    sig (float): Standard deviation of the Gaussian.
    tau (float): Decay constant of the exponential.

    Returns:
    array: Convolution of Gaussian and one-sided exponential.
    """
    gx0 = np.mean(x)
    g = Gaussian1D(x, sig, gx0)
    ex = exp_decay(x, tau, x0)
    conv = convolve(g, ex, "same")
    conv /= np.max(conv) + MIN_FLOAT
    return amp * conv

def exp_gauss_1(x, x1, amp1, sig1, tau1):
    return exp_gauss(x, x1, amp1, sig1, tau1)

def exp_gauss_2(x, x1, amp1, sig1, tau1, x2, amp2, sig2, tau2):
    g1 = exp_gauss(x, x1, amp1, sig1, tau1)
    g2 = exp_gauss(x, x2, amp2, sig2, tau2)
    return g1 + g2

def exp_gauss_3(x, x1, amp1, sig1, tau1, x2, amp2, sig2, tau2, x3, amp3, sig3, tau3):
    g1 = exp_gauss(x, x1, amp1, sig1, tau1)
    g2 = exp_gauss(x, x2, amp2, sig2, tau2)
    g3 = exp_gauss(x, x3, amp3, sig3, tau3)
    return g1 + g2 + g3

def exp_gauss_4(x, x1, amp1, sig1, tau1, x2, amp2, sig2, tau2, x3, amp3, sig3, tau3, x4, amp4, sig4, tau4):
    g1 = exp_gauss(x, x1, amp1, sig1, tau1)
    g2 = exp_gauss(x, x2, amp2, sig2, tau2)
    g3 = exp_gauss(x, x3, amp3, sig3, tau3)
    g4 = exp_gauss(x, x4, amp4, sig4, tau4)
    return g1 + g2 + g3 + g4

def sub_npy(npy_fil, freq_subfactor=1, time_subfactor=1, bandwidth=400., center_frequency=800., file_duration=83.33):
    """
    Returns original numpy array, a sub-banded numpy array scrunched by a subfactor, and the array timeseries.

    Parameters:
    npy_fil (str): Path to the .npy file.
    freq_subfactor (int, optional): Frequency subfactor.
    time_subfactor (int, optional): Time subfactor.
    bandwidth (float, optional): Bandwidth in MHz.
    center_frequency (float, optional): Center frequency in MHz.
    file_duration (float, optional): Duration of the file in seconds.

    Returns:
    tuple: (original numpy array, sub-banded numpy array, array timeseries)
    """
    npy = np.load(npy_fil)
    npy_fsub = np.flipud(np.nanmean(npy.reshape(-1, int(freq_subfactor), int(npy.shape[1])), axis=1))
    timeseries = npy_fsub.sum(0)
    return npy, npy_fsub, timeseries

def nested_sampling(npy_fil, p0, comp_num, nlive=500, bandwidth=400., center_frequency=800., file_duration=83.33, subfactor=1., debug=False):
    """
    Perform nested sampling profile fitting with bilby.

    Parameters:
    npy_fil (str): Path to the .npy file.
    p0 (list): Initial parameter values.
    comp_num (int): Number of components in the model.
    nlive (int, optional): Number of live points for nested sampling.
    bandwidth (float, optional): Bandwidth in MHz.
    center_frequency (float, optional): Center frequency in MHz.
    file_duration (float, optional): Duration of the file in seconds.
    subfactor (float, optional): Subfactor for downsampling.
    debug (bool, optional): If True, displays a debug plot.
    """
    npy, npy_sub, timeseries = sub_npy(npy_fil, subfactor, file_duration, bandwidth, center_frequency)
    peaks, widths, snrs = find_burst(timeseries)
    time_res = file_duration / npy.shape[1]
    print('Raw Time Resolution (microsec):', time_res * 1e3)
    num_chan = npy.shape[0]
    freq_res = bandwidth / num_chan
    print('Raw Frequency Resolution (kHz):', freq_res * 1e3)
    window_left = int(peaks - 1 * widths)
    window_right = int(peaks + 1 * widths)
    
    if window_right - window_left <= 100:
        window_right += 20
        window_left -= 20

    sub_factor_time = 1
    y_data = (npy[:].sum(0) / np.max(npy[:].sum(0)))[window_left:window_right]
    sampling_time = (file_duration / npy.shape[1]) * sub_factor_time
    print('Sampling Time (ms):', sampling_time)
    time = np.arange(len(y_data)) * sampling_time
    sigma = np.repeat(sampling_time, len(time))
    print('Time:', time)
    print('Sigma:', sigma)

    if debug:
        fig = plt.figure()
        plt.errorbar(time, y_data, yerr=sigma)
        plt.show()

    print('Fitting Initiated')

    label = str(npy_fil.split('/')[-1].split('_3')[0])
    outdir = str(npy_fil.split('/')[-1].split('_3')[0]) + '_profilefit'

    if comp_num == '1':
        lower_bounds = [(i - i / 2) for i in p0[0:4]]
        upper_bounds = [(i + i / 2) for i in p0[0:4]]
        lower_bounds = [round(i, 2) for i in lower_bounds]
        upper_bounds = [round(i, 2) + 1. for i in upper_bounds]

        print('Lower:', lower_bounds)
        print('Upper:', upper_bounds)

        likeli = bilby.core.likelihood.GaussianLikelihood(time, y_data, exp_gauss_1, sigma=sigma)
        prior = dict(x1=bilby.core.prior.Uniform(lower_bounds[0], upper_bounds[0], 'x1'),
                     amp1=bilby.core.prior.Uniform(lower_bounds[1], upper_bounds[1], 'amp1'),
                     sig1=bilby.core.prior.Uniform(lower_bounds[2], upper_bounds[2], 'sig1'),
                     tau1=bilby.core.prior.Uniform(lower_bounds[3], upper_bounds[3], 'tau1'))
        injection_params = dict(x1=p0[0], amp1=p0[1], sig1=p0[2], tau1=p0[3])
        print('Sampler Running')

        result = bilby.run_sampler(likelihood=likeli, priors=prior, injection_parameters=injection_params,
                                   sampler='dynesty', nlive=500, outdir=outdir, label=label)
        result.plot_corner()
        print('Fit Complete!')

    elif comp_num == '2':
        lower_bounds = [(i - i / 2) for i in p0[0:8]]
        upper_bounds = [(i + i / 2) for i in p0[0:8]]
        lower_bounds = [round(i, 2) for i in lower_bounds]
        upper_bounds = [round(i, 2) + 1. for i in upper_bounds]

        print('Lower:', lower_bounds)
        print('Upper:', upper_bounds)

        likeli = bilby.core.likelihood.GaussianLikelihood(time, y_data, exp_gauss_2, sigma=sigma)
        prior = dict(x1=bilby.core.prior.Uniform(lower_bounds[0], upper_bounds[0], 'x1'),
                     amp1=bilby.core.prior.Uniform(lower_bounds[1], upper_bounds[1], 'amp1'),
                     sig1=bilby.core.prior.Uniform(lower_bounds[2], upper_bounds[2], 'sig1'),
                     tau1=bilby.core.prior.Uniform(lower_bounds[3], upper_bounds[3], 'tau1'),
                     x2=bilby.core.prior.Uniform(lower_bounds[4], upper_bounds[4], 'x2'),
                     amp2=bilby.core.prior.Uniform(lower_bounds[5], upper_bounds[5], 'amp2'),
                     sig2=bilby.core.prior.Uniform(lower_bounds[6], upper_bounds[6], 'sig2'),
                     tau2=bilby.core.prior.Uniform(lower_bounds[7], upper_bounds[7], 'tau2'))
        injection_params = dict(x1=p0[0], amp1=p0[1], sig1=p0[2], tau1=p0[3],
                                x2=p0[4], amp2=p0[5], sig2=p0[6], tau2=p0[7])
        print('Sampler Running')

        result = bilby.run_sampler(likelihood=likeli, priors=prior, injection_parameters=injection_params,
                                   sampler='dynesty', nlive=500, outdir=outdir, label=label)
        result.plot_corner()
        print('Fit Complete!')

    elif comp_num == '3':
        lower_bounds = [(i - i / 2) for i in p0[0:12]]
        upper_bounds = [(i + i / 2) for i in p0[0:12]]
        lower_bounds = [round(i, 2) for i in lower_bounds]
        upper_bounds = [round(i, 2) + 1. for i in upper_bounds]

        print('Lower:', lower_bounds)
        print('Upper:', upper_bounds)

        likeli = bilby.core.likelihood.GaussianLikelihood(time, y_data, exp_gauss_3, sigma=sigma)
        prior = dict(x1=bilby.core.prior.Uniform(lower_bounds[0], upper_bounds[0], 'x1'),
                     amp1=bilby.core.prior.Uniform(lower_bounds[1], upper_bounds[1], 'amp1'),
                     sig1=bilby.core.prior.Uniform(lower_bounds[2], upper_bounds[2], 'sig1'),
                     tau1=bilby.core.prior.Uniform(lower_bounds[3], upper_bounds[3], 'tau1'),
                     x2=bilby.core.prior.Uniform(lower_bounds[4], upper_bounds[4], 'x2'),
                     amp2=bilby.core.prior.Uniform(lower_bounds[5], upper_bounds[5], 'amp2'),
                     sig2=bilby.core.prior.Uniform(lower_bounds[6], upper_bounds[6], 'sig2'),
                     tau2=bilby.core.prior.Uniform(lower_bounds[7], upper_bounds[7], 'tau2'),
                     x3=bilby.core.prior.Uniform(lower_bounds[8], upper_bounds[8], 'x3'),
                     amp3=bilby.core.prior.Uniform(lower_bounds[9], upper_bounds[9], 'amp3'),
                     sig3=bilby.core.prior.Uniform(lower_bounds[10], upper_bounds[10], 'sig3'),
                     tau3=bilby.core.prior.Uniform(lower_bounds[11], upper_bounds[11], 'tau3'))
        injection_params = dict(x1=p0[0], amp1=p0[1], sig1=p0[2], tau1=p0[3],
                                x2=p0[4], amp2=p0[5], sig2=p0[6], tau2=p0[7],
                                x3=p0[8], amp3=p0[9], sig3=p0[10], tau3=p0[11])
        print('Sampler Running')

        result = bilby.run_sampler(likelihood=likeli, priors=prior, injection_parameters=injection_params,
                                   sampler='dynesty', nlive=500, outdir=outdir, label=label)
        result.plot_corner()
        print('Fit Complete!')

    elif comp_num == '4':
        lower_bounds = [(i - i / 2) for i in p0[0:16]]
        upper_bounds = [(i + i / 2) for i in p0[0:16]]
        lower_bounds = [round(i, 2) for i in lower_bounds]
        upper_bounds = [round(i, 2) + 1. for i in upper_bounds]

        print('Lower:', lower_bounds)
        print('Upper:', upper_bounds)

        likeli = bilby.core.likelihood.GaussianLikelihood(time, y_data, exp_gauss_4, sigma=sigma)
        prior = dict(x1=bilby.core.prior.Uniform(lower_bounds[0], upper_bounds[0], 'x1'),
                     amp1=bilby.core.prior.Uniform(lower_bounds[1], upper_bounds[1], 'amp1'),
                     sig1=bilby.core.prior.Uniform(lower_bounds[2], upper_bounds[2], 'sig1'),
                     tau1=bilby.core.prior.Uniform(lower_bounds[3], upper_bounds[3], 'tau1'),
                     x2=bilby.core.prior.Uniform(lower_bounds[4], upper_bounds[4], 'x2'),
                     amp2=bilby.core.prior.Uniform(lower_bounds[5], upper_bounds[5], 'amp2'),
                     sig2=bilby.core.prior.Uniform(lower_bounds[6], upper_bounds[6], 'sig2'),
                     tau2=bilby.core.prior.Uniform(lower_bounds[7], upper_bounds[7], 'tau2'),
                     x3=bilby.core.prior.Uniform(lower_bounds[8], upper_bounds[8], 'x3'),
                     amp3=bilby.core.prior.Uniform(lower_bounds[9], upper_bounds[9], 'amp3'),
                     sig3=bilby.core.prior.Uniform(lower_bounds[10], upper_bounds[10], 'sig3'),
                     tau3=bilby.core.prior.Uniform(lower_bounds[11], upper_bounds[11], 'tau3'),
                     x4=bilby.core.prior.Uniform(lower_bounds[12], upper_bounds[12], 'x4'),
                     amp4=bilby.core.prior.Uniform(lower_bounds[13], upper_bounds[13], 'amp4'),
                     sig4=bilby.core.prior.Uniform(lower_bounds[14], upper_bounds[14], 'sig4'),
                     tau4=bilby.core.prior.Uniform(lower_bounds[15], upper_bounds[15], 'tau4'))
        injection_params = dict(x1=p0[0], amp1=p0[1], sig1=p0[2], tau1=p0[3],
                                x2=p0[4], amp2=p0[5], sig2=p0[6], tau2=p0[7],
                                x3=p0[8], amp3=p0[9], sig3=p0[10], tau3=p0[11],
                                x4=p0[12], amp4=p0[13], sig4=p0[14], tau4=p0[15])
        print('Sampler Running')

        result = bilby.run_sampler(likelihood=likeli, priors=prior, injection_parameters=injection_params,
                                   sampler='dynesty', nlive=500, outdir=outdir, label=label)
        result.plot_corner()
        print('Fit Complete!')

    else:
        print('No Burst Component Number Specified -- Please Identify and Define Number of Sub-Bursts')

    print('Injection Parameters:', injection_params)

#if __name__ == '__main__':
#    npy_fil = sys.argv[1]
#    injection_csv = sys.argv[2]
#    comp_num = sys.argv[3]
#    file_dur = sys.argv[4]

#    cwd = os.getcwd()
#    npy_file = str(cwd) + '/' + str(npy_fil)
#    print('Numpy File Path:', npy_file)

#    with open(str(cwd) + '/' + str(injection_csv)) as csv_file:
#        reader = csv.reader(csv_file)
#        injection_dict = dict(reader)

#    p0_str = injection_dict[npy_file.split('/')[-1].split('_3')[0]]
#    p0 = [float(i) for i in p0_str.split('[')[1].split(']')[0].split(',')]
#    print('P0:', p0)
#    print('P0 Shape:', len(p0))

#    nested_sampling(npy_file, list(p0), str(comp_num), file_duration=float(file_dur))
