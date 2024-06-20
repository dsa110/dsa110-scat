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
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)

def find_burst(ts, width_factor=4, min_width=1, max_width=2048):
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
    return np.exp(-(x - x0) * (x - x0) / (2 * sig * sig + MIN_FLOAT))

def exp_decay(x, tau, x0):
    res = np.zeros(len(x)) + MIN_FLOAT
    res[x > x0] = np.exp(-(x[x > x0] - x0) / (tau + MIN_FLOAT))
    return res

def exp_gauss(x, x0, amp, sig, tau):
    gx0 = np.mean(x)
    g = Gaussian1D(x, sig, gx0)
    ex = exp_decay(x, tau, x0)
    conv = convolve(g, ex, "same")
    conv /= np.max(conv) + MIN_FLOAT
    return amp * conv

def exp_gauss_n(x, *params):
    y = np.zeros_like(x)
    n = len(params) // 4
    for i in range(n):
        x0, amp, sig, tau = params[4*i:4*(i+1)]
        y += exp_gauss(x, x0, amp, sig, tau)
    return y

def sub_npy(npy_fil, freq_subfactor=1, time_subfactor=1, bandwidth=400., center_frequency=800., file_duration=83.33):
    npy = np.load(npy_fil)
    npy_fsub = np.flipud(np.nanmean(npy.reshape(-1, int(freq_subfactor), int(npy.shape[1])), axis=1))
    timeseries = npy_fsub.sum(0)
    return npy, npy_fsub, timeseries

def nested_sampling(npy_fil, p0, comp_num, nlive=500, bandwidth=400., center_frequency=800., file_duration=83.33, subfactor=1., debug=False):
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

    lower_bounds = [(i - i / 2) for i in p0]
    upper_bounds = [(i + i / 2) for i in p0]
    lower_bounds = [round(i, 2) for i in lower_bounds]
    upper_bounds = [round(i, 2) + 1. for i in upper_bounds]

    print('Lower:', lower_bounds)
    print('Upper:', upper_bounds)

    likeli = bilby.core.likelihood.GaussianLikelihood(time, y_data, exp_gauss_n, sigma=sigma)
    prior = {}
    injection_params = {}
    for i in range(comp_num):
        prior[f'x{i+1}'] = bilby.core.prior.Uniform(lower_bounds[4*i], upper_bounds[4*i], f'x{i+1}')
        prior[f'amp{i+1}'] = bilby.core.prior.Uniform(lower_bounds[4*i+1], upper_bounds[4*i+1], f'amp{i+1}')
        prior[f'sig{i+1}'] = bilby.core.prior.Uniform(lower_bounds[4*i+2], upper_bounds[4*i+2], f'sig{i+1}')
        prior[f'tau{i+1}'] = bilby.core.prior.Uniform(lower_bounds[4*i+3], upper_bounds[4*i+3], f'tau{i+1}')
        injection_params[f'x{i+1}'] = p0[4*i]
        injection_params[f'amp{i+1}'] = p0[4*i+1]
        injection_params[f'sig{i+1}'] = p0[4*i+2]
        injection_params[f'tau{i+1}'] = p0[4*i+3]

    print('Sampler Running')

    result = bilby.run_sampler(likelihood=likeli, priors=prior, injection_parameters=injection_params,
                               sampler='dynesty', nlive=500, outdir=outdir, label=label)
    result.plot_corner()
    print('Fit Complete!')
    print('Injection Parameters:', injection_params)

#if __name__ == '__main__':
#    npy_fil = sys.argv[1]
#    injection_csv = sys.argv[2]
#    comp_num = int(sys.argv[3])
#    file_dur = float(sys.argv[4])

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

#    nested_sampling(npy_file, list(p0), comp_num, file_duration=file_dur)
