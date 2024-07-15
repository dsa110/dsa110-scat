import re
import bilby
import logging
logging.getLogger(bilby.__name__).setLevel(logging.CRITICAL)
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
    if np.abs(tau) >= 2.41e-5:
        gx0 = np.mean(x)
        g = Gaussian1D(x, sig, gx0)
        ex = exp_decay(x, tau, x0)
        conv = convolve(g, ex, "same")
    else: 
        conv = Gaussian1D(x, sig, x0)
    conv /= np.max(conv) + MIN_FLOAT
    return amp * conv

def exp_gauss_n(x, *params):
    y = np.zeros_like(x)
    n = len(params) // 4
    for i in range(n):
        x0, amp, sig, tau = params[4*i:4*(i+1)]
        y += exp_gauss(x, x0, amp, sig, tau)
    return y

def nested_sampling(timeseries, p0, comp_num, time_resolution, outdir, label, nlive=500, debug=False, time=None,plot=False, verbose=False, lower_bounds=None, upper_bounds=None,clean=True,sigma=None,resume=False,dlogz=0.1):

    if verbose:
        print('Time Resolution (ms):', time_resolution)
    if time is None:
        time = np.arange(len(timeseries)) * time_resolution
    if sigma is None:
        sigma = np.repeat(time_resolution, len(time))
    if verbose:
        print('Time:', time)
        print('Sigma:', sigma)

    if debug:
        fig = plt.figure()
        plt.errorbar(time, timeseries, yerr=sigma)
        plt.show()
    if verbose:
        print('Fitting Initiated')

    #label = str(npy_fil.split('/')[-1].split('_3')[0])
    #outdir = str(npy_fil.split('/')[-1].split('_3')[0]) + '_profilefit'

    if lower_bounds is None:
        lower_bounds = [(i - i / 2) for i in p0] #this can be modified
        lower_bounds = [round(i, 2) for i in lower_bounds]
    if upper_bounds is None:
        upper_bounds = [(i + i / 2) for i in p0] #this can be modified
        upper_bounds = [round(i, 2) + 1. for i in upper_bounds]
    if verbose:
        print('Lower:', lower_bounds)
        print('Upper:', upper_bounds)

    likeli = bilby.core.likelihood.GaussianLikelihood(time, timeseries, exp_gauss_n, sigma=sigma)
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
    if verbose:
        print('Sampler Running')
    if plot:
        fig = plt.figure(figsize=(18,12))
    result = bilby.run_sampler(likelihood=likeli, priors=prior, injection_parameters=injection_params,
                               sampler='dynesty', nlive=nlive, outdir=outdir, label=label, clean=clean,
                               print_progress=False,resume=resume,dlogz=dlogz)
    if plot:
        result.plot_corner(fig=fig)
    if verbose:
        print('Fit Complete!')
        print('Injection Parameters:', injection_params)
    return result
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
