import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lmfit import Model, minimize, Parameters, fit_report

def doublelorentz(x, gamma1, m1, gamma2, m2, c):
    """
    Double Lorentzian function.

    Parameters:
    x (array): Independent variable.
    gamma1 (float): Width parameter for the first Lorentzian.
    m1 (float): Amplitude parameter for the first Lorentzian.
    gamma2 (float): Width parameter for the second Lorentzian.
    m2 (float): Amplitude parameter for the second Lorentzian.
    c (float): Constant offset.

    Returns:
    array: Result of the double Lorentzian function.
    """
    return m1**2 / (1 + (x / gamma1)**2) + m2**2 / (1 + (x / gamma2)**2) + c

def lorentz(x, gamma1, m1, c):
    """
    Single Lorentzian function.

    Parameters:
    x (array): Independent variable.
    gamma1 (float): Width parameter for the Lorentzian.
    m1 (float): Amplitude parameter for the Lorentzian.
    c (float): Constant offset.

    Returns:
    array: Result of the Lorentzian function.
    """
    return m1**2 / (1 + (x / gamma1)**2) + c

def shift(v, i, nchan):
    """
    Shifts the array v by an index i, accounting for negative lag.

    Parameters:
    v (array): Array to be shifted.
    i (int): Shift index.
    nchan (int): Number of frequency channels to account for negative lag.

    Returns:
    array: Shifted array.
    """
    n = len(v)
    r = np.zeros(3 * n)
    i += nchan - 1  # To account for negative lag
    i = int(i)
    r[i:i + n] = v
    return r

def autocorr(x, v=None, zerolag=False, maxlag=None):
    """
    Computes the autocorrelation function (ACF) of the 1D array x.

    Parameters:
    x (array): 1D array to autocorrelate.
    v (array, optional): Mask array with 1s (no mask) and 0s (mask).
    zerolag (bool, optional): If True, includes the zero lag noise spike.
    maxlag (int, optional): Maximum lag to compute the ACF. If None, computes for the entire length of x.

    Returns:
    array: Autocorrelation function of x.
    """
    nchan = len(x)
    if v is None:
        v = np.ones_like(x)
    x = x.copy()
    x[v != 0] -= x[v != 0].mean()
    if maxlag is None:
        ACF = np.zeros_like(x)
    else:
        ACF = np.zeros_like(x)[:int(maxlag)]
    
    #print(maxlag)
    #print('ACF length:', len(ACF))
    
    for i in tqdm(range(len(ACF))):
        if not zerolag:
            if i > 1:
                m = shift(v, 0, nchan) * shift(v, i, nchan)
                ACF[i - 1] = np.sum(shift(x, 0, nchan) * shift(x, i, nchan) * m) / \
                             np.sqrt(np.sum(shift(x, 0, nchan)**2 * m) * np.sum(shift(x, i, nchan)**2 * m))
        else:
            m = shift(v, 0, nchan) * shift(v, i, nchan)
            ACF[i] = np.sum(shift(x, 0, nchan) * shift(x, i, nchan) * m) / \
                     np.sqrt(np.sum(shift(x, 0, nchan)**2 * m) * np.sum(shift(x, i, nchan)**2 * m))
    
    return ACF

# Example usage (commented out):
# t_res = 2.56  # Time resolution
# lagrange_for_fit = 100  # Range for fitting the ACF

# Load profile data
# prof = np.load('./37888771_sb1.npy')

# Compute the autocorrelation function (ACF)
# acf = autocorr(prof)

# Create lags array
# lags = np.arange(len(acf)) + 1
# acf = acf[1:]
# lags = lags[1:]

# Create symmetric ACF and lags for fitting
# acf = np.concatenate((acf[::-1], acf))
# lags = np.concatenate((-1 * lags[::-1], lags)) * t_res

# Fit the ACF using a Lorentzian model
# gmodel = Model(lorentz)
# acf_for_fit = acf[int(len(acf) / 2.) - int(lagrange_for_fit / t_res) : int(len(acf) / 2.) + int(lagrange_for_fit / t_res)]
# lags_for_fit = lags[int(len(acf) / 2.) - int(lagrange_for_fit / t_res) : int(len(acf) / 2.) + int(lagrange_for_fit / t_res)]

# Perform the fit
# result = gmodel.fit(acf_for_fit, x = lags_for_fit, gamma1 = 10, m1 = 1, c = 0)
# Print the fit report
# print(result.fit_report())
