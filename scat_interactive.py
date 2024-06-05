import numpy as np
from scipy.signal import convolve
import sys
import os
import mercury as mr
import matplotlib.pyplot as plt

MIN_FLOAT = sys.float_info[3]

# Define utility functions

def get_mad(ts):
    return np.median(np.abs(ts - np.median(ts)))

def normalise(ts):
    return ts / (1.4826 * get_mad(ts))

def Gaussian1D(x, sig, x0):
    return np.exp(-(x - x0) * (x - x0) / (2 * sig * sig + MIN_FLOAT))

def linear(x, a, b):
    return a * x + b

def exp_decay(x, tau, x0):
    res = np.zeros(len(x)) + MIN_FLOAT
    res[x > x0] = np.exp(-(x[x > x0] - x0) / (tau + MIN_FLOAT))
    return res

def exp_gauss(x, x0, amp, sig, tau, eps):
    gx0 = np.mean(x)
    g = Gaussian1D(x, sig, gx0)
    ex = exp_decay(x, tau, x0)
    conv = convolve(g, ex, "same")
    conv /= np.max(conv) + MIN_FLOAT
    return amp * conv + eps

def exp_gauss_3(x, x1, amp1, sig1, tau1, x2, amp2, sig2, tau2, x3, amp3, sig3, tau3):
    g1 = exp_gauss(x, x1, amp1, sig1, tau1, 0)
    g2 = exp_gauss(x, x2, amp2, sig2, tau2, 0)
    g3 = exp_gauss(x, x3, amp3, sig3, tau3, 0)
    return g1 + g2 + g3

def lnlike(theta, x, y):
    model = exp_gauss_3(x, *theta)
    chisqr = -0.5 * (np.sum((y - model) ** 2))
    return chisqr

# Load data
y_data = np.load("./tseries.npy")
sampling_time, sampling_unit = 0.01024, "msec"
x = np.arange(len(y_data)) * sampling_time

# Define initial parameters
param_names = ['x1', 'amp1', 'sig1', 'tau1', 'x2', 'amp2', 'sig2', 'tau2', 'x3', 'amp3', 'sig3', 'tau3']
p0 = [5.0, 22., 0.08, 0.15, 6.15, 7., 0.08, 0.15, 6.9, 12., 0.08, 0.15]
lower_bounds = [4.85, 1, 0.001, 0.0001, 6.0, 1., 0.001, 0.0001, 6.75, 1., 0.001, 0.0001]
upper_bounds = [5.15, 30, 0.3, 0.6, 6.3, 20, 0.3, 0.6, 7.0, 20, 0.3, 0.6]

# Initial plot data
y = exp_gauss_3(x, *p0)

@mr.app
def interactive_fitter():
    # Define sliders for interactive control
    sliders = {
        name: mr.FloatSlider(p0[i], lower_bounds[i], upper_bounds[i], name=name)
        for i, name in enumerate(param_names)
    }

    # Define title input
    title = mr.Text("FRB181017", name="Plot Title")

    # Update function for sliders and title
    def update_plot():
        p = [sliders[name].value for name in param_names]
        y_new = exp_gauss_3(x, *p)
        lnlike_val = lnlike(p, x, y_data)

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(x, y_data, label="Data", color="blue")
        plt.plot(x, y_new, label="Model", color="red")
        plt.title(title.value)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(x, y_new - y_data, label="Residuals", color="black")
        plt.legend()

        plt.tight_layout()
        plt.show()

        return f"lnlikelihood: {lnlike_val:.3f}"

    # Layout of the app
    widgets = [title] + list(sliders.values())
    return mr.View(update_plot, widgets)

# Run the Mercury app
interactive_fitter()
