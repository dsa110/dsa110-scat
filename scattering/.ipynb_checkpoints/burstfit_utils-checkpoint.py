import numpy as np
from scipy.ndimage import gaussian_filter1d

from burstfit import (
    FRBModel,
    FRBFitter,
    FRBParams,
    build_priors,
    plot_dynamic,
    goodness_of_fit
)

def estimate_smear_scatter(
    time: np.ndarray,
    prof: np.ndarray,
    freqs: np.ndarray,
    beta: float = -4.0,
    smooth_sigma: float = 1.0
):
    """Estimate zeta (FWHM) and tau_1ghz from the collapsed profile,
       with optional Gaussian smoothing (smooth_sigma in # of bins).
    """
    # 0. Smooth the profile if requested
    if smooth_sigma is not None and smooth_sigma > 0:
        prof_sm = gaussian_filter1d(prof, sigma=smooth_sigma)
    else:
        prof_sm = prof

    # 1. FWHM on the smoothed profile
    i0 = np.argmax(prof_sm)
    half = prof_sm[i0] / 2.0

    # find edges
    left_idxs  = np.where(prof_sm[:i0] < half)[0]
    right_idxs = np.where(prof_sm[i0:] < half)[0]

    if len(left_idxs) and len(right_idxs):
        t_left  = time[left_idxs[-1]]
        t_right = time[i0 + right_idxs[0]]
        zeta_est = t_right - t_left
    else:
        # fallback to 10% of total width if FWHM fails
        dt = time[1] - time[0]
        zeta_est = 0.1 * (time[-1] - time[0])

    # 2. Exponential tail fit on the smoothed profile
    tail_end = min(i0 + int(0.5 * len(prof_sm)), len(prof_sm))
    # only keep strictly positive values for log‐fit
    prof_tail = prof_sm[i0:tail_end]
    mask     = prof_tail > 0
    t_tail   = time[i0:tail_end][mask]
    if len(t_tail) < 2:
        # not enough points to fit an exponential tail
        tau_obs = zeta_est
    else:
        y_tail = np.log(prof_tail[mask])
        slope, _ = np.polyfit(t_tail, y_tail, 1)
        tau_obs = abs(1.0 / slope) if slope != 0 else zeta_est

    # linear fit in log-space: ln(y) = intercept + slope * t
    slope, _ = np.polyfit(t_tail, y_tail, 1)
    tau_obs = abs(1.0 / slope) if slope != 0 else zeta_est

    # scale to 1 GHz using a scattering index beta
    nu_med = np.median(freqs)
    tau1ghz_est = tau_obs * (nu_med / 1000.0) ** beta

    return float(zeta_est), float(tau1ghz_est)

def smart_initial_guess(self, model_key: str = "M3") -> FRBParams:
    """Use Nelder–Mead to maximize the posterior as a starting point."""
    # pack default guess
    p0 = self._guess()
    names = FRBFitter(self.model, None, n_steps=0)._ORDER[model_key]
    x0 = np.array([getattr(p0, n) for n in names], dtype=float)

    def neglnp(x):
        params = FRBParams.from_sequence(list(x), model_key)
        lp = self.model.log_prior(params, model_key)
        if not np.isfinite(lp):
            return np.inf
        ll = self.model.log_likelihood(params, model_key)
        return -(ll + lp)

    res = minimize(
        neglnp, x0,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol":1e-3, "fatol":1e-3}
    )
    if not res.success:
        warnings.warn(f"smart_initial_guess: optimizer failed: {res.message}")
    return FRBParams.from_sequence(list(res.x), model_key)
