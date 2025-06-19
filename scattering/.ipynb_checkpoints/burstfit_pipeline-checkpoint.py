"""
burstfit_pipeline.py
====================

Object‑oriented **orchestrator** that glues together all core modules
(`burstfit.py`, `burstfit_modelselect.py`, `burstfit_robust.py`) and
reads telescope‑specific constants from `telescopes.yaml` via
`config_utils.load_telescope()`.

Typical use
-----------
```
python
from burstfit_pipeline import BurstPipeline
pipe = BurstPipeline("burst.npy", telescope="CHIME", pool=6)
result = pipe.run_full(model_scan=True, diagnostics=True)
print(result["best_params"])
```
At the command line:
```bash
python burstfit_pipeline.py burst.npy --telescope DSA-110 --plot
```

Dependencies
------------
* PyYAML (for `config_utils`)
* NumPy, Matplotlib, emcee (pulled in by `burstfit`)
"""
from __future__ import annotations

import logging
import warnings
import pickle
import contextlib
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from burstfit import (
    FRBModel,
    FRBFitter,
    FRBParams,
    build_priors,
    plot_dynamic,
)
from burstfit_modelselect import fit_models_bic
from burstfit_robust import (
    subband_consistency,
    leave_one_out_influence,
    plot_influence,
)

from burstfit_utils import estimate_smear_scatter, smart_initial_guess

from config_utils import load_telescope
from pool_utils import build_pool

log = logging.getLogger("burstfit.pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

###############################################################################
# 1. Dataset loader
###############################################################################

class BurstDataset:
    """Load *and* preprocess a burst cut‑out stored as a 2‑D ``.npy``.

    The preprocessing pipeline is split into discrete helpers so each
    step is unit‑testable and can be swapped out (e.g. different
    band‑pass correction).  The full load runs automatically unless
    ``lazy=True``.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        path: str | Path,
        *,
        telescope: str = "DSA-110",
        cfg_path: str | Path = "telescopes.yaml",
        f_factor: int = 1,
        t_factor: int = 1,
        outer_trim: float = 0.45,
        smooth_ms: float = 0.5,
        center_burst: bool = True,
        flip_freq: bool = True, # flip if raw stored high→low
        off_idx: slice | Sequence[int] = slice(0, 1000),
        lazy: bool = False,
    ) -> None:
        self.path = Path(path)
        self.params = load_telescope(telescope, cfg_path)
        assert 0.0 <= outer_trim < 0.5, "outer_trim must be < 0.5"
        self.f_factor = f_factor
        self.t_factor = t_factor
        self.outer_trim = outer_trim
        self.smooth_ms = smooth_ms   # FWHM of Gaussian in *ms*
        self.center_burst = center_burst
        self.flip_freq = flip_freq
        self.off_idx = off_idx

        # will be populated by _load()
        self.data: NDArray[np.floating] | None = None
        self.freq: NDArray[np.floating] | None = None
        self.time: NDArray[np.floating] | None = None
        self.df_MHz: float | None = None
        self.dt_ms: float | None = None

        if not lazy:
            self.load()

    # ------------------------------------------------------------------
    # publicd
    # ------------------------------------------------------------------
    def load(self):
        """Run the full preprocessing chain (idempotent)."""
        if self.data is not None:
            return  # already loaded
        raw = self._load_raw()
        if self.flip_freq:
            raw = np.flipud(raw)
        ds = self._bandpass_correct(raw)
        ds = self._trim_buffer(ds)
        ds = self._downsample(ds)
        ds = self._normalise(ds)
        self.data = ds
        self.freq, self.time = self._build_axes(ds)
        if self.center_burst:
            self._centre_burst()
        
    # quick‑look --------------------------------------------------------
    def quicklook(self, title: str = "Dynamic spectrum"):
        if self.data is None:
            self.load()
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_dynamic(ax, self.data, self.time, self.freq, cmap="plasma")
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Frequency [GHz]")
        ax.set_title(title)
        plt.tight_layout()

    # ------------------------------------------------------------------
    # pipeline helpers (private)
    # ------------------------------------------------------------------
    def _load_raw(self):
        return np.nan_to_num(np.load(self.path).astype(float))

    def _bandpass_correct(self, arr):
        """Per‑channel z‑score with dead‑channel protection.

        Channels whose off‑pulse variance is exactly zero (all NaNs or
        a hardware flag of zeros) would yield division‑by‑zero → ``±inf``
        after normalisation.  Here we set ``σ=NaN`` for those channels,
        carry the NaNs through the division, then convert to zeros so
        downstream averaging keeps them dark instead of blinding white.
        """
        if isinstance(self.off_idx, slice):
            idx = np.arange(arr.shape[1])[self.off_idx]
        else:
            idx = np.asarray(self.off_idx)

        mu = np.nanmean(arr[:, idx], axis=1, keepdims=True)
        sig = np.nanstd(arr[:, idx], axis=1, keepdims=True)
        sig[sig == 0] = np.nan  # mark zero‑variance (dead) channels

        arr_bp = (arr - mu) / sig
        return np.nan_to_num(arr_bp, nan=0.0, posinf=0.0, neginf=0.0)

    def _trim_buffer(self, arr):
        n_left = int(self.outer_trim * arr.shape[1])
        return arr[:, n_left:-n_left]

    def _downsample(self, arr):
        from burstfit import downsample
        return downsample(arr, self.f_factor, self.t_factor)

    def _normalise(self, arr):
        # Adjust off_idx for downsampling
        if isinstance(self.off_idx, slice):
            start = self.off_idx.start // self.t_factor if self.off_idx.start else 0
            stop = self.off_idx.stop // self.t_factor if self.off_idx.stop else arr.shape[1]
            idx = np.arange(arr.shape[1])[start:stop]
        else:
            idx = np.asarray(self.off_idx) // self.t_factor

        # Ensure indices are within bounds
        idx = idx[idx < arr.shape[1]]

        off = arr[:, idx]
        arr = (arr - np.nanmean(off)) / np.nanstd(off)
        return arr / np.nanmax(arr)
    
    def _centre_burst(self):
        """
        Roll dynamic spectrum so that the *smoothed* burst envelope peaks at
        the centre of the time axis.
        """
        # 1) burst profile
        prof = self.data.sum(axis=0)

        # 2) Gaussian smooth  (convert FWHM ms -> sigma in samples)
        sigma  = (self.smooth_ms / (2.355 * self.dt_ms))
        kernel = np.exp(-0.5*((np.arange(-4*sigma, 4*sigma+1))/sigma)**2)
        kernel /= kernel.sum()
        prof_s = np.convolve(prof, kernel, mode="same")

        # 3) roll so peak -> centre
        t_peak = np.argmax(prof_s)
        mid    = self.data.shape[1] // 2
        shift  = mid - t_peak
        self.data = np.roll(self.data, shift, axis=1)
        self.time = self.time + shift * self.dt_ms
        
        

    def _build_axes(self, ds):
        p = self.params
        self.df_MHz = p["df_MHz_raw"] * self.f_factor
        self.dt_ms = p["dt_ms_raw"] * self.t_factor
        freq = np.linspace(p["f_min_GHz"], p["f_max_GHz"], ds.shape[0])
        time = np.arange(ds.shape[1]) * self.dt_ms
        return freq, time

###############################################################################
# 2. Fitting layer
###############################################################################

class BurstFitter:
    """Run an emcee fit and return a sampler."""

    def __init__(
        self,
        dataset: BurstDataset,
        *,
        dm_init: float = 0.0,
        n_steps: int = 2000,
        pool = None,
    ) -> None:
        self.ds = dataset
        self.model = FRBModel(time=dataset.time, freq=dataset.freq, data=dataset.data, dm_init=dm_init)
        self.n_steps = n_steps
        self.pool = pool

    #def _guess(self) -> FRBParams:
    #    # original burstsum & timepeak
    #    c0 = float(np.nanmax(np.sum(self.ds.data, axis=1)))
    #    prof = np.sum(self.ds.data, axis=0)
    #    t  = self.ds.time
    #    t0 = float(t[np.argmax(prof)])

    #    # data estimates
    #    zeta_est, tau1ghz_est = estimate_smear_scatter(t, prof, self.ds.freq)

    #    # spectral index fallback
    #    gamma0 = -1.0

    #    return FRBParams(
    #        c0=c0,
    #        t0=t0,
    #        gamma=gamma0,
    #        zeta=zeta_est,
    #        tau_1ghz=tau1ghz_est
    #    )
    
    def fit(self, model_key="M3"):
        # 1) smart initial guess via optimizer
        p0 = self.smart_initial_guess(model_key)

        # 2) warm-up sampler (short + thin)
        names = FRBFitter(self.model, None, n_steps=0)._ORDER[model_key]
        ndim = len(names)
        nwalk = max(self.n_walkers_mult * ndim, 2 * ndim)
        pool  = self.pool

        warm_sampler = emcee.EnsembleSampler(nwalk, ndim, _log_prob_wrapper,
                                             args=(self.model, build_priors(p0,3.0),
                                                   FRBFitter(self.model,None,0)._ORDER, model_key),
                                             pool=pool)
        warm_sampler.run_mcmc(p0.to_sequence(model_key), self.n_steps_warm, thin_by=self.thin_warm, progress=True)
        warm = warm_sampler.get_chain(flat=True, discard=0, thin=self.thin_warm)

        # 3) estimate covariance & reinitialize (with jitter to avoid singular cov)
        cov = np.cov(warm, rowvar=False)
        # add a small fraction of the average variance to the diagonal
        eps = 1e-6 * np.trace(cov) / cov.shape[0]
        cov += np.eye(cov.shape[0]) * eps
        mean = np.mean(warm, axis=0)
        p0_ensemble = np.random.multivariate_normal(mean, cov, size=nwalk)

        # 4) final long run
        sampler = emcee.EnsembleSampler(nwalk, ndim, _log_prob_wrapper,
                                        args=(self.model, build_priors(p0,3.0),
                                              FRBFitter(self.model,None,0)._ORDER, model_key),
                                        pool=pool)
        sampler.run_mcmc(p0_ensemble, self.n_steps, progress=True)

        return sampler


    #def fit(self, model_key="M3"):
    #    p0 = self._guess()
    #    pri = build_priors(p0, scale=3.0)
    #    fitter = FRBFitter(self.model, pri, n_steps=self.n_steps, pool=self.pool)

    #    # — determine sampler dimensions from the chosen model —
    #    names = fitter._ORDER[model_key]
    #    ndim  = len(names)
    #    # walkers: at least 2×ndim, or fitter’s multiplier
    #    nwalk = max(fitter.n_walkers_mult * ndim, 2 * ndim)

    #    # — decide on thinning for HDFBackend storage —
    #    # here: one sample per 1000 steps (min 1)
    #    thin = max(1, self.n_steps // 1000)

    #    if thin > 1:
    #        # spin up the backend on disk
    #        backend = emcee.backends.HDFBackend("chain.h5")
    #        backend.reset(nwalk, ndim)

    #        # use the same log-prob wrapper as FRBFitter.sample
    #        from burstfit import _log_prob_wrapper

    #        sampler = emcee.EnsembleSampler(
    #            nwalk, ndim, _log_prob_wrapper,
    #            args=(self.model, pri, fitter._ORDER, model_key),
    #            pool=self.pool,
    #            backend=backend
    #        )
    #        # actually run the MCMC
    #        # (identical n_steps to FRBFitter.sample)
    #        p0_slice = FRBParams.from_sequence(
    #            [getattr(p0, n) for n in names] +
    #            [0] * (5 - len(names)),
    #            model_key
    #        )
    #        sampler.run_mcmc(p0_slice, self.n_steps, progress=True)
    #        return sampler

    #    # fallback: use the standard wrapper
    #    return fitter.sample(p0, model_key=model_key)

###############################################################################
# 3. Diagnostic layer
###############################################################################

class BurstDiagnostics:
    def __init__(self, dataset: BurstDataset, best_p: FRBParams, dm_init: float = 0.0):
        if dataset.data is None:
            dataset.load()
        self.ds = dataset
        self.best_p = best_p
        self.dm_init = dm_init
        self._subband_results = None
        self._influence_results = None

    def subband(self, best_key: str):
        self._subband_results = subband_consistency(
            self.ds.data, self.ds.freq, self.ds.time, 
            self.dm_init, self.best_p, model_key=best_key
        )
        return self._subband_results

    def influence(self, best_key: str, plot=False):
        delta = leave_one_out_influence(
            self.ds.data, self.ds.freq, self.ds.time, 
            self.dm_init, self.best_p, model_key=best_key
        )
        self._influence_results = delta
        if plot:
            fig, ax = plt.subplots(figsize=(8, 3))
            plot_influence(ax, delta, self.ds.freq)
        return delta

###############################################################################
# 4. Pipeline façade
###############################################################################

class BurstPipeline:
    def __init__(
        self,
        path: str | Path,
        *,
        telescope: str = "DSA-110",
        cfg_path: str | Path = "telescopes.yaml",
        n_steps: int = 2000,
        f_factor: int = 1,
        t_factor: int = 1,
        outer_trim: float = 0.45,
        center_burst: bool = True,   # default
        smooth_ms: float = 0.5,  # widen for very noisy data
        pool=None,
    ) -> None:
        self.dataset = BurstDataset(path, 
                                    telescope=telescope, 
                                    cfg_path=cfg_path, 
                                    f_factor=f_factor, 
                                    t_factor=t_factor, 
                                    outer_trim=outer_trim,
                                    center_burst=center_burst,
                                    smooth_ms=smooth_ms)
        self.pool  = pool      
        self.fitter = BurstFitter(self.dataset, n_steps=n_steps, pool=self.pool)

    def run_full(self, *, model_scan=True, diagnostics=True, plot=False):
        if model_scan:
            best_key, res = fit_models_bic(
                data=self.dataset.data,
                freq=self.dataset.freq,
                time=self.dataset.time,
                dm_init=0.0,
                init=self.fitter._guess(),
                n_steps=self.fitter.n_steps // 2,  # This is 1000 steps
                pool=self.pool
            )
            sampler = res[best_key][0]
            # For model scan, we used n_steps // 2
            n_steps_used = self.fitter.n_steps // 2
        else:
            best_key = "M3"
            sampler = self.fitter.fit(best_key)
            # For direct fit, we used full n_steps
            n_steps_used = self.fitter.n_steps

        def auto_burn_thin(sampler, safety_factor_burn=2.0, safety_factor_thin=0.5):
            """Estimate burn-in and thin by sampler autocorrelation times."""
            try:
                tau = sampler.get_autocorr_time(tol=0)
            except Exception as e:
                warnings.warn(f"Could not estimate autocorr time: {e}. Falling back to defaults.")
                # fallback: 25% burn, thin=1
                return sampler.iteration // 4, 1

            tau_max = np.max(tau)
            tau_min = np.min(tau)

            burn = int(safety_factor_burn * tau_max)
            thin = max(1, int(safety_factor_thin * tau_min))

            # ensure we don’t exceed total samples
            burn = min(burn, sampler.iteration // 2)
            return burn, thin

        burn, thin = auto_burn_thin(sampler)
        # Now use the correct n_steps for chain processing
        flat = sampler.get_chain(discard=burn, thin=thin, flat=True)
        # Get the flattened log probabilities with same discard/thin
        log_probs_flat = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
        
        # Find the best parameters
        best_idx = np.argmax(log_probs_flat)
        best_p = FRBParams.from_sequence(flat[best_idx], best_key)

        diag = None
        if diagnostics:
            diag = BurstDiagnostics(self.dataset, best_p)
            diag.subband(best_key)
            diag.influence(best_key, plot=plot)

        model = None
        model_dyn = None
            
        if plot:
            self.dataset.quicklook()

            # Create the model with proper initialization
            model = FRBModel(
                time=self.dataset.time, 
                freq=self.dataset.freq, 
                data=self.dataset.data,
                dm_init=0.0  # Use the same dm_init as in fit
            )
            model_dyn = model(best_p, best_key)

            # Plot the model
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_dynamic(ax, model_dyn, self.dataset.time, self.dataset.freq, cmap="plasma")
            ax.set_title("Best-fit model")
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Frequency [GHz]")
            plt.tight_layout()

            # Plot the residual - FIX THE ARGUMENT ORDER
            fig, ax = plt.subplots(figsize=(8, 4))
            residual = self.dataset.data - model_dyn
            plot_dynamic(ax, residual, self.dataset.time, self.dataset.freq, cmap="coolwarm")
            ax.set_title("Residual")
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Frequency [GHz]")
            plt.tight_layout()

            plt.show()

        return sampler, model, model_dyn, {
            "best_key": best_key,
            "best_params": best_p,
            "sampler": sampler,
            "diagnostics": diag,
        }

###############################################################################
# 5. CLI wrapper
###############################################################################


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Run BurstFit pipeline on a .npy burst cut‑out")
    parser.add_argument("npy", type=Path, help="Input .npy file")
    parser.add_argument("--telescope", default="DSA-110")
    parser.add_argument("--cfg", default="telescopes.yaml")
    parser.add_argument("--nproc", type=int, default=None, help="Pool size (0=serial, omit=auto‑detect)")
    parser.add_argument("--yes", action="store_true", help="Skip pool confirmation prompt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    pool = build_pool(args.nproc, auto_ok=args.yes, label="BurstFit")
    with pool or contextlib.nullcontext():
        pipe = BurstPipeline(
            args.npy,
            telescope=args.telescope,
            cfg_path=args.cfg,
            n_steps=args.steps,
            pool=pool,
        )
        res = pipe.run_full(model_scan=not args.no_scan, diagnostics=not args.no_diagnostics, plot=args.plot)
        print("Best model:", res["best_key"])
        print("Best parameters:", res["best_params"])

if __name__ == "__main__":
    import contextlib
    _main()
