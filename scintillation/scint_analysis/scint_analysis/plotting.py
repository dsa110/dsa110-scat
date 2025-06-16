# ==============================================================================
# File: scint_analysis/scint_analysis/plotting.py (NEW FILE)
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np
import logging

log = logging.getLogger(__name__)

def plot_dynamic_spectrum(spectrum_obj, **kwargs):
    """
    Plots the 2D dynamic spectrum.

    Args:
        spectrum_obj (DynamicSpectrum): The object to plot.
        **kwargs: Additional keyword arguments passed to plt.imshow().
    """
    log.info("Generating dynamic spectrum plot.")
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    # Use a percentile for the color limit to handle outliers gracefully
    vmax = np.percentile(spectrum_obj.power.compressed(), 99)
    
    im = ax.imshow(
        spectrum_obj.power,
        aspect='auto',
        origin='lower',
        extent=[spectrum_obj.times.min(), spectrum_obj.times.max(), 
                spectrum_obj.frequencies.min(), spectrum_obj.frequencies.max()],
        vmax=vmax,
        **kwargs
    )
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("Dynamic Spectrum")
    fig.colorbar(im, ax=ax, label="Power (arbitrary units)")
    plt.show()


def plot_acf(acf_obj, fit_result=None, **kwargs):
    """
    Plots an ACF and optionally its Lorentzian fit.

    Args:
        acf_obj (ACF): The ACF object to plot.
        fit_result (lmfit.ModelResult, optional): The result from an lmfit run.
        **kwargs: Additional keyword arguments passed to plt.plot().
    """
    log.info("Generating ACF plot.")
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))
    
    ax.plot(acf_obj.lags, acf_obj.acf, 'k-', alpha=0.8, label="ACF Data", **kwargs)
    
    if fit_result:
        ax.plot(acf_obj.lags, fit_result.eval(x=acf_obj.lags), 'r--', label="Lorentzian Fit")
        gamma = fit_result.params['gamma1'].value
        ax.set_title(f"Decorrelation Bandwidth = {gamma*1000:.2f} kHz")
        ax.set_xlim(-5 * gamma, 5 * gamma) # Auto-zoom to the feature
    
    ax.set_xlabel("Frequency Lag (MHz)")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

def plot_analysis_overview(analysis_results, acf_results, all_subband_fits, powlaw_fit_params, **kwargs):
    """
    Generates a multi-panel summary plot of the entire scintillation analysis.

    Args:
        analysis_results (dict): The final results dictionary from the pipeline.
        acf_results (dict): The dictionary containing the intermediate ACF data.
        all_subband_fits (list): List of dictionaries containing the lmfit result objects.
    """
    log.info("Generating full analysis overview plot.")
    best_model = analysis_results['best_model']
    num_components = len(analysis_results['components'])
    
    fig = plt.figure(figsize=kwargs.get('figsize', (12, 6 + 5 * num_components)))
    gs = fig.add_gridspec(2 + num_components, 2)
    fig.suptitle(f"Scintillation Analysis Summary", fontsize=16)

    # --- Panel 1: Stacked Sub-band ACFs with Fits ---
    ax_acf = fig.add_subplot(gs[0:2, 0])
    cmap = plt.get_cmap('viridis')
    num_subbands = len(acf_results['subband_acfs'])
    for i in range(num_subbands):
        offset = i * 1.5
        rgba = cmap(i / (num_subbands - 1))
        lags = acf_results['subband_lags_mhz'][i]
        acf = acf_results['subband_acfs'][i]
        
        ax_acf.plot(lags, acf + offset, color=rgba)
        
        # Plot the preferred fit model
        fit_obj = all_subband_fits[i].get(f'fit_{best_model}_comp')
        if fit_obj and fit_obj.success:
            ax_acf.plot(lags, fit_obj.eval(x=lags) + offset, 'k--', alpha=0.7)

    ax_acf.set_yticks([(i * 1.5) for i in range(num_subbands)])
    ax_acf.set_yticklabels([f"{cf:.1f}" for cf in acf_results['subband_center_freqs_mhz']])
    ax_acf.set_title("Sub-band ACFs with Best-Fit Models")
    ax_acf.set_xlabel("Frequency Lag (MHz)")
    ax_acf.set_ylabel("Center Frequency (MHz)")
    ax_acf.set_xlim(-0.5, 0.5)
    
    # --- Panel 2: BIC Model Comparison ---
    ax_bic = fig.add_subplot(gs[0, 1])
    model_labels = ["1-Comp", "2-Comp"]
    for i in range(num_subbands):
        fits = all_subband_fits[i]
        fit1 = fits.get('fit_1_comp')
        fit2 = fits.get('fit_2_comp')
        if fit1 and fit2 and fit1.success and fit2.success:
            ax_bic.plot([1, 2], [fit1.bic, fit2.bic], 'o-', color='k', alpha=0.2)
            
    ax_bic.axvline(best_model, color='r', linestyle='--', label=f'Preferred Model')
    ax_bic.set_xticks([1, 2])
    ax_bic.set_xticklabels(model_labels)
    ax_bic.set_ylabel("Bayesian Information Criterion (BIC)")
    ax_bic.set_title("Model Selection")
    ax_bic.legend()

    # --- Panels 3+: Power-Law Fits for Each Component ---
    comp_idx = 0
    for name, component_data in analysis_results['components'].items():
        ax_plaw = fig.add_subplot(gs[2 + comp_idx, :])
        
        #Extract data points directly from the measurements list
        measurements = component_data.get('subband_measurements', [])
        if not measurements:
            continue
            
        freqs = np.array([m.get('freq_mhz') for m in measurements])
        bws = np.array([m.get('bw') for m in measurements])
        fit_errs = np.array([m.get('bw_err', 0) for m in measurements])
        finite_errs = np.array([m.get('finite_err', 0) for m in measurements])
        total_errs = np.sqrt(np.nan_to_num(fit_errs)**2 + np.nan_to_num(finite_errs)**2)
        
        ax_plaw.errorbar(freqs, bws, yerr=total_errs, fmt='o', capsize=5, label='Sub-band Measurements')
        
        # Plot the best-fit power-law model
        params = powlaw_fit_params
        print(f'Params: {params}')
        c, n = params['c'].value, params['n'].value
        freq_model = np.linspace(min(freqs), max(freqs), 100)
        scint_model = c * (freq_model ** n)
        
        ax_plaw.plot(freq_model, scint_model, 'r--', label=f'Power-Law Fit ($\\alpha={n:.2f}$)')
        ax_plaw.set_title(f"Power-Law Fit: {name.replace('_', ' ').title()}")
        ax_plaw.set_xlabel("Frequency (MHz)")
        ax_plaw.set_ylabel("Decorrelation BW (MHz)")
        ax_plaw.legend()
        ax_plaw.grid(True, alpha=0.2)
        comp_idx += 1
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
