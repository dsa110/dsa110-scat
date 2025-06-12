import numpy as np
import emcee
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy.signal import fftconvolve

import numpy as np
from scipy.signal import fftconvolve

class CorrectedOptimizedFRBModel:
    """
    Corrected optimized version of FRBModel that properly handles dispersion delays
    """
    def __init__(self, data, time, frequencies, DM_init, beta=2):
        self.data = data
        self.time = time
        self.frequencies = frequencies
        self.DM_init = DM_init
        self.beta = beta
        self.N_channels = data.shape[0]
        self.N_time_samples = data.shape[1]
        
        # Pre-compute frequently used values
        self.dt = time[1] - time[0]
        self.nu_ref = frequencies.max()
        self.freq_ratio = self.frequencies / 1.0  # For scattering
        
        # Cache noise estimation
        self._noise_std = None
        self.estimate_noise()
        
        # Pre-allocate arrays for model computation
        self._model_workspace = np.zeros_like(data)
    
    def estimate_noise(self):
        """Cached noise estimation"""
        if self._noise_std is None:
            off_pulse_indices = np.concatenate([
                np.arange(0, self.N_time_samples // 4),
                np.arange(3 * self.N_time_samples // 4, self.N_time_samples)
            ])
            off_pulse_data = self.data[:, off_pulse_indices]
            self._noise_std = np.maximum(np.std(off_pulse_data, axis=1), 1e-3)
        return self._noise_std
    
    @property
    def noise_std(self):
        return self.estimate_noise()
    
    def t_i_DM(self, DM_err=0.):
        """
        Compute dispersion delay for each frequency channel.
        
        Parameters:
        - DM_err: Deviation of the burst DM from the initial DM
        
        Note: For pre-dedispersed data, DM_err should be ~0, so delays are minimal
        """
        # Faithful reproduction of original formula
        t_i_DM = (4.15) * DM_err * ((self.frequencies) ** (-self.beta) - self.nu_ref ** (-self.beta))
        return t_i_DM
    
    def sigma_i_DM(self, DM, zeta=0.):
        """
        Compute intra-channel dispersion smearing and intrinsic width.
        
        Parameters:
        - DM: Total dispersion measure
        - zeta: Intrinsic pulse width (only for Model 1 and Model 3)
        """
        sigma_i_DM = (1.622e-3) * DM * (self.frequencies) ** (-self.beta - 1)
        sigma_i_DM = np.sqrt(sigma_i_DM ** 2 + zeta ** 2)
        return sigma_i_DM

    def model_vectorized(self, params, model_type='model0'):
        """
        Corrected vectorized model computation
        """
        c0, t0, spectral_index = params[0], params[1], params[2]
        
        # Vectorized spectral model
        reference_frequency = self.frequencies[self.N_channels // 2]
        c_i = c0 * (self.frequencies / reference_frequency) ** spectral_index
        
        # Extract additional parameters based on model type
        zeta = 0
        tau_1GHz = 0
        if model_type == 'model0':
            pass  # Both zeta and tau_1GHz remain 0
        elif model_type == 'model1':
            zeta = params[3]
        elif model_type == 'model2':
            tau_1GHz = params[3]
        elif model_type == 'model3':
            zeta = params[3]
            tau_1GHz = params[4]
        else:
            raise ValueError("Invalid model type")
        
        # Calculate dispersion delays (will be ~0 for dedispersed data)
        t_i_DM = self.t_i_DM(DM_err=0.)  # Shape: (N_freq,)
        
        # Vectorized dispersion and width calculations
        DM = self.DM_init
        sigma_i_DM = self.sigma_i_DM(DM, zeta)  # Shape: (N_freq,)
        
        # Create proper mu values: mu_i = t0 + t_i_DM[i] for each frequency
        mu_i = t0 + t_i_DM  # Shape: (N_freq,)
        
        # Broadcast for vectorized computation
        mu_grid = mu_i[:, np.newaxis]  # Shape: (N_freq, 1)
        c_grid = c_i[:, np.newaxis]   # Shape: (N_freq, 1) 
        sigma_grid = sigma_i_DM[:, np.newaxis]  # Shape: (N_freq, 1)
        time_broadcast = self.time[np.newaxis, :]  # Shape: (1, N_time)
        
        # Vectorized Gaussian computation: exp(-(time - mu)^2 / (2*sigma^2))
        time_diff = time_broadcast - mu_grid  # Shape: (N_freq, N_time)
        gaussian_arg = -0.5 * (time_diff / sigma_grid)**2
        normalization = c_grid / np.sqrt(2 * np.pi * sigma_grid**2)
        
        # Compute the model
        model = normalization * np.exp(gaussian_arg)
        
        # Apply scattering convolution if needed
        if tau_1GHz > 0:
            model = self._apply_scattering_vectorized(model, tau_1GHz)
        
        # Flip frequency axis to match data convention
        return np.flip(model, axis=0)
    
    def _apply_scattering_vectorized(self, model, tau_1GHz):
        """
        Apply scattering convolution (frequency-dependent exponential tail)
        """
        alpha = 4.0  # Fixed scattering index
        tau_i = tau_1GHz * (self.freq_ratio ** (-alpha))
        
        # Apply scattering to each frequency channel
        for i in range(self.N_channels):
            if tau_i[i] > 0:
                # Create pulse broadening function (exponential decay)
                pbf = np.zeros_like(self.time)
                positive_time_mask = self.time >= 0
                pbf[positive_time_mask] = np.exp(-self.time[positive_time_mask] / tau_i[i])
                pbf /= np.sum(pbf) * self.dt  # Normalize
                
                # Convolve with the model
                conv_result = fftconvolve(model[i, :], pbf, mode='full')
                
                # Extract the relevant portion (matching original implementation)
                conv_time = np.arange(len(conv_result)) * self.dt + 2 * self.time[0]
                start_idx = np.searchsorted(conv_time, self.time[0])
                end_idx = start_idx + len(self.time)
                model[i, :] = conv_result[start_idx:end_idx]
        
        return model
    
    def log_likelihood(self, params, model_type='model0'):
        """
        Compute the log-likelihood of the model given the data.
        """
        model_spectrum = self.model_vectorized(params, model_type)
        sigma2 = self.noise_std[:, np.newaxis] ** 2
        sigma2 = np.maximum(sigma2, 1e-6)  # Prevent division by zero
        residuals = self.data - model_spectrum
        lnL = -0.5 * np.sum(residuals ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
        return lnL

    def log_prior(self, params, prior_bounds, model_type='model0'):
        """
        Define priors for the model parameters.
        """
        c0 = params[0]
        t0 = params[1]
        spectral_index = params[2]

        # Check basic parameter bounds
        if c0 <= prior_bounds['c0'][0] or c0 > prior_bounds['c0'][1]:
            return -np.inf
        if not (prior_bounds['t0'][0] <= t0 <= prior_bounds['t0'][1]):
            return -np.inf
        if not (prior_bounds['spectral_index'][0] <= spectral_index <= prior_bounds['spectral_index'][1]):
            return -np.inf

        # Additional priors based on model type
        if model_type == 'model1':
            zeta = params[3]
            if not (prior_bounds['zeta'][0] <= zeta <= prior_bounds['zeta'][1]):
                return -np.inf
        elif model_type == 'model2':
            tau_1GHz = params[3]
            if not (prior_bounds['tau_1GHz'][0] <= tau_1GHz <= prior_bounds['tau_1GHz'][1]):
                return -np.inf
        elif model_type == 'model3':
            zeta = params[3]
            tau_1GHz = params[4]
            if not (prior_bounds['zeta'][0] <= zeta <= prior_bounds['zeta'][1]):
                return -np.inf
            if not (prior_bounds['tau_1GHz'][0] <= tau_1GHz <= prior_bounds['tau_1GHz'][1]):
                return -np.inf

        return 0.0  # Log-prior is zero if within bounds

    def log_posterior(self, params, prior_bounds, model_type='model0'):
        """
        Compute the log-posterior probability.
        """
        lp = self.log_prior(params, prior_bounds, model_type)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(params, model_type)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

# Validation function to compare with original
def validate_models(original_model, optimized_model, test_params, model_type='model0'):
    """
    Compare outputs between original and optimized implementations
    """
    # Generate models
    model_orig = original_model.model(test_params, model_type)
    model_opt = optimized_model.model_vectorized(test_params, model_type)
    
    # Check if they match
    max_diff = np.max(np.abs(model_orig - model_opt))
    rel_diff = max_diff / np.max(np.abs(model_orig))
    
    print(f"Model comparison for {model_type}:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")
    print(f"  Models match: {np.allclose(model_orig, model_opt, rtol=1e-6)}")
    
    return np.allclose(model_orig, model_opt, rtol=1e-6)

# Example usage and testing
#if __name__ == "__main__":
#    # Create test data
#    np.random.seed(42)
#    n_freq, n_time = 48, 100
#    time_test = np.linspace(0, 10, n_time)
#    freq_test = np.linspace(0.4, 0.8, n_freq)
#    data_test = np.random.randn(n_freq, n_time) * 0.1
#    
#    # Add a fake burst
#    burst_time = 5.0
#    for i, f in enumerate(freq_test):
#        amplitude = 2.0 * (f / 0.6) ** (-2)
#        width = 0.5
#        data_test[i, :] += amplitude * np.exp(-0.5 * ((time_test - burst_time) / width)**2)
#    
#    # Create model instances
#    model_opt = CorrectedOptimizedFRBModel(data_test, time_test, freq_test, DM_init=0.0)
#    
#    # Test parameters
#    test_params = [1.0, 5.0, -2.0, 0.1, 0.05]  # c0, t0, gamma, zeta, tau
#    
#    # Test model generation
#    for model_type in ['model0', 'model1', 'model2', 'model3']:
#        n_params = {'model0': 3, 'model1': 4, 'model2': 4, 'model3': 5}[model_type]
#        params = test_params[:n_params]
#        
#        try:
#            model_result = model_opt.model_vectorized(params, model_type)
#            print(f"{model_type}: Generated model shape {model_result.shape}, max value {np.max(model_result):.3f}")
#        except Exception as e:
#            print(f"{model_type}: Error - {e}")
            
def downsample_data(data, f_factor = 1, t_factor = 1):   

    # Check data shape
    print(f'Power Shape (frequency axis): {data.shape[0]}')
    print(f'Power Shape (time axis): {data.shape[1]}')

    # Downsample in frequency
    # Ensure nearest multiple is not greater than the frequency axis length
    nrst_mltpl_f = f_factor * (data.shape[0] // f_factor)
    print(f'Nearest Multiple To Downsampling Factor (frequency): {nrst_mltpl_f}')

    # Clip the frequency axis to the nearest multiple
    data_clip_f = data[:nrst_mltpl_f, :]

    # Downsample along the frequency axis (y-axis)
    data_ds_f = data_clip_f.reshape([
        nrst_mltpl_f // f_factor, f_factor,
        data_clip_f.shape[1]
    ]).mean(axis=1)

    # Downsample in time
    # Ensure nearest multiple is not greater than the time axis length
    nrst_mltpl_t = t_factor * (data_ds_f.shape[1] // t_factor)
    print(f'Nearest Multiple To Downsampling Factor (time): {nrst_mltpl_t}')

    # Clip the time axis to the nearest multiple
    data_clip_t = data_ds_f[:, :nrst_mltpl_t]

    # Downsample along the time axis (x-axis)
    data_ds_t = data_clip_t.reshape([
        data_clip_t.shape[0],  # Frequency axis remains the same
        nrst_mltpl_t // t_factor, t_factor
    ]).mean(axis=2)

    # Output the final downsampled data
    print(f'Downsampled Data Shape: {data_ds_t.shape}')

    return data_ds_t