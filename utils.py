"""
Utility functions for Adept Sampler ComfyUI nodes.
Ported from Stable Diffusion WebUI reForge extension.
"""

import math
import torch

try:
    from torchvision.transforms.functional import gaussian_blur
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def to_d(x, sigma, denoised):
    """Convert denoised prediction to derivative with robust numerical stability."""
    diff = x - denoised
    safe_sigma = torch.clamp(sigma, min=1e-4)
    derivative = diff / safe_sigma
    
    # Normalize derivative by sigma to handle different prediction types
    sigma_adaptive_threshold = 1000.0 * (1.0 + sigma / 10.0)
    
    derivative_max = torch.abs(derivative).max()
    if derivative_max > sigma_adaptive_threshold:
        derivative = torch.clamp(derivative, -sigma_adaptive_threshold, sigma_adaptive_threshold)
    
    return derivative


def to_d_enhanced_ancestral(x, sigma, denoised, eta, progress, generator=None):
    """
    Enhanced derivative computation optimized for ancestral sampling.
    Provides ancestral-specific derivative corrections that adapt
    based on the sampling progress and eta value.
    """
    diff = x - denoised
    safe_sigma = torch.clamp(sigma, min=1e-4)
    base_derivative = diff / safe_sigma

    def safe_randn_like(tensor, generator=None):
        if generator is None:
            return torch.randn_like(tensor)
        try:
            return torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype, generator=generator)
        except (TypeError, AttributeError):
            return torch.randn_like(tensor)

    # Ancestral-specific enhancements (conservative to reduce noisiness)
    if eta > 1.0:
        eta_correction = 0.02 * (eta - 1.0) * safe_randn_like(diff, generator) * progress
        base_derivative = base_derivative + eta_correction
    elif eta < 1.0:
        eta_correction = 0.015 * (1.0 - eta) * safe_randn_like(diff, generator) * (1.0 - progress)
        base_derivative = base_derivative - eta_correction

    # Progress-based phase corrections
    if progress < 0.3:
        phase_correction = 0.01 * safe_randn_like(diff, generator)
        base_derivative = base_derivative + phase_correction
    elif progress > 0.7:
        phase_correction = 0.008 * safe_randn_like(diff, generator)
        base_derivative = base_derivative - phase_correction

    # Final safety check
    sigma_adaptive_threshold = 500.0 * (1.0 + sigma / 10.0)
    derivative_max = torch.abs(base_derivative).max()
    if derivative_max > sigma_adaptive_threshold:
        base_derivative = torch.clamp(base_derivative, -sigma_adaptive_threshold, sigma_adaptive_threshold)

    return base_derivative


def get_noise_sampler(x):
    """Get proper noise sampler with working fallback."""
    try:
        import comfy.k_diffusion.sampling as k_sampling
        if hasattr(k_sampling, 'default_noise_sampler'):
            return k_sampling.default_noise_sampler(x)
    except ImportError:
        pass

    # Fallback noise sampler
    def simple_noise_sampler(sigma_from, sigma_to):
        noise = torch.randn_like(x)
        if abs(sigma_to - sigma_from) > 1e-6:
            scale = (sigma_to / sigma_from.clamp(min=1e-6)).sqrt()
            noise = noise * scale
        return noise
    return simple_noise_sampler


def apply_dynamic_thresholding(x, percentile=0.995, clamp_range=1.0):
    """Optimized dynamic thresholding with better stability."""
    if percentile >= 1.0:
        return x
    
    try:
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        abs_max = torch.abs(x_flat).max(dim=1, keepdim=True)[0]
        
        if abs_max.max() < 5.0:
            return x
        
        k = max(1, int(x_flat.shape[1] * (1.0 - percentile)))
        topk_vals = torch.topk(torch.abs(x_flat), k=k, dim=1, largest=True)[0]
        s = topk_vals[:, -1:].clamp(min=1.0)
        
        threshold = s * 2.5
        
        mask = torch.abs(x_flat) > threshold
        x_flat = torch.where(mask, torch.sign(x_flat) * threshold, x_flat)
        x_flat = x_flat * 0.98
        
        return x_flat.view(x.shape)
        
    except Exception as e:
        print(f"⚠️ Dynamic thresholding failed: {e}")
        return x


def compute_compensation_ratio(r, step_idx, total_steps, base_ratio=1.0):
    """
    Compute dynamic compensation ratio inspired by DC-Solver.
    Adapts interpolation based on step position.
    """
    progress = step_idx / max(total_steps - 1, 1)
    
    if progress < 0.3:
        phase_weight = 1.5
    elif progress < 0.7:
        phase_weight = 1.0
    else:
        phase_weight = 1.3
    
    compensation = base_ratio * phase_weight * (1.0 + 0.1 * math.tanh(r - 1.0))
    return compensation


def compute_tau_eqvae(progress, base_tau=0.5, phase_strength=0.5):
    """
    Phase-aware tau function for SA-Solver style stochasticity control.
    Optimized for EQ-VAE's smooth latent space.
    """
    if progress < 0.30:
        phase_factor = 1.0 + 0.2 * phase_strength
    elif progress < 0.60:
        phase_factor = 1.0 - 0.15 * phase_strength
    else:
        phase_factor = 1.0 - 0.3 * phase_strength
    
    return min(1.0, max(0.0, base_tau * phase_factor))


def compute_smea_factor(progress, smea_strength=0.5):
    """
    SMEA (Sinusoidal Multipass Euler Ancestral) inspired interpolation.
    Uses sine-based schedule to improve coherency at high resolutions.
    """
    if smea_strength <= 0:
        return 1.0
    
    smea_interp = 0.5 * (1 + math.sin(math.pi * (progress - 0.5)))
    return 1.0 - smea_strength * (1.0 - smea_interp)


def compute_eqvae_tau(progress, base_tau, phase_strength):
    """
    EQ-VAE optimized tau function with shifted phase boundaries.

    EQ-VAE converges faster due to cleaner latents, so we shift
    phase transitions earlier compared to standard VAE.

    Phase boundaries (vs standard 30%/60%):
    - Foundation: 0-25%
    - Structure: 25-55%
    - Refinement: 55-100%

    Args:
        progress: Sampling progress (0.0 to 1.0)
        base_tau: Base stochasticity level
        phase_strength: Phase adaptation intensity

    Returns:
        tau value optimized for EQ-VAE
    """
    if progress < 0.25:
        phase_factor = 1.0 + 0.10 * phase_strength
    elif progress < 0.55:
        phase_factor = 1.0 - 0.10 * phase_strength
    else:
        phase_factor = 1.0 - 0.20 * phase_strength

    return min(1.0, max(0.0, base_tau * phase_factor))


def compute_eqvae_noise_scale(base_s_noise, progress):
    """
    Compute noise scale optimized for EQ-VAE's cleaner latent space.

    EQ-VAE has ~56% lower latent noise compared to standard SDXL VAE.
    Uses balanced settings that maintain sharpness.

    Args:
        base_s_noise: Base noise scale from user settings
        progress: Sampling progress (0.0 to 1.0)

    Returns:
        Adjusted noise scale for EQ-VAE
    """
    # Balanced settings (maintains sharpness)
    eqvae_base_factor = 0.88
    if progress < 0.25:
        phase_factor = 1.0 + 0.05 * (1.0 - progress / 0.25)  # 1.05 -> 1.0
    elif progress < 0.60:
        phase_factor = 1.0 - 0.05 * ((progress - 0.25) / 0.35)  # 1.0 -> 0.95
    else:
        phase_factor = 0.95  # Flat, no further reduction

    return base_s_noise * eqvae_base_factor * phase_factor


def compute_eqvae_ndb(progress, ndb_strength):
    """
    Native Detail Boost optimized for EQ-VAE's uniform frequency distribution.

    EQ-VAE has more uniform energy across frequencies, allowing for
    effective high-frequency boosting without artifacts.

    Args:
        progress: Sampling progress (0.0 to 1.0)
        ndb_strength: User-specified boost strength

    Returns:
        Tuple of (blur_sigma, high_freq_boost):
        - blur_sigma: Gaussian blur sigma for frequency separation
        - high_freq_boost: High-frequency component multiplier
    """
    if ndb_strength <= 0:
        return 0.5, 0.0

    # EQ-VAE benefits from slightly wider frequency separation
    blur_sigma = 0.6

    # Balanced boost curve
    if progress < 0.30:
        phase_progress = progress / 0.30
        high_freq_boost = 0.03 * ndb_strength * phase_progress
    elif progress < 0.60:
        phase_progress = (progress - 0.30) / 0.30
        high_freq_boost = (0.03 + 0.07 * phase_progress) * ndb_strength
    else:
        phase_progress = (progress - 0.60) / 0.40
        high_freq_boost = (0.10 + 0.10 * phase_progress) * ndb_strength

    return blur_sigma, high_freq_boost


def compute_native_detail_boost(progress, ndb_strength=0.0):
    """
    Native Detail Boost (NDB): Enhances detail emergence at native resolution.
    
    Returns:
        Tuple of (base_scale, high_freq_boost)
    """
    if ndb_strength <= 0:
        return 1.0, 0.0
    
    if progress < 0.30:
        phase_progress = progress / 0.30
        high_freq_boost = 0.03 * ndb_strength * phase_progress
    elif progress < 0.60:
        phase_progress = (progress - 0.30) / 0.30
        high_freq_boost = (0.03 + 0.07 * phase_progress) * ndb_strength
    else:
        phase_progress = (progress - 0.60) / 0.40
        high_freq_boost = (0.10 + 0.08 * phase_progress) * ndb_strength
    
    return 1.0, high_freq_boost


def sa_solver_step(x, d_history, sigma, sigma_next, tau, s_noise=1.0, noise_sampler=None, order=2, ndb_strength=0.0, progress=0.0, eqvae_mode=False, eqvae_blur_sigma=None):
    """
    SA-Solver inspired step with controlled stochasticity.
    Uses Adams-Bashforth coefficients for multi-step integration.

    Args:
        x: Current latent tensor
        d_history: List of (sigma, derivative) tuples from previous steps
        sigma: Current sigma value
        sigma_next: Next sigma value
        tau: Stochasticity control (0=ODE, 1=full SDE)
        s_noise: Noise scaling factor
        noise_sampler: Function to generate scaled noise
        order: Multi-step order (1, 2, or 3)
        ndb_strength: Native Detail Boost strength (0=disabled)
        progress: Sampling progress (0.0 to 1.0) for NDB phase calculation
        eqvae_mode: Whether to use EQ-VAE optimized parameters
        eqvae_blur_sigma: Custom blur sigma for EQ-VAE NDB (None = use default)

    Returns:
        Tuple of (next latent, sigma_up used for noise)
    """
    dt = sigma_next - sigma
    
    # Compute interpolated derivative based on order and history
    if len(d_history) >= 2 and order >= 2:
        sigma_cur, d_cur = d_history[-1]
        sigma_prev, d_prev = d_history[-2]
        
        h_prev = sigma_cur - sigma_prev
        r = abs(dt / (h_prev + 1e-8)) if abs(h_prev) > 1e-8 else 1.0
        r = min(r, 2.0)
        
        if len(d_history) >= 3 and order >= 3:
            sigma_0, d_0 = d_history[-3]
            h_0 = sigma_prev - sigma_0
            h_1 = h_prev
            
            if abs(h_0) > 1e-6 and abs(h_1) > 1e-6:
                r0 = min(abs(h_1 / h_0), 2.0)
                r1 = min(abs(dt / (h_1 + 1e-8)), 2.0)
                
                tau_blend = 1.0 - tau
                c0_ab3 = 1.0 + (1.0 + r0) * r1 / 2.0
                c1_ab3 = -(1.0 + r0) * r1 / 2.0
                c2_ab3 = r0 * r1 / 2.0
                c0 = tau_blend * c0_ab3 + (1.0 - tau_blend) * 1.0
                c1 = tau_blend * c1_ab3
                c2 = tau_blend * c2_ab3
                
                c_sum = c0 + c1 + c2
                if abs(c_sum) > 1e-8:
                    c0 /= c_sum
                    c1 /= c_sum
                    c2 /= c_sum
                else:
                    c0, c1, c2 = 1.0, 0.0, 0.0
                
                d_interp = c0 * d_cur + c1 * d_prev + c2 * d_0
            else:
                tau_blend = 1.0 - tau
                c1_ab2 = 1.0 + 0.5 * r
                c2_ab2 = -0.5 * r
                c1 = tau_blend * c1_ab2 + (1.0 - tau_blend) * 1.0
                c2 = tau_blend * c2_ab2
                c_sum = c1 + c2
                if abs(c_sum) > 1e-8:
                    c1 /= c_sum
                    c2 /= c_sum
                d_interp = c1 * d_cur + c2 * d_prev
        else:
            tau_blend = 1.0 - tau
            c1_ab2 = 1.0 + 0.5 * r
            c2_ab2 = -0.5 * r
            c1 = tau_blend * c1_ab2 + (1.0 - tau_blend) * 1.0
            c2 = tau_blend * c2_ab2
            c_sum = c1 + c2
            if abs(c_sum) > 1e-8:
                c1 /= c_sum
                c2 /= c_sum
            d_interp = c1 * d_cur + c2 * d_prev
    elif len(d_history) >= 1:
        d_interp = d_history[-1][1]
    else:
        d_interp = torch.zeros_like(x)
    
    # Compute sigma_up based on tau (controls stochasticity)
    sigma_up = 0.0
    if tau > 0 and sigma_next > 0 and noise_sampler is not None:
        sigma_ancestral_sq = sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / (sigma ** 2 + 1e-8)
        sigma_ancestral = sigma_ancestral_sq ** 0.5 if sigma_ancestral_sq > 0 else 0.0
        sigma_up = tau * sigma_ancestral
        
        sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
        dt_adjusted = sigma_down - sigma
        
        x_det = x + d_interp * dt_adjusted
        noise = noise_sampler(sigma, sigma_next) * s_noise * sigma_up
        
        # Apply Native Detail Boost if enabled
        if ndb_strength > 0 and TORCHVISION_AVAILABLE:
            # Use EQ-VAE optimized NDB parameters if in EQ-VAE mode
            if eqvae_mode:
                blur_sigma, high_freq_boost = compute_eqvae_ndb(progress, ndb_strength)
            else:
                _, high_freq_boost = compute_native_detail_boost(progress, ndb_strength)
                blur_sigma = 0.5  # Default blur sigma

            # Override blur_sigma if explicitly provided
            if eqvae_blur_sigma is not None:
                blur_sigma = eqvae_blur_sigma

            # Extract high-frequency component from noise using Gaussian blur
            try:
                low_freq_noise = gaussian_blur(noise, kernel_size=3, sigma=blur_sigma)
                high_freq_noise = noise - low_freq_noise
                noise = noise + high_freq_noise * high_freq_boost
            except Exception:
                pass  # Fallback: use original noise if blur fails
        
        x_next = x_det + noise
    else:
        x_next = x + d_interp * dt
    
    return x_next, sigma_up


def compute_adaptive_noise_scale(excess_samples, base_s_noise,
                                  correction_power=0.5,
                                  dampen_floor=0.80, boost_ceiling=1.15):
    """
    Adaptive noise scale using calibrated sigma-relative excess.

    After collecting excess ratio samples during a warmup window inside the
    texture phase, computes a single global correction factor based on the
    median excess.

    The excess ratio measures how much noise is slowing down convergence:
        excess = (change_i / change_{i-1}) / (sigma_{i+1} / sigma_i)
    When excess > 1.0, the model converges slower than sigma predicts.

    Args:
        excess_samples: List of excess ratio values from warmup window
        base_s_noise: The effective s_noise to modulate
        correction_power: How aggressively to correct (0.5 = square root)
        dampen_floor: Minimum correction multiplier (default 0.80)
        boost_ceiling: Maximum correction multiplier (default 1.15)

    Returns:
        Tuple of (correction_factor, median_excess)
    """
    if not excess_samples:
        return 1.0, 0.0

    sorted_samples = sorted(excess_samples)
    median_excess = sorted_samples[len(sorted_samples) // 2]

    if median_excess > 0:
        correction = 1.0 / (median_excess ** correction_power)
    else:
        correction = 1.0

    correction = max(dampen_floor, min(boost_ceiling, correction))
    return correction, median_excess


def compute_binned_corrections(excess_bins, global_correction,
                                correction_power=0.5, dampen_floor=0.80,
                                boost_ceiling=1.15, min_samples=3):
    """
    Compute per-phase corrections from binned excess samples.

    Bins with fewer than min_samples fall back to the global correction.

    Returns:
        Dict mapping bin name to correction factor.
    """
    bin_corrections = {}
    for bin_name, samples in excess_bins.items():
        if len(samples) >= min_samples:
            correction, _ = compute_adaptive_noise_scale(
                samples, 1.0, correction_power, dampen_floor, boost_ceiling
            )
            bin_corrections[bin_name] = correction
        else:
            bin_corrections[bin_name] = global_correction
    return bin_corrections


def get_phase_correction(sigma_val, bin_corrections, global_correction):
    """Look up the appropriate correction for the current sigma phase."""
    if bin_corrections is None:
        return global_correction
    if sigma_val > 5.0:
        return bin_corrections.get('structural', global_correction)
    elif sigma_val > 0.5:
        return bin_corrections.get('texture', global_correction)
    else:
        return bin_corrections.get('cleanup', global_correction)


def adaptive_noise_step(denoised_raw, prev_denoised_raw, prev_change_norm,
                        sigma, sigma_next, excess_samples, adaptive_correction,
                        base_s_noise, step_idx, total_steps, warmup=5,
                        binned_enabled=False, excess_bins=None,
                        adaptive_bin_corrections=None):
    """
    Process one step of adaptive noise calibration.

    Call after getting raw model output and before noise injection.
    Handles warmup collection, calibration trigger, and correction application.

    Returns:
        (effective_s_noise, change_norm, adaptive_correction, should_restart,
         adaptive_bin_corrections)
    """
    # Compute prediction change magnitude
    if prev_denoised_raw is not None:
        change_norm = torch.norm((denoised_raw - prev_denoised_raw).flatten(1), dim=1).mean().item()
    else:
        change_norm = None

    effective_s_noise = base_s_noise
    should_restart = False

    if sigma_next > 0:
        sigma_val = sigma.item() if torch.is_tensor(sigma) else float(sigma)
        sigma_next_val = sigma_next.item() if torch.is_tensor(sigma_next) else float(sigma_next)
        in_texture_phase = 0.5 < sigma_val < 5.0

        if adaptive_correction is not None:
            # Post-restart: apply correction
            if binned_enabled and adaptive_bin_corrections is not None:
                correction = get_phase_correction(sigma_val, adaptive_bin_corrections, adaptive_correction)
                effective_s_noise *= correction
            else:
                effective_s_noise *= adaptive_correction
        elif change_norm is not None and prev_change_norm is not None:
            change_ratio = change_norm / (prev_change_norm + 1e-8)
            sigma_ratio = sigma_next_val / (sigma_val + 1e-8)
            excess = change_ratio / (sigma_ratio + 1e-8)

            # Bin the excess sample by sigma phase
            if binned_enabled and excess_bins is not None:
                if sigma_val > 5.0:
                    excess_bins['structural'].append(excess)
                elif sigma_val > 0.5:
                    excess_bins['texture'].append(excess)
                else:
                    excess_bins['cleanup'].append(excess)

            # Texture-phase samples drive the warmup trigger
            if in_texture_phase:
                excess_samples.append(excess)

                if len(excess_samples) >= warmup:
                    adaptive_correction, median_excess = compute_adaptive_noise_scale(
                        excess_samples, effective_s_noise
                    )
                    if binned_enabled and excess_bins is not None:
                        adaptive_bin_corrections = compute_binned_corrections(
                            excess_bins, adaptive_correction
                        )
                        bin_info = {k: f"{v:.3f}" for k, v in adaptive_bin_corrections.items()}
                        print(f"   Adaptive Noise calibrated: global={adaptive_correction:.3f}, binned={bin_info}")
                    else:
                        print(f"   Adaptive Noise calibrated: correction={adaptive_correction:.3f}")
                    should_restart = True

    return effective_s_noise, change_norm, adaptive_correction, should_restart, adaptive_bin_corrections


def get_ancestral_step(sigma, sigma_next, eta=1.):
    """Calculate ancestral step sizes."""
    sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
    sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def apply_spectral_modulation_clybius(noise_pred, multiplier=1.0, percentile=5.0):
    """
    Clybius Spectral Modulation: Apply frequency-domain corrections to noise prediction.

    Based on ComfyUI-Latent-Modifiers. Applied to noise_pred (cond - uncond),
    NOT to the denoised latent directly.

    Args:
        noise_pred: The noise prediction tensor (cond - uncond)
        multiplier: Modulation strength (0=none, 1=full Clybius effect). Default: 1.0
        percentile: Upper/lower percentile threshold. Default: 5.0

    Returns:
        Spectrally modulated noise prediction
    """
    if multiplier == 0 or percentile <= 0:
        return noise_pred

    try:
        fourier = torch.fft.fft2(noise_pred, dim=(-2, -1))
        log_amp = torch.log(torch.sqrt(fourier.real ** 2 + fourier.imag ** 2) + 1e-8)

        log_amp_flat = log_amp.abs().flatten(2)
        quantile_low = torch.quantile(log_amp_flat, percentile * 0.01, dim=2)
        quantile_high = torch.quantile(log_amp_flat, 1 - percentile * 0.01, dim=2)

        quantile_low = quantile_low.unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)
        quantile_high = quantile_high.unsqueeze(-1).unsqueeze(-1).expand(log_amp.shape)

        mask_low = ((log_amp < quantile_low).float() + 1).clamp_(max=1.5)
        mask_high = ((log_amp < quantile_high).float()).clamp_(min=0.5)

        filtered_fourier = fourier * ((mask_low * mask_high) ** multiplier)
        result = torch.fft.ifft2(filtered_fourier, dim=(-2, -1)).real

        return result

    except Exception as e:
        print(f"⚠️ Spectral modulation failed: {e}")
        return noise_pred


def apply_combat_cfg_drift(latent, method='mean', intensity=1.0):
    """
    Combat CFG Drift: Reduce mean drift from high CFG values.

    Based on ComfyUI-Latent-Modifiers. As CFG increases, the latent mean
    can drift away from 0, causing color shifts and other artifacts.

    Note: Disable for inpaint/ADetailer passes to prevent patchy composites.

    Args:
        latent: The latent tensor to correct
        method: 'mean' or 'median'. Default: 'mean'
        intensity: How much drift to remove (0=none, 1=full). Default: 1.0

    Returns:
        Drift-corrected latent
    """
    if intensity <= 0:
        return latent

    try:
        if method == 'median':
            center = latent.view(latent.shape[0], -1).median(dim=-1, keepdim=True)[0]
            center = center.view(latent.shape[0], 1, 1, 1)
        else:
            center = latent.mean(dim=(1, 2, 3), keepdim=True)

        return latent - center * intensity

    except Exception as e:
        print(f"⚠️ Combat CFG drift failed: {e}")
        return latent


def apply_cfg_techniques(denoised, settings):
    """
    Apply enabled post-hoc CFG techniques to the denoised prediction.

    Spectral Modulation is handled separately via the AdeptSpectralModulation
    model patch node. This function handles Combat CFG Drift only.

    Args:
        denoised: The denoised prediction from the model
        settings: Dict with 'combat_cfg_drift' (bool) and 'combat_drift_intensity' (float)

    Returns:
        Enhanced denoised result
    """
    if not settings.get('combat_cfg_drift', False):
        return denoised

    intensity = settings.get('combat_drift_intensity', 0.5)
    return apply_combat_cfg_drift(denoised, method='mean', intensity=intensity)


def create_detail_enhanced_model(model, x, sigmas, settings):
    """Creates a model wrapper with detail enhancement."""
    if not TORCHVISION_AVAILABLE:
        return model
    
    base_strength = settings.get('detail_enhancement_strength', 0.05)
    radius = settings.get('detail_separation_radius', 0.5)
    total_steps = len(sigmas) - 1
    
    class DetailEnhancer:
        def __init__(self):
            self.current_step = 0
            
        def __call__(self, x_current, sigma, **kwargs):
            denoised = model(x_current, sigma, **kwargs)
            
            try:
                low_freq = gaussian_blur(denoised, kernel_size=3, sigma=radius)
                high_freq = denoised - low_freq
                
                progress = min(self.current_step / max(total_steps, 1), 1.0)
                strength = base_strength * (0.5 + progress)
                
                enhanced = denoised + high_freq * strength
                self.current_step += 1
                
                return enhanced
            except Exception as e:
                print(f"⚠️ Detail enhancement failed: {e}")
                return denoised
    
    return DetailEnhancer()
