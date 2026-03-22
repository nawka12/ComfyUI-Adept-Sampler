"""
Adept Sampler implementations for ComfyUI.
Ported from Stable Diffusion WebUI reForge extension.
"""

import torch
from .utils import (
    to_d,
    to_d_enhanced_ancestral,
    get_noise_sampler,
    apply_dynamic_thresholding,
    compute_compensation_ratio,
    compute_tau_eqvae,
    compute_eqvae_tau,
    compute_eqvae_noise_scale,
    compute_eqvae_ndb,
    compute_smea_factor,
    sa_solver_step,
    get_ancestral_step,
    create_detail_enhanced_model,
    apply_combat_cfg_drift,
    apply_cfg_techniques,
    adaptive_noise_step,
    TORCHVISION_AVAILABLE,
)


def sample_adept_solver(model, x, sigmas, extra_args=None, callback=None, disable=None, 
                        order=2, use_corrector=True, use_detail_enhancement=False, settings=None):
    """
    Adept Solver: A unified training-free diffusion solver synthesizing improvements from:
    - DPM-Solver++ (data prediction, dynamic thresholding)
    - UniPC (unified predictor-corrector framework)
    - DEIS (exponential integrator)
    - DC-Solver (dynamic compensation)
    """
    extra_args = {} if extra_args is None else extra_args
    settings = settings or {}
    s_in = x.new_ones([x.shape[0]])
    
    # Clamp order to valid range
    order = max(1, min(order, 3))
    
    print(f"🚀 Adept Solver active (Order: {order}, Corrector: {'On' if use_corrector else 'Off'})")
    
    # Apply detail enhancement wrapper if enabled
    active_model = model
    if use_detail_enhancement and TORCHVISION_AVAILABLE:
        active_model = create_detail_enhanced_model(model, x, sigmas, settings)
        print(f"🎨 Detail Enhancement: Model wrapper active")
    
    # Initialize history for multistep
    model_outputs = []
    
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # === PREDICTOR STEP ===
        denoised = active_model(x, sigma * s_in, **extra_args)
        
        # Apply dynamic thresholding for high CFG stability
        if extra_args.get('cond_scale', 1.0) > 7.0:
            denoised = apply_dynamic_thresholding(denoised, percentile=0.995)
        
        # Compute derivative
        d = to_d(x, sigma, denoised)

        # Safety check for extreme derivatives
        derivative_max = torch.abs(d).max()
        sigma_adaptive_threshold = 1000.0 * (1.0 + sigma / 10.0)
        if torch.isnan(d).any() or torch.isinf(d).any() or derivative_max > sigma_adaptive_threshold:
            print(f"⚠️ Extreme derivative detected at step {i}/{len(sigmas)-1}. Clamping for stability.")
            d = torch.clamp(d, -sigma_adaptive_threshold, sigma_adaptive_threshold)
            if torch.isnan(d).any() or torch.isinf(d).any():
                d = torch.zeros_like(d)

        # Store for multistep
        model_outputs.append((sigma, d))
        if len(model_outputs) > order:
            model_outputs.pop(0)
        
        # Compute predictor step using multistep Adams-Bashforth integration
        dt = sigma_next - sigma
        
        if len(model_outputs) == 1 or order == 1:
            # First-order (Euler step)
            x_pred = x + d * dt
        elif len(model_outputs) == 2 and order >= 2:
            # Second-order multistep with adaptive compensation
            sigma_prev, d_prev = model_outputs[-2]
            d_cur = model_outputs[-1][1]
            
            h = sigma - sigma_prev
            compensation_ratio = compute_compensation_ratio(h.item() if torch.is_tensor(h) else float(h), i, len(sigmas))
            
            d_interp = d_cur + compensation_ratio * (d_cur - d_prev)
            x_pred = x + d_interp * dt
        else:
            # Third-order multistep
            sigma_0, d_0 = model_outputs[-3]
            sigma_1, d_1 = model_outputs[-2]
            sigma_2, d_2 = model_outputs[-1]
            
            h_0 = sigma_2 - sigma_1
            h_1 = sigma_1 - sigma_0
            
            h_0_val = h_0.item() if torch.is_tensor(h_0) else float(h_0)
            h_1_val = h_1.item() if torch.is_tensor(h_1) else float(h_1)
            
            if abs(h_1_val) < 1e-6:
                compensation_ratio = compute_compensation_ratio(h_0_val, i, len(sigmas))
                d_interp = d_2 + compensation_ratio * (d_2 - d_1)
            else:
                r0 = h_0_val / h_1_val
                c0 = 1.0 + r0 / 2.0
                c1 = -r0 / 2.0
                c2 = 0.0
                
                c_sum = c0 + c1 + c2
                c0 /= c_sum
                c1 /= c_sum
                c2 = 1.0 - c0 - c1
                
                d_interp = c0 * d_2 + c1 * d_1 + c2 * d_0
            
            x_pred = x + d_interp * dt
        
        # === CORRECTOR STEP (optional) ===
        if use_corrector and i < len(sigmas) - 2:
            denoised_pred = active_model(x_pred, sigma_next * s_in, **extra_args)
            
            if extra_args.get('cond_scale', 1.0) > 7.0:
                denoised_pred = apply_dynamic_thresholding(denoised_pred, percentile=0.995)

            d_pred = to_d(x_pred, sigma_next, denoised_pred)

            if torch.isnan(d_pred).any() or torch.isinf(d_pred).any() or torch.abs(d_pred).max() > 1000.0:
                d_pred = torch.clamp(d_pred, -100.0, 100.0)
                if torch.isnan(d_pred).any() or torch.isinf(d_pred).any():
                    d_pred = torch.zeros_like(d_pred)
            
            # Trapezoidal rule
            dt = sigma_next - sigma
            x = x + (d + d_pred) * dt * 0.5
        else:
            x = x_pred
        
        # Error handling
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"❌ CRITICAL: NaN/Inf detected at step {i}/{len(sigmas)-1}!")
            if i == 0:
                raise RuntimeError("NaN/Inf on first step - check model/inputs")
            
            print("   Attempting recovery with conservative Euler step...")
            denoised_safe = active_model(x, sigma * s_in, **extra_args)
            if torch.isnan(denoised_safe).any():
                raise RuntimeError("Model producing NaN - check CFG scale and model")
            
            d_safe = to_d(x, sigma, denoised_safe)
            dt_safe = (sigma_next - sigma) * 0.5
            x = x + d_safe * dt_safe
            use_corrector = False
            print("   Recovery successful. Corrector disabled for stability.")
        
        # Callback for progress tracking
        if callback is not None:
            callback(i, denoised, x, len(sigmas) - 1)
    
    return x


def sample_adept_ancestral_solver(model, x, sigmas, extra_args=None, callback=None, disable=None,
                                   eta=1.0, s_noise=1.0, adaptive_eta=False, phase_noise=False,
                                   phase_strength=0.5, enhanced_derivative=False,
                                   use_detail_enhancement=False, settings=None,
                                   adaptive_noise=False):
    """
    Enhanced Adept Ancestral Solver: Advanced ancestral sampling with phase-aware adaptations.

    Key innovations:
    1. Adaptive ancestral step sizing that changes throughout sampling phases
    2. Phase-aware noise injection (more noise early, less noise late)
    3. Enhanced derivative computation with ancestral-specific corrections
    4. Dynamic eta scheduling for better control
    """
    extra_args = {} if extra_args is None else extra_args
    settings = settings or {}
    s_in = x.new_ones([x.shape[0]])

    print(f"🚀 Enhanced Adept Ancestral Solver active (η: {eta:.2f}, s_noise: {s_noise:.2f})")
    print(f"   Adaptive Eta: {adaptive_eta}, Phase Noise: {phase_noise}, Enhanced Derivative: {enhanced_derivative}")
    if adaptive_noise:
        print(f"   Adaptive Noise: ON (auto-adjusting s_noise per step)")

    # Apply detail enhancement wrapper if enabled
    active_model = model
    if use_detail_enhancement and TORCHVISION_AVAILABLE:
        active_model = create_detail_enhanced_model(model, x, sigmas, settings)
        print(f"🎨 Detail Enhancement: Model wrapper active")

    # Get noise sampler
    noise_sampler = get_noise_sampler(x)

    # Initialize history
    model_outputs = []
    n_steps = len(sigmas) - 1

    # Adaptive noise state
    prev_denoised_raw = None
    prev_change_norm = None
    excess_samples = []
    adaptive_correction = None
    excess_bins = {'structural': [], 'texture': [], 'cleanup': []} if adaptive_noise else None
    adaptive_bin_corrections = None
    x_initial = x.clone() if adaptive_noise else None

    i = 0
    while i < n_steps:
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        progress = i / max(n_steps, 1)

        # === ADAPTIVE ETA SCHEDULING ===
        if adaptive_eta:
            if progress < 0.3:
                current_eta = eta * 1.08
            elif progress < 0.7:
                current_eta = eta * 0.95
            else:
                current_eta = eta * 1.02
        else:
            current_eta = eta

        # === PREDICTOR STEP ===
        denoised = active_model(x, sigma * s_in, **extra_args)
        denoised_raw = denoised.clone() if adaptive_noise else None

        if extra_args.get('cond_scale', 1.0) > 7.0:
            denoised = apply_dynamic_thresholding(denoised, percentile=0.995)

        # === DERIVATIVE COMPUTATION ===
        if enhanced_derivative:
            d = to_d_enhanced_ancestral(x, sigma, denoised, current_eta, progress, None)
        else:
            d = to_d(x, sigma, denoised)

        # Safety check
        derivative_max = torch.abs(d).max()
        sigma_adaptive_threshold = 1000.0 * (1.0 + sigma / 10.0)
        if torch.isnan(d).any() or torch.isinf(d).any() or derivative_max > sigma_adaptive_threshold:
            d = torch.clamp(d, -sigma_adaptive_threshold, sigma_adaptive_threshold)
            if torch.isnan(d).any() or torch.isinf(d).any():
                d = torch.zeros_like(d)

        model_outputs.append((sigma, d))
        if len(model_outputs) > 1:
            model_outputs.pop(0)

        # === ADAPTIVE ANCESTRAL STEP ===
        if sigma_next > 0:
            sigma_up = min(sigma_next, current_eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
            sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
        else:
            sigma_up = 0.0
            sigma_down = 0.0

        dt = sigma_down - sigma
        x_pred = x + d * dt

        # === PHASE-AWARE NOISE INJECTION ===
        if sigma_next > 0:
            if phase_noise:
                if progress < 0.25:
                    target_multiplier = 1.0 + (0.05 * min(progress / 0.25, 1.0))
                elif progress < 0.6:
                    target_multiplier = 1.0 - (0.02 * min((progress - 0.25) / 0.35, 1.0))
                else:
                    target_multiplier = 1.0 - (0.05 * min((progress - 0.6) / 0.4, 1.0))

                noise_multiplier = 1.0 + (target_multiplier - 1.0) * phase_strength
                adaptive_s_noise = s_noise * noise_multiplier
            else:
                adaptive_s_noise = s_noise

            # Adaptive noise: modulate final s_noise
            if adaptive_noise:
                adaptive_s_noise, change_norm, adaptive_correction, should_restart, adaptive_bin_corrections = adaptive_noise_step(
                    denoised_raw, prev_denoised_raw, prev_change_norm,
                    sigma, sigma_next, excess_samples, adaptive_correction,
                    adaptive_s_noise, i, n_steps,
                    binned_enabled=True, excess_bins=excess_bins,
                    adaptive_bin_corrections=adaptive_bin_corrections
                )
                if should_restart and x_initial is not None:
                    x = x_initial.clone()
                    model_outputs = []
                    prev_denoised_raw = None
                    prev_change_norm = None
                    excess_bins = {'structural': [], 'texture': [], 'cleanup': []}
                    x_initial = None
                    i = 0
                    continue
                prev_change_norm = change_norm
                prev_denoised_raw = denoised_raw

            noise = noise_sampler(sigma, sigma_next) * adaptive_s_noise * sigma_up
            x = x_pred + noise
        else:
            x = x_pred

        # Error handling
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"❌ CRITICAL: NaN/Inf detected at step {i}/{n_steps}!")
            if i == 0:
                raise RuntimeError("NaN/Inf on first step - check model/inputs")

            print("   Attempting recovery...")
            denoised_safe = active_model(x, sigma * s_in, **extra_args)
            if torch.isnan(denoised_safe).any():
                raise RuntimeError("Model producing NaN - check CFG scale and model")

            d_safe = to_d(x, sigma, denoised_safe)
            dt_safe = (sigma_next - sigma) * 0.5
            x = x + d_safe * dt_safe
            if adaptive_noise:
                prev_denoised_raw = None
                prev_change_norm = None
            print("   Recovery successful.")

        if callback is not None:
            callback(i, denoised, x, n_steps)

        i += 1

    return x


def sample_mirror_correction_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                                    eta=1.0, s_noise=1.0, correction_phase=0.5, smooth_phase=False,
                                    adaptive_noise=False):
    """
    Mirror Correction Euler: plain Euler Ancestral with a semantic reflection probe.

    In the first `correction_phase` fraction of steps, uses a 3-call Heun correction:
      x_probe = 2*D(x) - x  (reflection of x through its own denoised prediction)
    Unlike a naive -x probe (where x terms cancel), this probe lies on the denoising
    trajectory, giving a meaningful curvature estimate for the Heun correction.

    Remaining steps: standard 1-call Euler Ancestral. Ancestral noise at every step.

    Args:
        eta: Ancestral noise coefficient. 0=deterministic, 1=full ancestral. Default: 1.0
        s_noise: Noise scale multiplier. Default: 1.0
        correction_phase: Fraction of steps that get the 3-call correction.
            0.0=no correction (plain Euler a), 1.0=all steps. Default: 0.5
        smooth_phase: If True, replaces binary cutoff with continuous log-sigma weight
            for smoother phase transitions. Default: False
        adaptive_noise: If True, auto-adjusts s_noise based on model behavior. Default: False
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    _probe_norm_limit = 5.0  # guard against out-of-distribution probe derivatives

    print(f"🔮 Mirror Correction Euler active (η: {eta:.2f}, s_noise: {s_noise:.2f})")
    print(f"   Correction Phase: {correction_phase:.2f}, Smooth Phase: {smooth_phase}")
    if adaptive_noise:
        print(f"   Adaptive Noise: ON (auto-adjusting s_noise per step)")

    noise_sampler = get_noise_sampler(x)
    n_steps = len(sigmas) - 1

    # Pre-compute log-sigma bounds for smooth phase mode
    log_sigma_phase = None
    log_sigma_max = None
    smooth_denom = 1e-6
    if smooth_phase and n_steps > 0:
        sigma_max_val = sigmas[0].clamp(min=1e-6)
        phase_idx = min(int(correction_phase * n_steps), n_steps - 1)
        sigma_phase_val = sigmas[phase_idx].clamp(min=1e-6)
        log_sigma_max = torch.log(sigma_max_val).item()
        log_sigma_phase = torch.log(sigma_phase_val).item()
        smooth_denom = max(log_sigma_max - log_sigma_phase, 1e-6)

    # Adaptive noise state
    prev_denoised_raw = None
    prev_change_norm = None
    excess_samples = []
    adaptive_correction = None
    excess_bins = {'structural': [], 'texture': [], 'cleanup': []} if adaptive_noise else None
    adaptive_bin_corrections = None
    x_initial = x.clone() if adaptive_noise else None

    i = 0
    while i < n_steps:
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        progress = i / max(n_steps - 1, 1)

        denoised = model(x, sigma * s_in, **extra_args)
        denoised_raw = denoised.clone() if adaptive_noise else None
        if callback is not None:
            callback(i, denoised, x, n_steps)

        d = to_d(x, sigma, denoised)

        # Ancestral step decomposition (standard formula)
        if sigma_next > 0:
            sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
            sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
        else:
            sigma_up = 0.0
            sigma_down = 0.0
        dt = sigma_down - sigma

        if smooth_phase and log_sigma_phase is not None:
            # Smooth mode: log-sigma weight + soft blend
            log_sig = torch.log(sigma.clamp(min=1e-6)).item()
            t = max(0.0, min(1.0, (log_sig - log_sigma_phase) / smooth_denom))
            correction_weight = t ** 0.5  # sqrt curve: steep early, shallow near phase_end

            if correction_weight > 1e-3 and sigma_next > 0:
                x_probe = 2 * denoised - x
                d_probe = to_d(x_probe, sigma, model(x_probe, sigma * s_in, **extra_args))
                d_norm = d.norm()
                d_probe_norm = d_probe.norm()
                if d_norm > 0 and d_probe_norm > _probe_norm_limit * d_norm:
                    d_probe = d_probe * (_probe_norm_limit * d_norm / d_probe_norm)
                x3 = x + ((d + d_probe) / 2) * dt
                d3 = to_d(x3, sigma, model(x3, sigma * s_in, **extra_args))
                d3_norm = d3.norm()
                if d_norm > 0 and d3_norm > _probe_norm_limit * d_norm:
                    d3 = d3 * (_probe_norm_limit * d_norm / d3_norm)
                d_heun = (d + d3) / 2
                if not (torch.isnan(d_heun).any() or torch.isinf(d_heun).any()):
                    d = d + correction_weight * (d_heun - d)  # soft blend
        else:
            # Binary mode
            if progress < correction_phase and sigma_next > 0:
                x_probe = 2 * denoised - x
                d_probe = to_d(x_probe, sigma, model(x_probe, sigma * s_in, **extra_args))
                # Guard: scale down probe derivative if wildly larger than d
                d_norm = d.norm()
                d_probe_norm = d_probe.norm()
                if d_norm > 0 and d_probe_norm > _probe_norm_limit * d_norm:
                    d_probe = d_probe * (_probe_norm_limit * d_norm / d_probe_norm)
                x3 = x + ((d + d_probe) / 2) * dt
                d3 = to_d(x3, sigma, model(x3, sigma * s_in, **extra_args))
                d3_norm = d3.norm()
                if d_norm > 0 and d3_norm > _probe_norm_limit * d_norm:
                    d3 = d3 * (_probe_norm_limit * d_norm / d3_norm)
                d = (d + d3) / 2
                if torch.isnan(d).any() or torch.isinf(d).any():
                    d = torch.zeros_like(d)

        x = x + d * dt

        # Adaptive noise: compute effective s_noise
        effective_s_noise = s_noise
        if adaptive_noise:
            effective_s_noise, change_norm, adaptive_correction, should_restart, adaptive_bin_corrections = adaptive_noise_step(
                denoised_raw, prev_denoised_raw, prev_change_norm,
                sigma, sigma_next, excess_samples, adaptive_correction,
                effective_s_noise, i, n_steps,
                binned_enabled=True, excess_bins=excess_bins,
                adaptive_bin_corrections=adaptive_bin_corrections
            )
            if should_restart and x_initial is not None:
                x = x_initial.clone()
                prev_denoised_raw = None
                prev_change_norm = None
                excess_bins = {'structural': [], 'texture': [], 'cleanup': []}
                x_initial = None
                i = 0
                continue
            prev_change_norm = change_norm
            prev_denoised_raw = denoised_raw

        if sigma_next > 0:
            x = x + noise_sampler(sigma, sigma_next) * effective_s_noise * sigma_up

        i += 1

    return x


def sample_akashic_solver(model, x, sigmas, extra_args=None, callback=None, disable=None,
                           tau=0.5, eta=1.0, s_noise=1.0, adaptive_eta=True, phase_strength=0.5,
                           order=2, smea_strength=0.0, ndb_strength=0.0,
                           use_detail_enhancement=False, settings=None, eqvae_mode='Off',
                           combat_cfg_drift=False, combat_drift_intensity=0.5,
                           adaptive_noise=False):
    """
    AkashicSolver v2 [EXPERIMENTAL]: Advanced sampler optimized for EQ-VAE models.

    Combines:
    1. SA-SOLVER BASE: Multi-step Adams-Bashforth integration with tau function
    2. PHASE-AWARE SAMPLING: Three-phase approach with adaptive parameters
    3. SMEA COHERENCY: Sine-based interpolation for high-resolution coherency

    Args:
        eqvae_mode: EQ-VAE optimization mode ('Off' or 'Balanced')
        adaptive_noise: Auto-adjust s_noise based on model behavior. Default: False
    """
    extra_args = {} if extra_args is None else extra_args
    settings = settings or {}
    s_in = x.new_ones([x.shape[0]])

    # Parse EQ-VAE mode setting
    if isinstance(eqvae_mode, bool):
        # Backwards compatibility with old boolean setting
        eqvae_enabled = eqvae_mode
    else:
        eqvae_enabled = eqvae_mode == 'Balanced'

    if eqvae_enabled:
        print(f"🌀 AkashicSolver v2 [EQ-VAE BALANCED] active")
        print(f"   Optimized for EQ-VAE's cleaner latent space")
    else:
        print(f"🌀 AkashicSolver v2 [EXPERIMENTAL] active")
    print(f"   τ (tau): {tau:.2f}, η (eta): {eta:.2f}, s_noise: {s_noise:.2f}")
    print(f"   Order: {order}, Adaptive Eta: {adaptive_eta}, Phase Strength: {phase_strength:.2f}")
    if smea_strength > 0:
        print(f"   SMEA: {smea_strength:.2f} (high-res coherency)")
    if ndb_strength > 0:
        print(f"   Native Detail Boost: {ndb_strength:.2f} (detail enhancement)")
    if combat_cfg_drift:
        print(f"   ✨ Combat CFG Drift: On (intensity: {combat_drift_intensity:.2f})")
    if adaptive_noise:
        print(f"   Adaptive Noise: ON (auto-adjusting s_noise per step)")
    if not eqvae_enabled:
        print(f"   ⚠️ Use external rescaleCFG (e.g., 0.7) for EQ-VAE models")

    # Apply detail enhancement wrapper if enabled
    active_model = model
    if use_detail_enhancement and TORCHVISION_AVAILABLE:
        active_model = create_detail_enhanced_model(model, x, sigmas, settings)
        print(f"🎨 Detail Enhancement: Model wrapper active")

    noise_sampler = get_noise_sampler(x)
    total_steps = len(sigmas) - 1
    d_history = []

    # Adaptive noise state
    prev_denoised_raw = None
    prev_change_norm = None
    excess_samples = []
    adaptive_correction = None
    excess_bins = {'structural': [], 'texture': [], 'cleanup': []} if adaptive_noise else None
    adaptive_bin_corrections = None
    x_initial = x.clone() if adaptive_noise else None

    i = 0
    while i < total_steps:
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        progress = i / max(total_steps - 1, 1)

        # === COMPUTE PHASE-AWARE TAU ===
        if adaptive_eta:
            if eqvae_enabled:
                current_tau = compute_eqvae_tau(progress, tau, phase_strength)
            else:
                current_tau = compute_tau_eqvae(progress, tau, phase_strength)
        else:
            current_tau = tau

        # === PHASE-AWARE ADAPTIVE ETA ===
        if adaptive_eta:
            if eqvae_enabled:
                if progress < 0.25:
                    current_eta = eta * (1.0 + 0.03 * phase_strength)
                elif progress < 0.55:
                    current_eta = eta * (1.0 - 0.03 * phase_strength)
                else:
                    current_eta = eta * (1.0 + 0.02 * phase_strength)
            else:
                if progress < 0.30:
                    current_eta = eta * (1.0 + 0.08 * phase_strength)
                elif progress < 0.60:
                    current_eta = eta * (1.0 - 0.05 * phase_strength)
                else:
                    current_eta = eta * (1.0 + 0.02 * phase_strength)
        else:
            current_eta = eta

        # SMEA factor
        smea_factor = compute_smea_factor(progress, smea_strength)

        # === MODEL PREDICTION ===
        denoised = active_model(x, sigma * s_in, **extra_args)

        # Store raw model output for adaptive noise (before any post-processing)
        denoised_raw = denoised.clone() if adaptive_noise else None

        cfg_scale = extra_args.get('cond_scale', 1.0)
        if cfg_scale > 7.0:
            denoised = apply_dynamic_thresholding(denoised, percentile=0.995)

        # === POST-HOC CFG FIXES (Combat Drift) ===
        if combat_cfg_drift:
            denoised = apply_combat_cfg_drift(denoised, intensity=combat_drift_intensity)

        # === COMPUTE DERIVATIVE ===
        d = to_d(x, sigma, denoised)

        derivative_max = torch.abs(d).max()
        sigma_adaptive_threshold = 1000.0 * (1.0 + sigma / 10.0)
        if torch.isnan(d).any() or torch.isinf(d).any() or derivative_max > sigma_adaptive_threshold:
            d = torch.clamp(d, -sigma_adaptive_threshold, sigma_adaptive_threshold)
            if torch.isnan(d).any() or torch.isinf(d).any():
                d = torch.zeros_like(d)

        d_history.append((sigma, d))
        if len(d_history) > order:
            d_history.pop(0)

        # === SA-SOLVER STEP WITH TAU CONTROL ===
        effective_tau = current_tau

        if eqvae_enabled:
            effective_s_noise = compute_eqvae_noise_scale(s_noise * current_eta, progress) * smea_factor
        else:
            effective_s_noise = s_noise * current_eta * smea_factor

            if progress < 0.30:
                noise_multiplier = 1.0 + 0.03 * phase_strength
            elif progress < 0.60:
                noise_multiplier = 1.0 - 0.01 * phase_strength
            else:
                noise_multiplier = 1.0 - 0.02 * phase_strength

            effective_s_noise *= noise_multiplier

        # === ADAPTIVE NOISE SCALE (final multiplier) ===
        if adaptive_noise and prev_denoised_raw is not None:
            change_norm = torch.norm((denoised_raw - prev_denoised_raw).flatten(1), dim=1).mean().item()
        else:
            change_norm = None

        if adaptive_noise and sigma_next > 0:
            sigma_val = sigma.item() if torch.is_tensor(sigma) else float(sigma)
            sigma_next_val = sigma_next.item() if torch.is_tensor(sigma_next) else float(sigma_next)
            in_texture_phase = 0.5 < sigma_val < 5.0

            if adaptive_correction is not None:
                # Post-restart: apply calibrated correction
                from .utils import get_phase_correction
                if adaptive_bin_corrections is not None:
                    correction = get_phase_correction(sigma_val, adaptive_bin_corrections, adaptive_correction)
                    effective_s_noise *= correction
                else:
                    effective_s_noise *= adaptive_correction

            elif change_norm is not None and prev_change_norm is not None:
                from .utils import compute_adaptive_noise_scale, compute_binned_corrections
                change_ratio = change_norm / (prev_change_norm + 1e-8)
                sigma_ratio = sigma_next_val / (sigma_val + 1e-8)
                excess = change_ratio / (sigma_ratio + 1e-8)

                # Bin the excess sample by sigma phase
                if excess_bins is not None:
                    if sigma_val > 5.0:
                        excess_bins['structural'].append(excess)
                    elif sigma_val > 0.5:
                        excess_bins['texture'].append(excess)
                    else:
                        excess_bins['cleanup'].append(excess)

                if in_texture_phase:
                    excess_samples.append(excess)

                    if len(excess_samples) >= 5:
                        adaptive_correction, median_excess = compute_adaptive_noise_scale(
                            excess_samples, effective_s_noise
                        )
                        if excess_bins is not None:
                            adaptive_bin_corrections = compute_binned_corrections(
                                excess_bins, adaptive_correction
                            )
                            bin_info = {k: f"{v:.3f}" for k, v in adaptive_bin_corrections.items()}
                            print(f"   Adaptive Noise calibrated: global={adaptive_correction:.3f}, binned={bin_info}")
                        else:
                            print(f"   Adaptive Noise calibrated: correction={adaptive_correction:.3f}")
                        # Restart generation with correction applied from step 0
                        if x_initial is not None:
                            x = x_initial.clone()
                            d_history = []
                            prev_denoised_raw = None
                            prev_change_norm = None
                            excess_bins = {'structural': [], 'texture': [], 'cleanup': []}
                            x_initial = None  # Prevent double restart
                            i = 0
                            continue
                        effective_s_noise *= adaptive_correction

        # Determine NDB parameters (EQ-VAE uses different blur sigma)
        if eqvae_enabled and ndb_strength > 0:
            eqvae_blur_sigma, _ = compute_eqvae_ndb(progress, ndb_strength)
        else:
            eqvae_blur_sigma = None

        # Execute SA-Solver step
        x, sigma_up = sa_solver_step(
            x=x,
            d_history=d_history,
            sigma=sigma,
            sigma_next=sigma_next,
            tau=effective_tau,
            s_noise=effective_s_noise,
            noise_sampler=noise_sampler,
            order=order,
            ndb_strength=ndb_strength,
            progress=progress,
            eqvae_mode=eqvae_enabled,
            eqvae_blur_sigma=eqvae_blur_sigma
        )

        # === ERROR HANDLING ===
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"❌ AkashicSolver v2: NaN/Inf detected at step {i}/{total_steps}!")

            if i == 0:
                raise RuntimeError("NaN/Inf on first step - check model/inputs")

            print("   Attempting recovery...")
            denoised_safe = active_model(x, sigma * s_in, **extra_args)
            if torch.isnan(denoised_safe).any():
                raise RuntimeError("Model producing NaN - reduce CFG scale or check model")

            d_safe = to_d(x, sigma, denoised_safe)
            dt_safe = (sigma_next - sigma) * 0.5
            x = x + d_safe * dt_safe

            d_history.clear()
            if adaptive_noise:
                prev_denoised_raw = None
                prev_change_norm = None
            print("   Recovery successful. Multi-step history cleared.")
        else:
            # Update adaptive noise state only on clean steps
            if adaptive_noise:
                if change_norm is not None:
                    prev_change_norm = change_norm
                prev_denoised_raw = denoised_raw

        if callback is not None:
            callback(i, denoised, x, total_steps)

        i += 1

    return x
