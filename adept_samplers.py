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
    compute_smea_factor,
    sa_solver_step,
    get_ancestral_step,
    create_detail_enhanced_model,
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
                                   use_detail_enhancement=False, settings=None):
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
    
    # Apply detail enhancement wrapper if enabled
    active_model = model
    if use_detail_enhancement and TORCHVISION_AVAILABLE:
        active_model = create_detail_enhanced_model(model, x, sigmas, settings)
        print(f"🎨 Detail Enhancement: Model wrapper active")
    
    # Get noise sampler
    noise_sampler = get_noise_sampler(x)
    
    # Initialize history
    model_outputs = []
    
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        progress = i / max(len(sigmas) - 1, 1)
        
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
            
            noise = noise_sampler(sigma, sigma_next) * adaptive_s_noise * sigma_up
            x = x_pred + noise
        else:
            x = x_pred
        
        # Error handling
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"❌ CRITICAL: NaN/Inf detected at step {i}/{len(sigmas)-1}!")
            if i == 0:
                raise RuntimeError("NaN/Inf on first step - check model/inputs")
            
            print("   Attempting recovery...")
            denoised_safe = active_model(x, sigma * s_in, **extra_args)
            if torch.isnan(denoised_safe).any():
                raise RuntimeError("Model producing NaN - check CFG scale and model")
            
            d_safe = to_d(x, sigma, denoised_safe)
            dt_safe = (sigma_next - sigma) * 0.5
            x = x + d_safe * dt_safe
            print("   Recovery successful.")
        
        if callback is not None:
            callback(i, denoised, x, len(sigmas) - 1)
    
    return x


def sample_akashic_solver(model, x, sigmas, extra_args=None, callback=None, disable=None,
                           tau=0.5, eta=1.0, s_noise=1.0, adaptive_eta=True, phase_strength=0.5,
                           order=2, smea_strength=0.0, ndb_strength=0.0, 
                           use_detail_enhancement=False, settings=None):
    """
    AkashicSolver v2 [EXPERIMENTAL]: Advanced sampler optimized for EQ-VAE models.
    
    Combines:
    1. SA-SOLVER BASE: Multi-step Adams-Bashforth integration with tau function
    2. PHASE-AWARE SAMPLING: Three-phase approach with adaptive parameters
    3. SMEA COHERENCY: Sine-based interpolation for high-resolution coherency
    """
    extra_args = {} if extra_args is None else extra_args
    settings = settings or {}
    s_in = x.new_ones([x.shape[0]])
    
    print(f"🌀 AkashicSolver v2 [EXPERIMENTAL] active")
    print(f"   τ (tau): {tau:.2f}, η (eta): {eta:.2f}, s_noise: {s_noise:.2f}")
    print(f"   Order: {order}, Adaptive Eta: {adaptive_eta}, Phase Strength: {phase_strength:.2f}")
    if smea_strength > 0:
        print(f"   SMEA: {smea_strength:.2f} (high-res coherency)")
    if ndb_strength > 0:
        print(f"   Native Detail Boost: {ndb_strength:.2f}")
    print(f"   ⚠️ Use external rescaleCFG (e.g., 0.7) for EQ-VAE models")
    
    # Apply detail enhancement wrapper if enabled
    active_model = model
    if use_detail_enhancement and TORCHVISION_AVAILABLE:
        active_model = create_detail_enhanced_model(model, x, sigmas, settings)
        print(f"🎨 Detail Enhancement: Model wrapper active")
    
    noise_sampler = get_noise_sampler(x)
    total_steps = len(sigmas) - 1
    d_history = []
    
    for i in range(total_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        progress = i / max(total_steps - 1, 1)
        
        # === PHASE-AWARE TAU ===
        if adaptive_eta:
            current_tau = compute_tau_eqvae(progress, tau, phase_strength)
        else:
            current_tau = tau
        
        # === PHASE-AWARE ADAPTIVE ETA ===
        if adaptive_eta:
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
        
        cfg_scale = extra_args.get('cond_scale', 1.0)
        if cfg_scale > 7.0:
            denoised = apply_dynamic_thresholding(denoised, percentile=0.995)
        
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
        
        # === SA-SOLVER STEP ===
        effective_tau = current_tau
        effective_s_noise = s_noise * current_eta * smea_factor
        
        # Phase-aware noise adjustment
        if progress < 0.30:
            noise_multiplier = 1.0 + 0.03 * phase_strength
        elif progress < 0.60:
            noise_multiplier = 1.0 - 0.01 * phase_strength
        else:
            noise_multiplier = 1.0 - 0.02 * phase_strength
        
        effective_s_noise *= noise_multiplier
        
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
            progress=progress
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
            print("   Recovery successful. Multi-step history cleared.")
        
        if callback is not None:
            callback(i, denoised, x, total_steps)
    
    return x
