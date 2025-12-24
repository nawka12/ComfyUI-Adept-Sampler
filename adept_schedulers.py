"""
Adept Scheduler implementations for ComfyUI.
Ported from Stable Diffusion WebUI reForge extension.
"""

import math
import torch


def create_aos_v_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """AOS-V (Anime-Optimized Schedule for v-prediction models)."""
    rho = 7.0
    
    p1_steps = int(num_steps * 0.2)
    p2_steps = int(num_steps * 0.6)
    
    ramp = torch.empty(num_steps, device=device, dtype=torch.float32)
    
    if p1_steps > 0:
        torch.linspace(0, 1, p1_steps, out=ramp[:p1_steps])
        ramp[:p1_steps].pow_(0.5).mul_(0.6)
    
    if p2_steps > p1_steps:
        torch.linspace(0.6, 0.9, p2_steps - p1_steps, out=ramp[p1_steps:p2_steps])
    
    if num_steps > p2_steps:
        torch.linspace(0, 1, num_steps - p2_steps, out=ramp[p2_steps:])
        ramp[p2_steps:].pow_(3).mul_(0.1).add_(0.9)
    
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    ramp.mul_(min_inv_rho - max_inv_rho).add_(max_inv_rho).pow_(rho)
    
    return torch.cat([ramp, torch.zeros(1, device=device)])


def create_aos_e_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """AOS-ε (Anime-Optimized Schedule for epsilon-prediction models)."""
    rho = 7.0
    
    p1_frac, p2_frac = 0.35, 0.7
    ramp_p1_val, ramp_p2_val = 0.4, 0.75

    p1_steps = int(num_steps * p1_frac)
    p2_steps = int(num_steps * p2_frac)

    phase1_ramp = torch.linspace(0, 1, p1_steps, device=device) ** 1.5 * ramp_p1_val
    phase2_ramp = torch.linspace(ramp_p1_val, ramp_p2_val, p2_steps - p1_steps, device=device)
    phase3_base = torch.linspace(0, 1, num_steps - p2_steps, device=device) ** 0.7
    phase3_ramp = phase3_base * (1 - ramp_p2_val) + ramp_p2_val
    
    if p1_steps == 0: phase1_ramp = torch.empty(0, device=device)
    if p2_steps - p1_steps == 0: phase2_ramp = torch.empty(0, device=device)
    if num_steps - p2_steps == 0: phase3_ramp = torch.empty(0, device=device)

    ramp = torch.cat([phase1_ramp, phase2_ramp, phase3_ramp])
    
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_aos_akashic_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """
    AkashicAOS v2: Detail-Progressive Schedule for EQ-VAE SDXL models.
    Single continuous curve with progressive detail bias.
    """
    rho = 7.0
    
    u = torch.linspace(0, 1, num_steps, device=device)
    
    # Detail-progressive transformation
    detail_power = 0.85
    u_progressive = u ** detail_power
    
    # Mid-range enhancement
    mid_boost_strength = 0.08
    mid_boost = mid_boost_strength * torch.sin(math.pi * u) * (1 - u * 0.5)
    
    u_modulated = u_progressive + mid_boost
    
    # Normalize to [0, 1]
    u_min, u_max = u_modulated.min(), u_modulated.max()
    if u_max - u_min > 1e-8:
        u_modulated = (u_modulated - u_min) / (u_max - u_min)
    
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + u_modulated * (min_inv_rho - max_inv_rho)) ** rho
    
    # Step ratio smoothing
    for i in range(1, len(sigmas)):
        if sigmas[i] >= sigmas[i-1]:
            sigmas[i] = sigmas[i-1] * 0.995
        max_ratio = 1.5
        if i > 0 and sigmas[i-1] / sigmas[i] > max_ratio:
            sigmas[i] = sigmas[i-1] / max_ratio
    
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_entropic_sigmas(sigma_max, sigma_min, num_steps, power=3.0, device='cpu'):
    """Entropic power schedule: blends linear with power-based curve."""
    rho = 7.0
    
    linear_ramp = torch.linspace(0, 1, num_steps, device=device)
    power_ramp = 1 - torch.linspace(1, 0, num_steps, device=device) ** power
    
    ramp = (linear_ramp + power_ramp) / 2.0
    
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_snr_optimized_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """Schedule optimized around log SNR = 0 region."""
    rho = 7.0
    
    log_snr_max = 2 * torch.log(sigma_max)
    log_snr_min = 2 * torch.log(sigma_min)
    
    t = torch.linspace(0, 1, num_steps, device=device)
    
    concentration_power = 3.0
    sigmoid_t = torch.sigmoid(concentration_power * (t - 0.5))
    
    linear_t = t
    blend_factor = 0.7
    combined_t = blend_factor * sigmoid_t + (1 - blend_factor) * linear_t
    
    log_snr = log_snr_max + combined_t * (log_snr_min - log_snr_max)
    sigmas = torch.exp(log_snr / 2)
    
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_constant_rate_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """Constant rate of distributional change throughout sampling."""
    rho = 7.0
    
    t = torch.linspace(0, 1, num_steps, device=device)
    corrected_t = t + 0.3 * torch.sin(math.pi * t) * (1 - t)
    
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + corrected_t * (min_inv_rho - max_inv_rho)) ** rho
    
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_adaptive_optimized_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """Adaptive schedule combining multiple strategies."""
    rho = 7.0
    
    base_t = torch.linspace(0, 1, num_steps, device=device)
    
    strategies = [
        lambda t: t,
        lambda t: t ** 0.8,
        lambda t: t + 0.2 * torch.sin(2 * math.pi * t) * (1 - t),
        lambda t: 1 / (1 + torch.exp(-3 * (t - 0.5))),
    ]
    
    weights = [0.2, 0.3, 0.2, 0.3]
    combined_t = sum(w * s(base_t) for w, s in zip(weights, strategies))
    
    if (combined_t.max() - combined_t.min()) > 1e-6:
        combined_t = (combined_t - combined_t.min()) / (combined_t.max() - combined_t.min())
    
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + combined_t * (min_inv_rho - max_inv_rho)) ** rho
    
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_cosine_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """Cosine-annealed schedule: smooth start, strong early drop, gentle tail."""
    rho = 7.0
    u = torch.linspace(0, 1, num_steps, device=device)
    t = (1 - torch.cos(math.pi * u)) / 2
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_logsnr_uniform_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """Uniform in log-SNR space for a neutral, theory-aligned schedule."""
    u = torch.linspace(0, 1, num_steps, device=device)
    log_snr_max = 2 * torch.log(sigma_max)
    log_snr_min = 2 * torch.log(sigma_min)
    log_snr = log_snr_max + u * (log_snr_min - log_snr_max)
    sigmas = torch.exp(log_snr / 2)
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_tanh_midboost_sigmas(sigma_max, sigma_min, num_steps, device='cpu', k=4.0):
    """Concentrate steps near mid-range sigmas using tanh shaping."""
    rho = 7.0
    u = torch.linspace(0, 1, num_steps, device=device)
    k_tensor = torch.tensor(k, device=device, dtype=u.dtype)
    t = 0.5 * (torch.tanh(k_tensor * (u - 0.5)) / torch.tanh(k_tensor / 2) + 1.0)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_exponential_tail_sigmas(sigma_max, sigma_min, num_steps, device='cpu', pivot=0.7, gamma=0.8, beta=5.0):
    """Faster early lock-in with extra resolution in the final steps."""
    rho = 7.0
    u = torch.linspace(0, 1, num_steps, device=device)
    pivot_tensor = torch.tensor(pivot, device=device, dtype=u.dtype)
    gamma_tensor = torch.tensor(gamma, device=device, dtype=u.dtype)
    beta_tensor = torch.tensor(beta, device=device, dtype=u.dtype)

    front = (u / pivot_tensor).clamp(0, 1) ** gamma_tensor * pivot_tensor
    tail_raw = 1 - torch.exp(-beta_tensor * (u - pivot_tensor)).clamp(min=0)
    tail = pivot_tensor + (1 - pivot_tensor) * (tail_raw / (1 - torch.exp(-beta_tensor)))
    t = torch.where(u < pivot_tensor, front, tail)

    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_jittered_karras_sigmas(sigma_max, sigma_min, num_steps, device='cpu', jitter_strength=0.5):
    """Karras baseline with stratified jitter to reduce resonance/banding."""
    rho = 7.0
    indices = torch.arange(num_steps, device=device, dtype=torch.float32)
    rand = (torch.rand(num_steps, device=device) - 0.5) * jitter_strength
    denom = max(1, num_steps - 1)
    u = (indices + 0.5 + rand).clamp_(0, num_steps - 1) / denom

    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + u * (min_inv_rho - max_inv_rho)) ** rho

    sigmas, _ = torch.sort(sigmas, descending=True)
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_stochastic_sigmas(sigma_max, sigma_min, num_steps, device='cpu', noise_type='brownian', noise_scale=0.3, base_schedule='karras'):
    """Stochastic scheduler with controlled randomness in timestep selection."""
    rho = 7.0

    if base_schedule == 'uniform':
        u_base = torch.linspace(0, 1, num_steps, device=device)
    elif base_schedule == 'cosine':
        u_base = (1 - torch.cos(torch.pi * torch.linspace(0, 1, num_steps, device=device))) / 2
    else:  # 'karras'
        u_base = torch.linspace(0, 1, num_steps, device=device)

    if noise_type == 'brownian':
        noise = torch.randn(num_steps, device=device)
        brownian_noise = torch.cumsum(noise, dim=0)
        brownian_noise = (brownian_noise - brownian_noise.min()) / (brownian_noise.max() - brownian_noise.min() + 1e-8)
        perturbation = (brownian_noise - 0.5) * noise_scale
    elif noise_type == 'normal':
        perturbation = torch.randn(num_steps, device=device) * noise_scale
    else:  # 'uniform'
        perturbation = (torch.rand(num_steps, device=device) - 0.5) * 2 * noise_scale

    u_stochastic = torch.clamp(u_base + perturbation, 0.0, 1.0)

    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + u_stochastic * (min_inv_rho - max_inv_rho)) ** rho

    sigmas, _ = torch.sort(sigmas, descending=True)
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def create_jys_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """
    JYS (Jump Your Steps) schedule using dynamically computed timestep sequences.
    Strategy: Large jumps early, dense clustering in detail region, fine steps at end.
    """
    jys_timesteps = _compute_jys_timesteps(num_steps)

    rho = 7.0

    normalized_timesteps = [(1000 - t) / 1000.0 for t in jys_timesteps]
    t_tensor = torch.tensor(normalized_timesteps, device=device, dtype=torch.float32)

    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + t_tensor * (min_inv_rho - max_inv_rho)) ** rho

    sigmas, _ = torch.sort(sigmas, descending=True)
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def _compute_jys_timesteps(num_steps):
    """Dynamically compute optimized JYS timestep sequence."""
    if num_steps <= 0:
        return [0]
    if num_steps == 1:
        return [1000, 0]
    elif num_steps == 2:
        return [1000, 500, 0]
    elif num_steps == 3:
        return [1000, 600, 200, 0]

    early_steps = max(1, int(num_steps * 0.2))
    final_steps = max(1, int(num_steps * 0.2))
    middle_steps = max(1, num_steps - early_steps - final_steps)

    early_jump_size = max(50, (1000 - 600) // early_steps)
    early_timesteps = []
    current_t = 1000
    for i in range(early_steps):
        early_timesteps.append(int(current_t))
        current_t = max(600, current_t - early_jump_size)

    middle_timesteps = []
    structure_steps = max(1, middle_steps // 2)
    structure_jump_size = max(10, (600 - 300) // structure_steps)
    current_t = 600
    for i in range(structure_steps):
        middle_timesteps.append(int(current_t))
        current_t = max(300, current_t - structure_jump_size)

    detail_steps = middle_steps - structure_steps
    if detail_steps > 0:
        detail_start = 300
        detail_end = 200
        detail_jump_size = max(5, (detail_start - detail_end) // detail_steps)
        current_t = detail_start
        for i in range(detail_steps):
            middle_timesteps.append(int(current_t))
            current_t = max(detail_end, current_t - detail_jump_size)

    final_timesteps = []
    final_start = min(middle_timesteps) if middle_timesteps else 200
    final_jump_size = max(5, final_start // final_steps)
    current_t = final_start
    for i in range(final_steps):
        final_timesteps.append(int(current_t))
        current_t = max(0, current_t - final_jump_size)

    all_timesteps = early_timesteps + middle_timesteps + final_timesteps
    unique_timesteps = list(dict.fromkeys(all_timesteps))
    unique_timesteps.sort(reverse=True)

    while len(unique_timesteps) < num_steps:
        for i in range(len(unique_timesteps) - 1):
            mid_point = (unique_timesteps[i] + unique_timesteps[i + 1]) // 2
            if mid_point not in unique_timesteps:
                unique_timesteps.insert(i + 1, mid_point)
                if len(unique_timesteps) >= num_steps:
                    break

    if len(unique_timesteps) > num_steps:
        unique_timesteps = unique_timesteps[:num_steps]

    if unique_timesteps[-1] != 0:
        unique_timesteps.append(0)

    return unique_timesteps


def create_hybrid_jys_karras_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """Hybrid schedule: locks exposure like Jittered-Karras with JYS mid-phase detail density."""
    if num_steps <= 0:
        return torch.cat([sigma_max.unsqueeze(0), torch.zeros(1, device=device)])

    rho = 7.0

    jys_sigmas = create_jys_sigmas(sigma_max, sigma_min, num_steps, device=device)[:-1]

    indices = torch.arange(num_steps, device=device, dtype=torch.float32)
    denom = max(1, num_steps - 1)
    base = (indices + 0.5) / denom
    jitter_seed = torch.sin((indices + 1) * 2.3999632)
    jitter_strength = 0.35
    jitter = jitter_seed * jitter_strength / denom
    u = torch.clamp(base + jitter, 0.0, 1.0)

    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    karras_sigmas = (max_inv_rho + u * (min_inv_rho - max_inv_rho)) ** rho

    positions = torch.linspace(0, 1, num_steps, device=device)
    jys_weight = torch.empty_like(positions)
    early_mask = positions < 0.3
    mid_mask = (positions >= 0.3) & (positions < 0.8)
    late_mask = positions >= 0.8
    jys_weight[early_mask] = 0.2 + 0.4 * (positions[early_mask] / 0.3)
    jys_weight[mid_mask] = 0.6 + 0.3 * ((positions[mid_mask] - 0.3) / 0.5)
    jys_weight[late_mask] = 0.9
    jys_weight = jys_weight.clamp(0.2, 0.9)

    log_jys = torch.log(jys_sigmas.clamp_min(1e-6))
    log_karras = torch.log(karras_sigmas.clamp_min(1e-6))
    log_hybrid = torch.lerp(log_karras, log_jys, jys_weight)

    hybrid = torch.exp(log_hybrid)

    smoothing = 1.0 - 0.05 * (1 - positions) ** 2
    hybrid = hybrid * smoothing

    for i in range(1, hybrid.shape[0]):
        if hybrid[i] > hybrid[i - 1]:
            hybrid[i] = hybrid[i - 1] * 0.999

    return torch.cat([hybrid, torch.zeros(1, device=device)])


def create_ays_sdxl_sigmas(sigma_max, sigma_min, num_steps, device='cpu'):
    """
    AYS (Align Your Steps) optimized sigma schedule for SDXL.
    Based on NVIDIA's paper (CVPR 2024).
    """
    import numpy as np
    
    AYS_SCHEDULES = {
        10: [1.0000, 0.8751, 0.7502, 0.6254, 0.5004, 0.3755, 0.2506, 0.1253, 0.0502, 0.0000],
        15: [1.0000, 0.9167, 0.8334, 0.7501, 0.6668, 0.5835, 0.5002, 0.4169, 0.3336, 
             0.2503, 0.1670, 0.0837, 0.0335, 0.0084, 0.0000],
        20: [1.0000, 0.9375, 0.8750, 0.8125, 0.7500, 0.6875, 0.6250, 0.5625, 0.5000,
             0.4375, 0.3750, 0.3125, 0.2500, 0.1875, 0.1250, 0.0625, 0.0313, 0.0156, 
             0.0039, 0.0000],
        25: [1.0000, 0.9500, 0.9000, 0.8500, 0.8000, 0.7500, 0.7000, 0.6500, 0.6000,
             0.5500, 0.5000, 0.4500, 0.4000, 0.3500, 0.3000, 0.2500, 0.2000, 0.1500,
             0.1000, 0.0625, 0.0391, 0.0195, 0.0098, 0.0024, 0.0000],
        30: [1.0000, 0.9583, 0.9167, 0.8750, 0.8333, 0.7917, 0.7500, 0.7083, 0.6667,
             0.6250, 0.5833, 0.5417, 0.5000, 0.4583, 0.4167, 0.3750, 0.3333, 0.2917,
             0.2500, 0.2083, 0.1667, 0.1250, 0.0833, 0.0521, 0.0326, 0.0163, 0.0081,
             0.0041, 0.0010, 0.0000],
    }
    
    if num_steps in AYS_SCHEDULES:
        normalized = torch.tensor(AYS_SCHEDULES[num_steps], device=device, dtype=torch.float32)
    else:
        available_steps = sorted(AYS_SCHEDULES.keys())
        
        if num_steps < available_steps[0]:
            ref_steps = available_steps[0]
        elif num_steps > available_steps[-1]:
            ref_steps = available_steps[-1]
        else:
            ref_steps = min([s for s in available_steps if s >= num_steps], default=available_steps[-1])
        
        ref_schedule = np.array(AYS_SCHEDULES[ref_steps])
        
        t_ref = np.linspace(0, 1, len(ref_schedule))
        t_new = np.linspace(0, 1, num_steps + 1)
        
        log_ref = np.log(ref_schedule + 1e-8)
        log_ref[-1] = log_ref[-2] - 3.0
        
        log_interp = np.interp(t_new, t_ref, log_ref)
        normalized_np = np.exp(log_interp)
        normalized_np[-1] = 0.0
        
        normalized = torch.tensor(normalized_np, device=device, dtype=torch.float32)
    
    sigma_range = sigma_max - sigma_min
    sigmas = normalized * sigma_range + sigma_min
    
    sigmas[0] = sigma_max
    sigmas[-1] = 0.0
    
    for i in range(1, len(sigmas) - 1):
        if sigmas[i] >= sigmas[i-1]:
            sigmas[i] = sigmas[i-1] * 0.999
    
    return sigmas


# List of all available schedulers
SCHEDULER_NAMES = [
    "AOS-V",
    "AOS-ε", 
    "AkashicAOS",
    "Entropic",
    "SNR-Optimized",
    "Constant-Rate",
    "Adaptive-Optimized",
    "Cosine-Annealed",
    "LogSNR-Uniform",
    "Tanh Mid-Boost",
    "Exponential Tail",
    "Jittered-Karras",
    "Stochastic",
    "JYS (Dynamic)",
    "Hybrid JYS-Karras",
    "AYS-SDXL",
]


def get_scheduler_function(name):
    """Get scheduler function by name."""
    mapping = {
        "AOS-V": create_aos_v_sigmas,
        "AOS-ε": create_aos_e_sigmas,
        "AkashicAOS": create_aos_akashic_sigmas,
        "Entropic": create_entropic_sigmas,
        "SNR-Optimized": create_snr_optimized_sigmas,
        "Constant-Rate": create_constant_rate_sigmas,
        "Adaptive-Optimized": create_adaptive_optimized_sigmas,
        "Cosine-Annealed": create_cosine_sigmas,
        "LogSNR-Uniform": create_logsnr_uniform_sigmas,
        "Tanh Mid-Boost": create_tanh_midboost_sigmas,
        "Exponential Tail": create_exponential_tail_sigmas,
        "Jittered-Karras": create_jittered_karras_sigmas,
        "Stochastic": create_stochastic_sigmas,
        "JYS (Dynamic)": create_jys_sigmas,
        "Hybrid JYS-Karras": create_hybrid_jys_karras_sigmas,
        "AYS-SDXL": create_ays_sdxl_sigmas,
    }
    return mapping.get(name)
