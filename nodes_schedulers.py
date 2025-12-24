"""
ComfyUI Scheduler Nodes for Adept Sampler.
"""

import torch
from .adept_schedulers import (
    create_aos_v_sigmas,
    create_aos_e_sigmas,
    create_aos_akashic_sigmas,
    create_entropic_sigmas,
    create_jys_sigmas,
    create_snr_optimized_sigmas,
    create_constant_rate_sigmas,
    create_adaptive_optimized_sigmas,
    create_cosine_sigmas,
    create_logsnr_uniform_sigmas,
    create_tanh_midboost_sigmas,
    create_exponential_tail_sigmas,
    create_jittered_karras_sigmas,
    create_stochastic_sigmas,
    create_hybrid_jys_karras_sigmas,
    create_ays_sdxl_sigmas,
    SCHEDULER_NAMES,
)


class AdeptSchedulerAOS_V:
    """AOS-V (Anime-Optimized Schedule for v-prediction models)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/adept/schedulers"
    
    def get_sigmas(self, model, steps):
        sigma_min = model.get_model_object("model_sampling").sigma_min
        sigma_max = model.get_model_object("model_sampling").sigma_max
        device = model.load_device
        
        sigmas = create_aos_v_sigmas(sigma_max, sigma_min, steps, device)
        return (sigmas,)


class AdeptSchedulerAOS_E:
    """AOS-ε (Anime-Optimized Schedule for epsilon-prediction models)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/adept/schedulers"
    
    def get_sigmas(self, model, steps):
        sigma_min = model.get_model_object("model_sampling").sigma_min
        sigma_max = model.get_model_object("model_sampling").sigma_max
        device = model.load_device
        
        sigmas = create_aos_e_sigmas(sigma_max, sigma_min, steps, device)
        return (sigmas,)


class AdeptSchedulerAkashicAOS:
    """AkashicAOS v2: Detail-Progressive Schedule for EQ-VAE SDXL models."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/adept/schedulers"
    
    def get_sigmas(self, model, steps):
        sigma_min = model.get_model_object("model_sampling").sigma_min
        sigma_max = model.get_model_object("model_sampling").sigma_max
        device = model.load_device
        
        sigmas = create_aos_akashic_sigmas(sigma_max, sigma_min, steps, device)
        return (sigmas,)


class AdeptSchedulerEntropic:
    """Entropic scheduler with configurable power parameter."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "power": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/adept/schedulers"
    
    def get_sigmas(self, model, steps, power):
        sigma_min = model.get_model_object("model_sampling").sigma_min
        sigma_max = model.get_model_object("model_sampling").sigma_max
        device = model.load_device
        
        sigmas = create_entropic_sigmas(sigma_max, sigma_min, steps, power, device)
        return (sigmas,)


class AdeptSchedulerJYS:
    """JYS (Jump Your Steps) dynamic scheduler."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/adept/schedulers"
    
    def get_sigmas(self, model, steps):
        sigma_min = model.get_model_object("model_sampling").sigma_min
        sigma_max = model.get_model_object("model_sampling").sigma_max
        device = model.load_device
        
        sigmas = create_jys_sigmas(sigma_max, sigma_min, steps, device)
        return (sigmas,)


class AdeptSchedulerAYS:
    """AYS (Align Your Steps) scheduler based on NVIDIA's CVPR 2024 paper."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/adept/schedulers"
    
    def get_sigmas(self, model, steps):
        sigma_min = model.get_model_object("model_sampling").sigma_min
        sigma_max = model.get_model_object("model_sampling").sigma_max
        device = model.load_device
        
        sigmas = create_ays_sdxl_sigmas(sigma_max, sigma_min, steps, device)
        return (sigmas,)


class AdeptSchedulerStochastic:
    """Stochastic scheduler with controlled randomness."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "noise_type": (["brownian", "uniform", "normal"],),
                "noise_scale": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "base_schedule": (["karras", "uniform", "cosine"],),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/adept/schedulers"
    
    def get_sigmas(self, model, steps, noise_type, noise_scale, base_schedule):
        sigma_min = model.get_model_object("model_sampling").sigma_min
        sigma_max = model.get_model_object("model_sampling").sigma_max
        device = model.load_device
        
        sigmas = create_stochastic_sigmas(sigma_max, sigma_min, steps, device, noise_type, noise_scale, base_schedule)
        return (sigmas,)


class AdeptSchedulerAdvanced:
    """Advanced scheduler with dropdown selection for all scheduler types."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "scheduler": (SCHEDULER_NAMES,),
            },
            "optional": {
                "entropic_power": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/adept/schedulers"
    
    def get_sigmas(self, model, steps, scheduler, entropic_power=6.0):
        sigma_min = model.get_model_object("model_sampling").sigma_min
        sigma_max = model.get_model_object("model_sampling").sigma_max
        device = model.load_device
        
        scheduler_map = {
            "AOS-V": lambda: create_aos_v_sigmas(sigma_max, sigma_min, steps, device),
            "AOS-ε": lambda: create_aos_e_sigmas(sigma_max, sigma_min, steps, device),
            "AkashicAOS": lambda: create_aos_akashic_sigmas(sigma_max, sigma_min, steps, device),
            "Entropic": lambda: create_entropic_sigmas(sigma_max, sigma_min, steps, entropic_power, device),
            "SNR-Optimized": lambda: create_snr_optimized_sigmas(sigma_max, sigma_min, steps, device),
            "Constant-Rate": lambda: create_constant_rate_sigmas(sigma_max, sigma_min, steps, device),
            "Adaptive-Optimized": lambda: create_adaptive_optimized_sigmas(sigma_max, sigma_min, steps, device),
            "Cosine-Annealed": lambda: create_cosine_sigmas(sigma_max, sigma_min, steps, device),
            "LogSNR-Uniform": lambda: create_logsnr_uniform_sigmas(sigma_max, sigma_min, steps, device),
            "Tanh Mid-Boost": lambda: create_tanh_midboost_sigmas(sigma_max, sigma_min, steps, device),
            "Exponential Tail": lambda: create_exponential_tail_sigmas(sigma_max, sigma_min, steps, device),
            "Jittered-Karras": lambda: create_jittered_karras_sigmas(sigma_max, sigma_min, steps, device),
            "Stochastic": lambda: create_stochastic_sigmas(sigma_max, sigma_min, steps, device),
            "JYS (Dynamic)": lambda: create_jys_sigmas(sigma_max, sigma_min, steps, device),
            "Hybrid JYS-Karras": lambda: create_hybrid_jys_karras_sigmas(sigma_max, sigma_min, steps, device),
            "AYS-SDXL": lambda: create_ays_sdxl_sigmas(sigma_max, sigma_min, steps, device),
        }
        
        sigmas = scheduler_map[scheduler]()
        return (sigmas,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AdeptSchedulerAOS_V": AdeptSchedulerAOS_V,
    "AdeptSchedulerAOS_E": AdeptSchedulerAOS_E,
    "AdeptSchedulerAkashicAOS": AdeptSchedulerAkashicAOS,
    "AdeptSchedulerEntropic": AdeptSchedulerEntropic,
    "AdeptSchedulerJYS": AdeptSchedulerJYS,
    "AdeptSchedulerAYS": AdeptSchedulerAYS,
    "AdeptSchedulerStochastic": AdeptSchedulerStochastic,
    "AdeptSchedulerAdvanced": AdeptSchedulerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdeptSchedulerAOS_V": "Adept Scheduler (AOS-V)",
    "AdeptSchedulerAOS_E": "Adept Scheduler (AOS-ε)",
    "AdeptSchedulerAkashicAOS": "Adept Scheduler (AkashicAOS)",
    "AdeptSchedulerEntropic": "Adept Scheduler (Entropic)",
    "AdeptSchedulerJYS": "Adept Scheduler (JYS)",
    "AdeptSchedulerAYS": "Adept Scheduler (AYS-SDXL)",
    "AdeptSchedulerStochastic": "Adept Scheduler (Stochastic)",
    "AdeptSchedulerAdvanced": "Adept Scheduler (Advanced)",
}
