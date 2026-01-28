"""
ComfyUI Sampler Nodes for Adept Sampler.
"""

import torch
from functools import partial
from .adept_samplers import (
    sample_adept_solver,
    sample_adept_ancestral_solver,
    sample_akashic_solver,
)


class AdeptSolverSampler:
    """
    Adept Solver: Multistep predictor-corrector sampler.
    
    Combines techniques from DPM-Solver++, UniPC, DEIS, and DC-Solver.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "order": ("INT", {"default": 2, "min": 1, "max": 3}),
                "use_corrector": ("BOOLEAN", {"default": True}),
                "use_detail_enhancement": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "detail_strength": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_radius": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/adept/samplers"
    
    def get_sampler(self, order, use_corrector, use_detail_enhancement, 
                    detail_strength=0.05, detail_radius=0.5):
        settings = {
            'detail_enhancement_strength': detail_strength,
            'detail_separation_radius': detail_radius,
        }
        
        sampler = KSAMPLER(
            sample_adept_solver,
            extra_options={
                'order': order,
                'use_corrector': use_corrector,
                'use_detail_enhancement': use_detail_enhancement,
                'settings': settings,
            }
        )
        return (sampler,)


class AdeptAncestralSampler:
    """
    Adept Ancestral Solver: Phase-aware ancestral sampling.
    
    Features adaptive eta, phase-aware noise injection, and enhanced derivatives.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "adaptive_eta": ("BOOLEAN", {"default": False}),
                "phase_noise": ("BOOLEAN", {"default": False}),
                "enhanced_derivative": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "phase_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "use_detail_enhancement": ("BOOLEAN", {"default": False}),
                "detail_strength": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_radius": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/adept/samplers"
    
    def get_sampler(self, eta, s_noise, adaptive_eta, phase_noise, enhanced_derivative,
                    phase_strength=0.5, use_detail_enhancement=False, 
                    detail_strength=0.05, detail_radius=0.5):
        settings = {
            'detail_enhancement_strength': detail_strength,
            'detail_separation_radius': detail_radius,
        }
        
        sampler = KSAMPLER(
            sample_adept_ancestral_solver,
            extra_options={
                'eta': eta,
                's_noise': s_noise,
                'adaptive_eta': adaptive_eta,
                'phase_noise': phase_noise,
                'phase_strength': phase_strength,
                'enhanced_derivative': enhanced_derivative,
                'use_detail_enhancement': use_detail_enhancement,
                'settings': settings,
            }
        )
        return (sampler,)


class AkashicSolverSampler:
    """
    AkashicSolver v2 [EXPERIMENTAL]: SA-Solver optimized for EQ-VAE models.

    Combines SA-Solver multi-step integration with phase-aware adaptation and SMEA coherency.

    EQ-VAE Mode:
    - Off: Standard mode, use external rescaleCFG (0.7) for EQ-VAE models
    - Balanced: Optimized for EQ-VAE's cleaner latent space, maintains sharpness
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tau": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "order": ("INT", {"default": 2, "min": 1, "max": 3}),
                "adaptive_eta": ("BOOLEAN", {"default": True}),
                "eqvae_mode": (["Off", "Balanced"], {"default": "Off"}),
            },
            "optional": {
                "phase_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "smea_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "ndb_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_detail_enhancement": ("BOOLEAN", {"default": False}),
                "detail_strength": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_radius": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/adept/samplers"

    def get_sampler(self, tau, eta, s_noise, order, adaptive_eta, eqvae_mode,
                    phase_strength=0.5, smea_strength=0.0, ndb_strength=0.0,
                    use_detail_enhancement=False, detail_strength=0.05, detail_radius=0.5):
        settings = {
            'detail_enhancement_strength': detail_strength,
            'detail_separation_radius': detail_radius,
        }

        sampler = KSAMPLER(
            sample_akashic_solver,
            extra_options={
                'tau': tau,
                'eta': eta,
                's_noise': s_noise,
                'order': order,
                'adaptive_eta': adaptive_eta,
                'phase_strength': phase_strength,
                'smea_strength': smea_strength,
                'ndb_strength': ndb_strength,
                'use_detail_enhancement': use_detail_enhancement,
                'settings': settings,
                'eqvae_mode': eqvae_mode,
            }
        )
        return (sampler,)


class KSAMPLER:
    """Wrapper class to create a SAMPLER object compatible with ComfyUI."""
    
    def __init__(self, sampler_function, extra_options=None):
        self.sampler_function = sampler_function
        self.extra_options = extra_options or {}
    
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        """Execute the sampling."""
        # Start with noise or latent image
        if latent_image is not None and noise is not None:
            x = latent_image + noise * sigmas[0]
        elif noise is not None:
            x = noise * sigmas[0]
        else:
            x = latent_image
        
        # Extract our options
        opts = self.extra_options.copy()
        
        # Call the sampler function with our parameters
        return self.sampler_function(
            model=model_wrap,
            x=x,
            sigmas=sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable_pbar,
            **opts
        )


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AdeptSolverSampler": AdeptSolverSampler,
    "AdeptAncestralSampler": AdeptAncestralSampler,
    "AkashicSolverSampler": AkashicSolverSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdeptSolverSampler": "Adept Solver Sampler",
    "AdeptAncestralSampler": "Adept Ancestral Sampler",
    "AkashicSolverSampler": "AkashicSolver v2 [EXPERIMENTAL]",
}
