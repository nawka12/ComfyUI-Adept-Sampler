"""
ComfyUI-Adept-Sampler

Custom samplers and schedulers for ComfyUI.
Ported from Stable Diffusion WebUI reForge extension.

Samplers:
- Adept Solver: Multistep predictor-corrector (DPM-Solver++/UniPC/DEIS/DC-Solver)
- Adept Ancestral Solver: Phase-aware ancestral sampling
- AkashicSolver v2: SA-Solver optimized for EQ-VAE models

Schedulers:
- AOS-V: Anime-Optimized Schedule for v-prediction
- AOS-ε: Anime-Optimized Schedule for epsilon-prediction
- AkashicAOS: Detail-progressive for EQ-VAE
- Entropic, JYS, AYS-SDXL, and 10+ more
"""

from .nodes_schedulers import (
    NODE_CLASS_MAPPINGS as SCHEDULER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SCHEDULER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_samplers import (
    NODE_CLASS_MAPPINGS as SAMPLER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SAMPLER_NODE_DISPLAY_NAME_MAPPINGS,
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **SCHEDULER_NODE_CLASS_MAPPINGS,
    **SAMPLER_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **SCHEDULER_NODE_DISPLAY_NAME_MAPPINGS,
    **SAMPLER_NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = None

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ ComfyUI-Adept-Sampler loaded successfully!")
