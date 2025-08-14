"""
🚀 Universal NN Latent Upscaler for ComfyUI

A neural network-based latent upscaler that provides better quality than
bilinear interpolation for SD1.5, SDXL, Flux, and Wan2.2 models.

Built upon the excellent foundation of Ttl's ComfyUi_NNLatentUpscale:
https://github.com/Ttl/ComfyUi_NNLatentUpscale

Features:
- Universal model support (SD1.5, SDXL, Flux, Wan2.2)
- Neural network upscaling (better than bilinear)
- Configurable scale factors (1.0x - 2.0x)
- Automatic model detection and loading
- Memory efficient implementation

Author: denrakeiw (based on Ttl's work)
License: MIT
Repository: https://github.com/yourusername/wan_nn_latent
"""

from .nn_upscale import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export the mappings for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Package metadata
__version__ = "1.0.0"
__author__ = "denrakeiw"
__license__ = "MIT"
__description__ = "Universal Neural Network Latent Upscaler for ComfyUI"
__url__ = "https://github.com/yourusername/wan_nn_latent"

# Startup messages
print("🚀 " + "="*60)
print(f"   {__description__}")
print(f"   Version: {__version__} | Author: {__author__} | License: {__license__}")
print("   Based on Ttl's ComfyUi_NNLatentUpscale")
print("   " + "="*60)
print("✨ Supported Models: SD1.5, SDXL, Flux, Wan2.2")
print("📍 Node Location: Add Node -> latent -> Universal NN Latent Upscale")
print("⚠️  Status: WIP - Improved models in development!")
print("🙏 Original work: https://github.com/Ttl/ComfyUi_NNLatentUpscale")
print("🔗 This project: " + __url__)
print("🚀 " + "="*60)
