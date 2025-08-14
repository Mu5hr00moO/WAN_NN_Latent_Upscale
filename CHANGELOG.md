# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In Development
- 🔄 Training improved Wan2.2 model with 2,852 real photo samples
- 🔄 Advanced loss functions (MSE + Gradient + Frequency domain)
- 🔄 20,000 training steps with slow learning rate for stability
- 🔄 Performance benchmarking against other upscaling methods

### Planned
- 📋 Support for custom model training interface
- 📋 Additional model architectures (SDXL-Turbo, etc.)
- 📋 Batch processing optimization
- 📋 Advanced configuration options
- 📋 Model quality metrics in UI

## [1.0.0] - 2025-01-14

### Added
- ✨ Initial release of Universal NN Latent Upscaler
- 🎯 Support for SD1.5, SDXL, Flux, and Wan2.2 models
- 🧠 Neural network-based upscaling (better than bilinear)
- ⚙️ Configurable scale factors (1.0x - 2.0x)
- 🔄 Automatic model detection and loading
- 💾 Memory efficient model management
- 📊 Custom trained Wan2.2 model included

### Technical Details
- Neural network architecture with encoder-upsampler design
- Skip connections for detail preservation
- Adaptive scaling for non-2x factors
- Automatic VRAM management
- ComfyUI integration with proper node registration

### Performance (v1.0)
- **Wan2.2 SSIM**: 0.3247 vs 0.2690 bilinear (+20.7% improvement)
- **Wan2.2 PSNR**: 9.84 dB vs 9.77 dB bilinear (+0.7 dB improvement)
- **Wan2.2 MSE**: 0.1038 vs 0.1054 bilinear (-1.5% improvement)

### Known Issues
- Wan2.2 model quality could be improved (training v2.0)
- Limited to 2.0x maximum upscale factor
- No custom model training interface yet

## [0.9.0] - 2025-01-13

### Added
- 🧪 Beta version with basic functionality
- 🔬 Initial model training and testing
- 📝 Basic documentation

### Changed
- 🔧 Refined neural network architecture
- 📊 Improved training methodology

## [0.1.0] - 2025-01-10

### Added
- 🎬 Initial project setup
- 🏗️ Basic neural network architecture
- 🧪 Proof of concept implementation

---

## Legend

- ✨ New features
- 🔧 Improvements
- 🐛 Bug fixes
- 📝 Documentation
- 🔄 Work in progress
- 📋 Planned features
- ⚠️ Breaking changes
- 🧪 Experimental features
