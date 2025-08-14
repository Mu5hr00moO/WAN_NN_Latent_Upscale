# Contributing to Universal NN Latent Upscaler

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

**Note**: This project builds upon [Ttl's ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale). Please also consider contributing to the original project that made this work possible.

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/yourusername/wan_nn_latent.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes**
5. **Test thoroughly**
6. **Submit a pull request**

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Use the [issue template](https://github.com/yourusername/wan_nn_latent/issues/new)
- Include ComfyUI version, Python version, and GPU info
- Provide steps to reproduce the issue
- Include error messages and logs

### ✨ Feature Requests
- Check existing issues first
- Describe the use case and expected behavior
- Consider implementation complexity
- Discuss with maintainers before large changes

### 🔧 Code Contributions
- Follow the coding standards below
- Add tests for new functionality
- Update documentation as needed
- Ensure backward compatibility

### 📝 Documentation
- Fix typos and improve clarity
- Add examples and use cases
- Update README for new features
- Translate documentation (if applicable)

## 🛠️ Development Setup

### Prerequisites
```bash
# Python 3.8+
python --version

# ComfyUI (latest)
git clone https://github.com/comfyanonymous/ComfyUI.git

# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Local Development
```bash
# Clone your fork
git clone https://github.com/yourusername/wan_nn_latent.git
cd wan_nn_latent

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Start ComfyUI for testing
cd ../ComfyUI
python main.py
```

## 📋 Coding Standards

### Python Style
- Follow [PEP 8](https://pep8.org/)
- Use type hints where possible
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Code Structure
```python
# Good
def upscale_latent(latent: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Upscale latent tensor using neural network.
    
    Args:
        latent: Input latent tensor [B, C, H, W]
        scale_factor: Upscaling factor (1.0 - 2.0)
        
    Returns:
        Upscaled latent tensor
    """
    # Implementation here
    pass

# Bad
def upscale(x, s):
    # No docstring, unclear parameters
    pass
```

### Documentation
- Use Google-style docstrings
- Document all public functions and classes
- Include examples for complex functionality
- Keep README.md updated

## 🧪 Testing

### Running Tests
```bash
# All tests
python -m pytest

# Specific test file
python -m pytest tests/test_models.py

# With coverage
python -m pytest --cov=wan_nn_latent
```

### Writing Tests
```python
import pytest
import torch
from wan_nn_latent.latent_resizer import WanLatentResizer

def test_wan_resizer_output_shape():
    """Test that WAN resizer produces correct output shape."""
    model = WanLatentResizer(16, 16, 256)
    input_tensor = torch.randn(1, 16, 32, 32)
    output = model(input_tensor)
    
    assert output.shape == (1, 16, 64, 64)
    assert output.dtype == input_tensor.dtype
```

## 🏗️ Model Development

### Training New Models
```bash
# Create dataset
python create_real_dataset.py --model wan22 --samples 1000

# Train model
python train_model.py --config configs/wan22_config.yaml

# Evaluate model
python evaluate_model.py --model models/new_model.pt
```

### Model Requirements
- Input/output tensor compatibility
- Memory efficiency (< 100MB model size preferred)
- Performance improvement over bilinear interpolation
- Stable training convergence

### Adding New Model Types
1. Add configuration to `MODEL_CONFIGS` in `latent_resizer.py`
2. Create model class if needed
3. Add to `weight_paths` in `nn_upscale.py`
4. Update dropdown options
5. Add tests for new model type

## 📊 Performance Guidelines

### Benchmarking
```python
# Use the provided benchmark script
python benchmark_model.py --model wan22 --samples 100

# Expected metrics:
# - SSIM > 0.30 (vs bilinear ~0.27)
# - PSNR > 9.5 dB (vs bilinear ~9.5 dB)
# - Inference time < 100ms (on RTX 3080)
```

### Memory Usage
- Models should use < 2GB VRAM for inference
- Implement proper cleanup in `__del__` methods
- Use `torch.no_grad()` for inference
- Clear cache when switching models

## 🔄 Pull Request Process

### Before Submitting
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Existing tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

### Review Process
1. Automated tests run
2. Code review by maintainers
3. Performance testing (if applicable)
4. Documentation review
5. Approval and merge

## 🏷️ Release Process

### Version Numbering
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist
- [ ] Update version in `__init__.py`
- [ ] Update CHANGELOG.md
- [ ] Create release notes
- [ ] Tag release in git
- [ ] Test installation from release

## 💬 Communication

### Getting Help
- 💬 [GitHub Discussions](https://github.com/yourusername/wan_nn_latent/discussions)
- 🐛 [Issues](https://github.com/yourusername/wan_nn_latent/issues)
- 📧 Email: your-email@example.com

### Community Guidelines
- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and experiences
- Give constructive feedback

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Universal NN Latent Upscaler! 🚀
