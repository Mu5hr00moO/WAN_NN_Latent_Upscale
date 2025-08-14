# Development Scripts

This directory contains development and training scripts for the Universal NN Latent Upscaler project.

⚠️ **Note**: These scripts are for development purposes and are not needed for normal usage of the ComfyUI node.

## 📁 File Overview

### Dataset Creation
- `create_real_dataset.py` - Create datasets from real photos
- `create_real_photo_dataset.py` - Process real photos for training
- `create_real_wan22_dataset.py` - Create Wan2.2-specific datasets
- `create_synthetic_wan22_dataset.py` - Generate synthetic training data
- `download_div2k_and_create_pairs.py` - Download and process DIV2K dataset

### Model Training
- `slow_long_training_20k.py` - Main training script (20k steps, slow LR)
- `final_mario_training.py` - Experimental training approach
- `retrain_with_real_data.py` - Retrain with real photo data

### Analysis & Testing
- `analyze_model.py` - Analyze model performance
- `compare_models.py` - Compare different model versions
- `check_dataset_vae.py` - Validate dataset quality
- `check_real_datasets.py` - Check available real datasets
- `test_mario_integration.py` - Integration testing
- `simple_mario_test.py` - Simple model testing
- `simple_vae_check.py` - VAE validation

## 🚀 Quick Start (Development)

### 1. Setup Environment
```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install matplotlib tqdm pillow

# Ensure ComfyUI is available
export PYTHONPATH="${PYTHONPATH}:../../.."
```

### 2. Create Dataset
```bash
# Create real photo dataset
python create_real_photo_dataset.py

# Or create synthetic dataset
python create_synthetic_wan22_dataset.py
```

### 3. Train Model
```bash
# Train with 20k steps (recommended)
python slow_long_training_20k.py

# Monitor training progress
# Check training_curves_20k_slow.png
```

### 4. Test Model
```bash
# Check model performance
python check_real_datasets.py

# Analyze results
python analyze_model.py
```

## 📊 Training Configuration

### Current Best Settings (slow_long_training_20k.py):
- **Steps**: 20,000
- **Learning Rate**: 5e-5 (with warmup)
- **Batch Size**: 6
- **Loss**: MSE + Gradient + Frequency domain
- **Dataset**: 2,852 real photos
- **Architecture**: 16-channel, 256 hidden dimensions

### Expected Results:
- **Training Time**: ~3-4 hours (RTX 3080)
- **Final Val Loss**: ~0.048 (vs 0.126 initial)
- **SSIM Improvement**: 20%+ over bilinear

## 🔧 Development Notes

### Model Architecture
- **Input**: 16-channel latents (32x32 for Wan2.2)
- **Output**: 16-channel latents (64x64 for Wan2.2)
- **Hidden**: 256 dimensions
- **Layers**: Encoder (3 CNN) + Upsampler + Skip connections

### Dataset Requirements
- **Format**: PyTorch tensors (.pt files)
- **Structure**: `{'hr_latents': tensor, 'lr_latents': tensor, 'metadata': dict}`
- **Size**: HR: [N, 16, 64, 64], LR: [N, 16, 32, 32]
- **Range**: Typically [-2.0, 2.0] for Wan2.2 latents

### Performance Targets
- **SSIM**: > 0.32 (vs ~0.27 bilinear)
- **PSNR**: > 9.8 dB (vs ~9.7 dB bilinear)
- **MSE**: < 0.105 (vs ~0.105 bilinear)
- **Speed**: < 100ms inference (RTX 3080)

## 🐛 Common Issues

### Training Issues
- **CUDA OOM**: Reduce batch size to 4 or use CPU
- **Slow convergence**: Check learning rate and dataset quality
- **NaN losses**: Reduce learning rate or check data normalization

### Dataset Issues
- **Wrong shapes**: Ensure HR/LR latents have correct dimensions
- **Bad quality**: Check VAE encoding and source images
- **Memory issues**: Process datasets in smaller batches

### Model Issues
- **Poor quality**: Try different loss functions or architectures
- **Overfitting**: Add regularization or more training data
- **Slow inference**: Optimize model size or use quantization

## 📈 Monitoring Training

### Key Metrics to Watch:
1. **Validation Loss**: Should decrease steadily
2. **Training/Val Gap**: Should remain small (< 2x)
3. **Learning Rate**: Should decay smoothly
4. **Gradient Norms**: Should be stable

### Visualization:
- Training curves saved as PNG files
- Use TensorBoard for detailed monitoring (if available)
- Check sample outputs during training

## 🔄 Contributing

When adding new development scripts:
1. Follow the naming convention
2. Add proper documentation
3. Include example usage
4. Update this README
5. Test thoroughly before committing

## 📞 Support

For development-related questions:
- Check the main project README
- Open an issue on GitHub
- Contact the development team

---

**Happy developing! 🚀**
