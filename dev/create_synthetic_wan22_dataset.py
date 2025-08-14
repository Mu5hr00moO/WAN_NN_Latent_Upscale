import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import matplotlib.pyplot as plt

def create_synthetic_wan22_latents():
    """
    Create synthetic Wan2.2 latents for training.
    This creates realistic 16-channel latents that mimic real Wan2.2 VAE output.
    """
    
    print("🧠 Creating Synthetic Wan2.2 Latent Dataset")
    print("=" * 50)
    
    # Parameters
    num_samples = 500  # Same as DIV2K validation set
    channels = 16      # Wan2.2 uses 16 channels
    hr_size = 64       # High resolution latent size (64x64)
    lr_size = 32       # Low resolution latent size (32x32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    print(f"📊 Creating {num_samples} synthetic latent pairs")
    print(f"📐 HR: {channels}x{hr_size}x{hr_size}, LR: {channels}x{lr_size}x{lr_size}")
    
    hr_latents = []
    lr_latents = []
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    for i in tqdm(range(num_samples), desc="Generating latents"):
        # Create realistic HR latent
        # Wan2.2 latents have specific characteristics:
        # - Multi-scale features (some channels for fine details, others for coarse)
        # - Spatial correlations
        # - Realistic value ranges
        
        # Base pattern with spatial structure
        base_pattern = torch.randn(1, 4, hr_size, hr_size) * 0.5
        
        # Fine detail channels (high frequency)
        fine_details = torch.randn(1, 6, hr_size, hr_size) * 0.3
        
        # Coarse feature channels (low frequency)
        coarse_features = torch.randn(1, 6, hr_size//2, hr_size//2) * 0.8
        coarse_features = F.interpolate(coarse_features, size=(hr_size, hr_size), mode='bilinear', align_corners=False)
        
        # Combine all channels
        hr_latent = torch.cat([base_pattern, fine_details, coarse_features], dim=1)
        
        # Add some spatial correlation
        kernel = torch.ones(1, 1, 3, 3) / 9.0  # Simple blur kernel
        for c in range(hr_latent.shape[1]):
            if c % 3 == 0:  # Apply blur to every 3rd channel
                hr_latent[:, c:c+1] = F.conv2d(hr_latent[:, c:c+1], kernel, padding=1)
        
        # Create corresponding LR latent by downsampling + adding noise
        # This simulates what happens when we encode a lower resolution image
        lr_latent = F.interpolate(hr_latent, size=(lr_size, lr_size), mode='bilinear', align_corners=False)
        
        # Add some noise to simulate encoding differences
        noise = torch.randn_like(lr_latent) * 0.1
        lr_latent = lr_latent + noise
        
        # Normalize to realistic ranges (Wan2.2 VAE typically outputs values in [-2, 2])
        hr_latent = torch.clamp(hr_latent, -2.0, 2.0)
        lr_latent = torch.clamp(lr_latent, -2.0, 2.0)
        
        hr_latents.append(hr_latent)
        lr_latents.append(lr_latent)
    
    # Stack all latents
    hr_latents = torch.cat(hr_latents, dim=0)
    lr_latents = torch.cat(lr_latents, dim=0)
    
    print(f"✅ Generated latent dataset:")
    print(f"  HR Latents: {hr_latents.shape}")
    print(f"  LR Latents: {lr_latents.shape}")
    print(f"  HR range: [{hr_latents.min():.3f}, {hr_latents.max():.3f}]")
    print(f"  LR range: [{lr_latents.min():.3f}, {lr_latents.max():.3f}]")
    
    # Create output directory
    output_dir = Path("datasets/real_latent")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset = {
        'hr_latents': hr_latents,
        'lr_latents': lr_latents,
        'metadata': {
            'type': 'synthetic_wan22_latents',
            'description': 'High-quality synthetic Wan2.2 latents for upscaler training',
            'source': 'Synthetic generation with realistic characteristics',
            'vae_type': 'Wan2.2 (16-channel)',
            'hr_size': f'{hr_size}x{hr_size}',
            'lr_size': f'{lr_size}x{lr_size}',
            'samples': len(hr_latents),
            'channels': channels,
            'value_range': '[-2.0, 2.0]',
            'created_for': '20k_step_slow_training',
            'seed': 42
        }
    }
    
    output_path = output_dir / "synthetic_wan22_20k_dataset.pt"
    torch.save(dataset, output_path)
    
    print(f"💾 Saved dataset: {output_path}")
    
    # Create visualization
    visualize_latents(hr_latents, lr_latents)
    
    print(f"🎉 Synthetic Wan2.2 dataset ready for 20K step training!")
    return True

def visualize_latents(hr_latents, lr_latents):
    """Create visualization of the generated latents."""
    
    print("📊 Creating visualization...")
    
    # Select a few samples for visualization
    sample_indices = [0, 1, 2, 3]
    
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle('Synthetic Wan2.2 Latents Visualization', fontsize=16)
    
    for i, idx in enumerate(sample_indices):
        hr_sample = hr_latents[idx]
        lr_sample = lr_latents[idx]
        
        # Show first 4 channels of HR
        for c in range(4):
            ax = axes[i, c]
            im = ax.imshow(hr_sample[c].numpy(), cmap='viridis', vmin=-2, vmax=2)
            ax.set_title(f'HR Ch{c}' if i == 0 else '')
            ax.axis('off')
        
        # Show first 4 channels of LR
        for c in range(4):
            ax = axes[i, c + 4]
            im = ax.imshow(lr_sample[c].numpy(), cmap='viridis', vmin=-2, vmax=2)
            ax.set_title(f'LR Ch{c}' if i == 0 else '')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('synthetic_wan22_latents_preview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # HR latent statistics
    hr_flat = hr_latents.flatten().numpy()
    axes[0, 0].hist(hr_flat, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('HR Latent Value Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # LR latent statistics
    lr_flat = lr_latents.flatten().numpy()
    axes[0, 1].hist(lr_flat, bins=50, alpha=0.7, color='red')
    axes[0, 1].set_title('LR Latent Value Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Channel-wise mean
    hr_channel_means = hr_latents.mean(dim=(0, 2, 3)).numpy()
    lr_channel_means = lr_latents.mean(dim=(0, 2, 3)).numpy()
    
    axes[1, 0].bar(range(16), hr_channel_means, alpha=0.7, color='blue', label='HR')
    axes[1, 0].bar(range(16), lr_channel_means, alpha=0.7, color='red', label='LR')
    axes[1, 0].set_title('Channel-wise Mean Values')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Mean Value')
    axes[1, 0].legend()
    
    # Channel-wise std
    hr_channel_stds = hr_latents.std(dim=(0, 2, 3)).numpy()
    lr_channel_stds = lr_latents.std(dim=(0, 2, 3)).numpy()
    
    axes[1, 1].bar(range(16), hr_channel_stds, alpha=0.7, color='blue', label='HR')
    axes[1, 1].bar(range(16), lr_channel_stds, alpha=0.7, color='red', label='LR')
    axes[1, 1].set_title('Channel-wise Standard Deviation')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Std Dev')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('synthetic_wan22_latents_stats.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved:")
    print("  - synthetic_wan22_latents_preview.png")
    print("  - synthetic_wan22_latents_stats.png")

def main():
    """Main function."""
    create_synthetic_wan22_latents()

if __name__ == "__main__":
    main()
