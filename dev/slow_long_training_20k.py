import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from latent_resizer import WanLatentResizer

class RealLatentDataset(Dataset):
    """Dataset for real latent upscaling training with DIV2K data."""

    def __init__(self, dataset_path=None):
        # Use the BEST real photo dataset (2852 real photos, 16-channel)
        possible_paths = [
            "datasets/real_latent/real_photo_dataset.pt",  # BEST: 2852 real photos
            "datasets/real_latent/real_wan22_coco_dataset.pt",
            "datasets/real_latent/improved_synthetic_dataset.pt",
        ]

        if dataset_path:
            possible_paths.insert(0, dataset_path)

        dataset_loaded = False
        for path in possible_paths:
            try:
                print(f"🔍 Trying to load: {path}")
                self.data = torch.load(path, map_location='cpu')
                self.hr_latents = self.data['hr_latents']
                self.lr_latents = self.data['lr_latents']
                dataset_type = self.data['metadata'].get('type', 'unknown')

                print(f"✅ Loaded {dataset_type} dataset from {path}:")
                print(f"  HR Latents: {self.hr_latents.shape}")
                print(f"  LR Latents: {self.lr_latents.shape}")
                print(f"  Samples: {len(self.hr_latents)}")
                
                # Show some metadata if available
                if 'metadata' in self.data:
                    metadata = self.data['metadata']
                    print(f"  Dataset metadata:")
                    for key, value in metadata.items():
                        print(f"    {key}: {value}")
                
                dataset_loaded = True
                self.dataset_path = path
                break
            except FileNotFoundError:
                print(f"❌ Not found: {path}")
                continue
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
                continue

        if not dataset_loaded:
            raise FileNotFoundError("❌ No dataset found! Please create a dataset first")
        
    def __len__(self):
        return len(self.hr_latents)
    
    def __getitem__(self, idx):
        hr_latent = self.hr_latents[idx]
        lr_latent = self.lr_latents[idx]

        # For training, we use LR latent as input and HR latent as target
        # The model should learn to upscale from 32x32 to 64x64
        return lr_latent, hr_latent

class AdvancedPerceptualLoss(nn.Module):
    """
    Advanced perceptual loss for better detail preservation.
    Combines MSE with gradient loss and frequency domain loss.
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        # MSE Loss
        mse = self.mse_loss(pred, target)
        
        # Gradient Loss (for edge preservation)
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        grad_loss = self.mse_loss(pred_grad_x, target_grad_x) + self.mse_loss(pred_grad_y, target_grad_y)
        
        # Frequency domain loss (helps with texture preservation)
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        freq_loss = self.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        # Combined loss
        total_loss = mse + 0.1 * grad_loss + 0.05 * freq_loss
        
        return total_loss, mse, grad_loss, freq_loss

def train_slow_long_20k():
    """Train the model with slow learning rate and 20k steps using real DIV2K data."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    print(f"🔥 SLOW LONG TRAINING - 20K STEPS WITH REAL DIV2K DATA")
    print("=" * 60)
    
    # Load dataset
    try:
        dataset = RealLatentDataset()
        print(f"📊 Using dataset: {dataset.dataset_path}")
    except FileNotFoundError as e:
        print(e)
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Data loaders
    batch_size = 6  # Slightly smaller batch for more stable gradients
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = WanLatentResizer(in_channels=16, out_channels=16, hidden_dim=256)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Training parameters
    total_steps = 20000
    learning_rate = 5e-5  # Slower learning rate for stable training
    warmup_steps = 1000   # Warmup for first 1000 steps
    
    # Loss and optimizer
    criterion = AdvancedPerceptualLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    step_count = 0
    
    print(f"\n🎯 Training Configuration:")
    print(f"  Total Steps: {total_steps:,}")
    print(f"  Learning Rate: {learning_rate} (with warmup)")
    print(f"  Warmup Steps: {warmup_steps:,}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Dataset Size: {len(dataset)} samples")
    print(f"  Train/Val Split: {len(train_dataset)}/{len(val_dataset)}")
    print(f"  Steps per epoch: ~{len(train_loader)}")
    print(f"  Estimated epochs: ~{total_steps // len(train_loader)}")
    print()
    
    print(f"🚀 Starting SLOW LONG training for {total_steps:,} steps...")
    print("🎯 Goal: Ultra-high quality Wan2.2 latent upscaling with real photo data")
    print()
    
    # Training loop
    model.train()
    train_loss_accum = 0.0
    train_mse_accum = 0.0
    train_grad_accum = 0.0
    train_freq_accum = 0.0
    log_interval = 100  # Log every 100 steps
    val_interval = 1000  # Validate every 1000 steps
    save_interval = 2000  # Save checkpoint every 2000 steps
    
    progress_bar = tqdm(total=total_steps, desc="Training Progress")
    
    while step_count < total_steps:
        for batch_idx, (lr_input, hr_target) in enumerate(train_loader):
            if step_count >= total_steps:
                break
                
            lr_input = lr_input.to(device, non_blocking=True)
            hr_target = hr_target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(lr_input)
            
            # Loss
            total_loss, mse_loss, grad_loss, freq_loss = criterion(output, hr_target)
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Accumulate losses
            train_loss_accum += total_loss.item()
            train_mse_accum += mse_loss.item()
            train_grad_accum += grad_loss.item()
            train_freq_accum += freq_loss.item()
            
            step_count += 1
            progress_bar.update(1)
            
            # Log progress
            if step_count % log_interval == 0:
                avg_train_loss = train_loss_accum / log_interval
                avg_train_mse = train_mse_accum / log_interval
                avg_train_grad = train_grad_accum / log_interval
                avg_train_freq = train_freq_accum / log_interval
                current_lr = scheduler.get_last_lr()[0]
                
                progress_bar.set_postfix({
                    'Loss': f'{avg_train_loss:.6f}',
                    'MSE': f'{avg_train_mse:.6f}',
                    'LR': f'{current_lr:.2e}'
                })
                
                train_losses.append(avg_train_loss)
                
                # Reset accumulators
                train_loss_accum = 0.0
                train_mse_accum = 0.0
                train_grad_accum = 0.0
                train_freq_accum = 0.0
            
            # Validation
            if step_count % val_interval == 0:
                model.eval()
                val_loss = 0.0
                val_mse = 0.0
                val_grad = 0.0
                val_freq = 0.0
                
                with torch.no_grad():
                    for val_lr_input, val_hr_target in val_loader:
                        val_lr_input = val_lr_input.to(device, non_blocking=True)
                        val_hr_target = val_hr_target.to(device, non_blocking=True)
                        
                        val_output = model(val_lr_input)
                        val_total_loss, val_mse_loss, val_grad_loss, val_freq_loss = criterion(val_output, val_hr_target)
                        
                        val_loss += val_total_loss.item()
                        val_mse += val_mse_loss.item()
                        val_grad += val_grad_loss.item()
                        val_freq += val_freq_loss.item()
                
                # Average validation losses
                val_loss /= len(val_loader)
                val_mse /= len(val_loader)
                val_grad /= len(val_loader)
                val_freq /= len(val_loader)
                val_losses.append(val_loss)
                
                print(f"\n📊 Step {step_count:,}/{total_steps:,} ({step_count/total_steps*100:.1f}%):")
                print(f"  Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, Grad: {val_grad:.6f}, Freq: {val_freq:.6f})")
                print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "models/wan2.2_resizer_20k_slow_best.pt")
                    print(f"  ✅ NEW BEST MODEL! Val Loss: {val_loss:.6f}")
                
                model.train()
            
            # Save checkpoint
            if step_count % save_interval == 0:
                checkpoint = {
                    'step': step_count,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'config': {
                        'total_steps': total_steps,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'dataset_path': dataset.dataset_path
                    }
                }
                torch.save(checkpoint, f"models/wan2.2_resizer_20k_slow_step_{step_count}.pt")
                print(f"  💾 Checkpoint saved at step {step_count:,}")
    
    progress_bar.close()
    
    # Final validation
    model.eval()
    final_val_loss = 0.0
    final_val_mse = 0.0
    final_val_grad = 0.0
    final_val_freq = 0.0

    with torch.no_grad():
        for val_lr_input, val_hr_target in val_loader:
            val_lr_input = val_lr_input.to(device, non_blocking=True)
            val_hr_target = val_hr_target.to(device, non_blocking=True)

            val_output = model(val_lr_input)
            val_total_loss, val_mse_loss, val_grad_loss, val_freq_loss = criterion(val_output, val_hr_target)

            final_val_loss += val_total_loss.item()
            final_val_mse += val_mse_loss.item()
            final_val_grad += val_grad_loss.item()
            final_val_freq += val_freq_loss.item()

    final_val_loss /= len(val_loader)
    final_val_mse /= len(val_loader)
    final_val_grad /= len(val_loader)
    final_val_freq /= len(val_loader)

    # Save final model
    torch.save(model.state_dict(), "models/wan2.2_resizer_20k_slow_final.pt")

    # Plot training curves
    plt.figure(figsize=(20, 6))

    # Plot 1: Training loss over steps
    plt.subplot(1, 4, 1)
    steps = [i * log_interval for i in range(len(train_losses))]
    plt.plot(steps, train_losses, label='Train Loss', alpha=0.7, linewidth=1)
    val_steps = [i * val_interval for i in range(len(val_losses))]
    plt.plot(val_steps, val_losses, label='Val Loss', linewidth=2, color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Progress (20K Steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Log scale
    plt.subplot(1, 4, 2)
    plt.plot(steps, train_losses, label='Train Loss', alpha=0.7, linewidth=1)
    plt.plot(val_steps, val_losses, label='Val Loss', linewidth=2, color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Progress (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Learning rate schedule
    plt.subplot(1, 4, 3)
    lr_steps = list(range(0, total_steps, 100))
    lr_values = []
    for step in lr_steps:
        if step < warmup_steps:
            lr_val = learning_rate * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr_val = learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
        lr_values.append(lr_val)

    plt.plot(lr_steps, lr_values, linewidth=2, color='green')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule\n(Warmup + Cosine)')
    plt.grid(True, alpha=0.3)

    # Plot 4: Loss comparison
    plt.subplot(1, 4, 4)
    if len(val_losses) > 0:
        improvement = ((val_losses[0] - best_val_loss) / val_losses[0]) * 100
        plt.bar(['Initial', 'Best', 'Final'],
                [val_losses[0] if val_losses else 0, best_val_loss, final_val_loss],
                color=['lightblue', 'green', 'orange'])
        plt.ylabel('Validation Loss')
        plt.title(f'Loss Improvement\n{improvement:.1f}% better')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves_20k_slow.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n🎉 SLOW LONG TRAINING COMPLETED!")
    print(f"📊 Final validation loss: {final_val_loss:.6f} (MSE: {final_val_mse:.6f})")
    print(f"📊 Best validation loss: {best_val_loss:.6f}")
    print(f"📈 Improvement: {((val_losses[0] - best_val_loss) / val_losses[0] * 100):.1f}% better than initial" if val_losses else "N/A")
    print(f"💾 Best model: models/wan2.2_resizer_20k_slow_best.pt")
    print(f"💾 Final model: models/wan2.2_resizer_20k_slow_final.pt")
    print(f"📈 Training curves: training_curves_20k_slow.png")
    print(f"🔥 Ready for ComfyUI integration!")

if __name__ == "__main__":
    train_slow_long_20k()
