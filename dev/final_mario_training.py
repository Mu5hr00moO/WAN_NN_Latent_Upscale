import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import time

class WanNNLatentDataset(Dataset):
    """
    Dataset für FINALE Wan NN Training mit OFFIZIELLER VAE!
    """
    def __init__(self, dataset_path):
        print(f"🔥 Loading FINAL OFFICIAL dataset: {dataset_path}")
        
        data = torch.load(dataset_path, map_location='cpu')
        
        self.hr_latents = data['hr_latents']  # [N, 16, 64, 64]
        self.lr_latents = data['lr_latents']  # [N, 16, 32, 32]
        self.metadata = data['metadata']
        
        print(f"✅ Dataset loaded:")
        print(f"  Samples: {len(self.hr_latents)}")
        print(f"  HR Shape: {self.hr_latents.shape}")
        print(f"  LR Shape: {self.lr_latents.shape}")
        print(f"  VAE: {self.metadata.get('vae_used', 'unknown')}")
        print(f"  Source: {self.metadata.get('source', 'unknown')}")
        
    def __len__(self):
        return len(self.hr_latents)
    
    def __getitem__(self, idx):
        return self.lr_latents[idx], self.hr_latents[idx]

class FinalWanNN(nn.Module):
    """
    FINALE Wan NN Architektur - optimiert für echte VAE-Latents!
    """
    def __init__(self):
        super().__init__()
        
        print("🏗️  Building FINAL Wan NN Architecture...")
        
        # Input: [16, 32, 32] -> Output: [16, 64, 64]
        
        # Encoder für bessere Feature-Extraktion
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling-Pfad
        self.upsample = nn.Sequential(
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Output-Layer
        self.output = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Tanh()  # Für VAE-Latents geeignet
        )
        
        # Residual Connection für bessere Gradients
        self.residual_proj = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)
        
        print("✅ Architecture built!")
        
    def forward(self, x):
        # x: [B, 16, 32, 32]
        
        # Residual connection
        residual = self.residual_proj(x)  # [B, 16, 64, 64]
        
        # Main path
        features = self.encoder(x)  # [B, 64, 32, 32]
        upsampled = self.upsample(features)  # [B, 32, 64, 64]
        output = self.output(upsampled)  # [B, 16, 64, 64]
        
        # Combine with residual
        result = output + residual
        
        return result

class FinalMarioTrainer:
    """
    MARIO TRAINER für das finale Training! 🍄
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎮 MARIO TRAINER initialized on {self.device}")
        
        # Model
        self.model = FinalWanNN().to(self.device)
        
        # Optimizer - Adam mit guten Defaults
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # Loss - MSE für Latent-Space
        self.criterion = nn.MSELoss()
        
        # Scheduler für bessere Konvergenz
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Tracking
        self.train_losses = []
        self.best_loss = float('inf')
        
        print("🚀 MARIO TRAINER ready!")
    
    def train_epoch(self, dataloader, epoch):
        """Ein Trainings-Epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"🍄 Mario Epoch {epoch}")
        
        for batch_idx, (lr_batch, hr_batch) in enumerate(pbar):
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_hr = self.model(lr_batch)
            
            # Loss
            loss = self.criterion(pred_hr, hr_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping für Stabilität
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Tracking
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg': f'{epoch_loss/(batch_idx+1):.6f}'
            })
        
        avg_loss = epoch_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Checkpoint speichern"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses
        }
        
        # Regular checkpoint
        checkpoint_path = f"models/wan_nn_latent_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = "models/wan_nn_latent_best.pth"
            torch.save(checkpoint, best_path)
            print(f"🏆 NEW BEST MODEL saved! Loss: {loss:.6f}")
    
    def train(self, dataset_path, epochs=100, batch_size=8):
        """HAUPTTRAINING - MARIO STYLE! 🍄"""
        
        print(f"🚀 MARIO TRAINING STARTS!")
        print(f"🎯 Target: Fix Wan NN artifacts with OFFICIAL VAE data!")
        print(f"📊 Config: {epochs} epochs, batch_size={batch_size}")
        print(f"🔥 Dataset: {dataset_path}")
        
        # Dataset laden
        dataset = WanNNLatentDataset(dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        print(f"📦 DataLoader ready: {len(dataloader)} batches")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\n🍄 === MARIO EPOCH {epoch}/{epochs} ===")
            
            # Train
            avg_loss = self.train_epoch(dataloader, epoch)
            
            # Scheduler step
            self.scheduler.step(avg_loss)
            
            # Check if best
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, avg_loss, is_best)
            
            # Progress report
            elapsed = time.time() - start_time
            print(f"📊 Epoch {epoch} Summary:")
            print(f"   Loss: {avg_loss:.6f}")
            print(f"   Best: {self.best_loss:.6f}")
            print(f"   Time: {elapsed/60:.1f}min")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check
            if avg_loss < 1e-6:
                print(f"🎉 CONVERGENCE REACHED! Loss < 1e-6")
                break
        
        total_time = time.time() - start_time
        print(f"\n🏁 MARIO TRAINING COMPLETE!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Best loss: {self.best_loss:.6f}")
        print(f"💾 Model saved as: models/wan_nn_latent_best.pth")
        
        return self.best_loss

def main():
    """MARIO MAIN FUNCTION! 🍄"""
    
    print("🍄" * 50)
    print("🚀 MARIO'S FINAL WAN NN TRAINING!")
    print("🎯 Mission: Fix artifacts with OFFICIAL VAE!")
    print("🍄" * 50)
    
    # Erstelle models Ordner
    os.makedirs("models", exist_ok=True)
    
    # Dataset path
    dataset_path = "datasets/real_latent/final_official_wan22_dataset.pt"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Trainer erstellen
    trainer = FinalMarioTrainer()
    
    # TRAINING STARTEN! 🚀
    try:
        best_loss = trainer.train(
            dataset_path=dataset_path,
            epochs=50,  # Erstmal 50 Epochen
            batch_size=8
        )
        
        print(f"\n🎉 MARIO TRAINING SUCCESS!")
        print(f"🏆 Final best loss: {best_loss:.6f}")
        print(f"🔥 Ready to fix those artifacts!")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
