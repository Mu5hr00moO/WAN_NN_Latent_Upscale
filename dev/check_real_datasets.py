import torch
import os
from pathlib import Path

def check_datasets():
    """Check what real datasets we have available."""
    
    print("🔍 Checking available real datasets...")
    print("=" * 50)
    
    datasets_dir = Path('datasets/real_latent')
    if not datasets_dir.exists():
        print("❌ No datasets directory found")
        return
    
    dataset_files = list(datasets_dir.glob('*.pt'))
    if not dataset_files:
        print("❌ No .pt dataset files found")
        return
    
    print(f"📁 Found {len(dataset_files)} dataset files:")
    print()
    
    best_dataset = None
    best_score = 0
    
    for file in dataset_files:
        try:
            print(f"📊 {file.name}:")
            data = torch.load(file, map_location='cpu')
            
            hr_latents = data['hr_latents']
            lr_latents = data['lr_latents']
            
            print(f"  HR shape: {hr_latents.shape}")
            print(f"  LR shape: {lr_latents.shape}")
            print(f"  Samples: {len(hr_latents)}")
            print(f"  Channels: {hr_latents.shape[1]}")
            
            if 'metadata' in data:
                metadata = data['metadata']
                print(f"  Type: {metadata.get('type', 'unknown')}")
                print(f"  Source: {metadata.get('source', 'unknown')}")
                print(f"  Description: {metadata.get('description', 'N/A')}")
                
                # Score datasets (prefer real photos, 16 channels, more samples)
                score = 0
                if 'real' in metadata.get('type', '').lower():
                    score += 10
                if 'photo' in metadata.get('type', '').lower():
                    score += 10
                if 'div2k' in metadata.get('source', '').lower():
                    score += 5
                if hr_latents.shape[1] == 16:  # 16 channels for Wan2.2
                    score += 5
                score += len(hr_latents) / 100  # More samples = better
                
                print(f"  Quality Score: {score:.1f}")
                
                if score > best_score:
                    best_score = score
                    best_dataset = file
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error loading: {e}")
            print()
    
    if best_dataset:
        print(f"🏆 BEST DATASET FOR TRAINING: {best_dataset.name}")
        print(f"   Quality Score: {best_score:.1f}")
        print()
        print("✅ This dataset will be used for 20K step training!")
    else:
        print("❌ No suitable dataset found for training")

if __name__ == "__main__":
    check_datasets()
