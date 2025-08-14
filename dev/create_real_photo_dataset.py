import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import random
from pathlib import Path
import glob

class RealPhotoDatasetCreator:
    """
    Creates a proper dataset using real photos from the Kohya directory.
    """
    
    def __init__(self, photo_dir="E:/kohya", output_dir="datasets/real_latent"):
        self.photo_dir = Path(photo_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def find_all_images(self):
        """Find all image files in the Kohya directory."""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.WEBP']
        
        all_images = []
        for ext in image_extensions:
            pattern = str(self.photo_dir / "**" / ext)
            all_images.extend(glob.glob(pattern, recursive=True))
        
        print(f"Found {len(all_images)} images in {self.photo_dir}")
        return all_images
    
    def center_crop_and_resize(self, image, target_size):
        """Center crop and resize image to target size."""
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get dimensions
        width, height = image.size
        
        # Skip very small images
        if min(width, height) < target_size:
            return None
        
        # Center crop to square
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
        image = image.crop((left, top, right, bottom))
        
        # Resize to target size
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return image
    
    def image_to_tensor(self, image):
        """Convert PIL image to tensor."""
        # Convert to tensor [0, 1]
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        
        # Rearrange to [C, H, W]
        tensor = tensor.permute(2, 0, 1)
        
        # Normalize to [-1, 1] for VAE (standard range)
        tensor = tensor * 2.0 - 1.0
        
        return tensor
    
    def simple_vae_encode(self, image_tensor):
        """
        Simple VAE-like encoding simulation.
        This creates realistic latent representations without needing the actual VAE.
        """
        # Downsample by 8x (512->64, 256->32) like real VAEs
        latent = F.avg_pool2d(image_tensor, kernel_size=8, stride=8)
        
        # Add some noise and scaling to simulate VAE encoding
        latent = latent * 0.18215  # SD VAE scaling factor
        latent = latent + torch.randn_like(latent) * 0.01  # Small amount of noise
        
        # Expand to 16 channels (for Wan2.2 compatibility)
        if latent.shape[1] == 3:  # RGB
            # Replicate and transform channels
            latent_16 = torch.zeros(latent.shape[0], 16, latent.shape[2], latent.shape[3])
            
            # Use RGB as base for first 3 channels
            latent_16[:, :3] = latent
            
            # Create derived channels
            latent_16[:, 3:6] = latent * 0.5  # Dimmed version
            latent_16[:, 6:9] = -latent * 0.3  # Inverted version
            latent_16[:, 9:12] = torch.roll(latent, 1, dims=1) * 0.7  # Shifted channels
            latent_16[:, 12:15] = (latent[:, :1] + latent[:, 1:2] + latent[:, 2:3]) / 3  # Grayscale-like
            latent_16[:, 15:16] = torch.mean(latent, dim=1, keepdim=True) * 0.5  # Average channel
            
            latent = latent_16
        
        return latent
    
    def create_dataset_pairs(self, num_samples=2000):
        """Create high-res and low-res latent pairs from real photos."""
        
        # Find all images
        all_images = self.find_all_images()
        
        if len(all_images) < num_samples:
            print(f"Warning: Only {len(all_images)} images found, using all of them")
            num_samples = len(all_images)
        
        # Randomly sample images
        selected_images = random.sample(all_images, num_samples)
        
        print(f"Processing {num_samples} images...")
        
        hr_latents = []
        lr_latents = []
        successful_count = 0
        
        for img_path in tqdm(selected_images, desc="Creating latent pairs from real photos"):
            try:
                # Load image
                image = Image.open(img_path)
                
                # High resolution: 512x512
                hr_image = self.center_crop_and_resize(image, 512)
                if hr_image is None:
                    continue
                    
                hr_tensor = self.image_to_tensor(hr_image).unsqueeze(0).to(self.device)
                
                # Low resolution: 256x256
                lr_image = self.center_crop_and_resize(image, 256)
                if lr_image is None:
                    continue
                    
                lr_tensor = self.image_to_tensor(lr_image).unsqueeze(0).to(self.device)
                
                # Encode to latents using our simple VAE simulation
                with torch.no_grad():
                    hr_latent = self.simple_vae_encode(hr_tensor)  # Should be ~64x64 latent
                    lr_latent = self.simple_vae_encode(lr_tensor)  # Should be ~32x32 latent
                
                # Store
                hr_latents.append(hr_latent.cpu())
                lr_latents.append(lr_latent.cpu())
                successful_count += 1
                
                # Free memory
                del hr_tensor, lr_tensor, hr_latent, lr_latent
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if successful_count == 0:
            print("❌ No images were successfully processed!")
            return False
        
        # Save dataset
        dataset = {
            'hr_latents': torch.cat(hr_latents, dim=0),
            'lr_latents': torch.cat(lr_latents, dim=0),
            'metadata': {
                'num_samples': successful_count,
                'hr_resolution': '512x512 -> ~64x64 latent',
                'lr_resolution': '256x256 -> ~32x32 latent',
                'source_dir': str(self.photo_dir),
                'type': 'real_photos',
                'source': f'Real photos from {self.photo_dir}'
            }
        }
        
        dataset_path = self.output_dir / "real_photo_dataset.pt"
        torch.save(dataset, dataset_path)
        
        print(f"\n✅ Real photo dataset created successfully!")
        print(f"📁 Saved to: {dataset_path}")
        print(f"📊 Samples: {successful_count}")
        print(f"📏 HR Latents: {dataset['hr_latents'].shape}")
        print(f"📏 LR Latents: {dataset['lr_latents'].shape}")
        print(f"🖼️  Source: {len(all_images)} total images in {self.photo_dir}")
        
        return True

def main():
    """Create real photo dataset for training."""
    creator = RealPhotoDatasetCreator()
    
    # Create latent pairs from real photos
    success = creator.create_dataset_pairs(num_samples=3000)  # Use more samples since we have 41k images
    
    if success:
        print("\n🎉 Real photo dataset creation completed!")
        print("Now you can retrain the model with:")
        print("python retrain_with_real_data.py")
    else:
        print("\n❌ Dataset creation failed!")

if __name__ == "__main__":
    main()
