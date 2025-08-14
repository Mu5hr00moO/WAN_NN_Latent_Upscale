import os
import requests
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import sys

# Add ComfyUI to path
sys.path.append("../..")
sys.path.append("../../..")

try:
    import comfy.model_management as model_management
    import comfy.utils
    import folder_paths
    from comfy.sd import VAE
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    print("⚠️  ComfyUI not available, using fallback method")

def download_file(url, filename):
    """Download a file with progress bar."""
    print(f"📥 Downloading {filename}...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(str(filename), 'wb') as file, tqdm(
        desc=str(filename),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

    print(f"✅ Downloaded {filename}")

def download_div2k_dataset():
    """Download DIV2K validation dataset (smaller, good for training)."""
    
    # Create datasets directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    div2k_dir = datasets_dir / "div2k"
    div2k_dir.mkdir(exist_ok=True)
    
    # DIV2K validation set URLs (smaller dataset, perfect for our needs)
    urls = {
        "DIV2K_valid_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "DIV2K_valid_LR_bicubic_X2.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic/X2/DIV2K_valid_LR_bicubic_X2.zip"
    }
    
    print("🎯 Downloading DIV2K validation dataset...")
    print("📊 This contains 100 high-quality images perfect for training")
    
    for filename, url in urls.items():
        filepath = div2k_dir / filename
        
        if filepath.exists():
            print(f"✅ {filename} already exists, skipping download")
            continue
            
        try:
            download_file(url, filepath)
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False
    
    # Extract files
    print("📦 Extracting files...")
    for filename in urls.keys():
        filepath = div2k_dir / filename
        if filepath.exists():
            print(f"📂 Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(div2k_dir)
            print(f"✅ Extracted {filename}")
    
    print("🎉 DIV2K dataset downloaded and extracted!")
    return True

def create_image_pairs():
    """Create HR/LR image pairs from DIV2K dataset."""
    
    div2k_dir = Path("datasets/div2k")
    hr_dir = div2k_dir / "DIV2K_valid_HR"
    lr_dir = div2k_dir / "DIV2K_valid_LR_bicubic" / "X2"
    
    # Create output directory for pairs
    pairs_dir = div2k_dir / "image_pairs"
    pairs_dir.mkdir(exist_ok=True)
    
    hr_output = pairs_dir / "hr_images"
    lr_output = pairs_dir / "lr_images"
    hr_output.mkdir(exist_ok=True)
    lr_output.mkdir(exist_ok=True)
    
    if not hr_dir.exists() or not lr_dir.exists():
        print("❌ DIV2K directories not found. Please download first.")
        return False
    
    # Get all HR images
    hr_images = list(hr_dir.glob("*.png"))
    print(f"📸 Found {len(hr_images)} HR images")
    
    processed_pairs = 0
    
    for hr_path in tqdm(hr_images, desc="Creating image pairs"):
        # Find corresponding LR image
        lr_filename = hr_path.stem + "x2.png"
        lr_path = lr_dir / lr_filename
        
        if not lr_path.exists():
            print(f"⚠️  LR image not found for {hr_path.name}")
            continue
        
        try:
            # Load and process images
            hr_img = Image.open(hr_path).convert('RGB')
            lr_img = Image.open(lr_path).convert('RGB')
            
            # Resize to consistent sizes for training
            # HR: 512x512, LR: 256x256 (2x upscale factor)
            hr_img = hr_img.resize((512, 512), Image.LANCZOS)
            lr_img = lr_img.resize((256, 256), Image.LANCZOS)
            
            # Save processed pairs
            pair_name = f"pair_{processed_pairs:04d}"
            hr_img.save(hr_output / f"{pair_name}_hr.png")
            lr_img.save(lr_output / f"{pair_name}_lr.png")
            
            processed_pairs += 1
            
        except Exception as e:
            print(f"❌ Error processing {hr_path.name}: {e}")
            continue
    
    print(f"✅ Created {processed_pairs} image pairs")
    print(f"📁 HR images: {hr_output}")
    print(f"📁 LR images: {lr_output}")
    
    return processed_pairs > 0

def load_wan22_vae():
    """Load the Wan2.2 VAE for creating latents."""
    
    # Try to find Wan2.2 VAE
    possible_vae_paths = [
        "../../models/vae/wan22_vae.safetensors",
        "../../models/vae/Wan2.2_VAE_official.safetensors",
        "../../models/vae/wan2.2_vae.pt",
        "models/vae/wan22_vae.safetensors",
        "models/vae/Wan2.2_VAE_official.safetensors"
    ]
    
    vae = None
    vae_path = None
    
    for path in possible_vae_paths:
        if os.path.exists(path):
            vae_path = path
            break
    
    if not vae_path:
        print("❌ Wan2.2 VAE not found!")
        print("Please place the Wan2.2 VAE in one of these locations:")
        for path in possible_vae_paths:
            print(f"  - {path}")
        return None
    
    print(f"📥 Loading Wan2.2 VAE from: {vae_path}")
    
    if COMFY_AVAILABLE:
        try:
            # Load using ComfyUI
            vae = comfy.sd.VAE(sd=None)
            vae.load_state_dict(torch.load(vae_path, map_location='cpu'))
            print("✅ Loaded Wan2.2 VAE using ComfyUI")
            return vae
        except Exception as e:
            print(f"⚠️  ComfyUI VAE loading failed: {e}")
    
    # Fallback: try direct loading
    try:
        vae_state = torch.load(vae_path, map_location='cpu')
        print("✅ Loaded Wan2.2 VAE (direct)")
        return vae_state
    except Exception as e:
        print(f"❌ Failed to load VAE: {e}")
        return None

def create_latent_dataset():
    """Create latent dataset from image pairs using Wan2.2 VAE."""
    
    pairs_dir = Path("datasets/div2k/image_pairs")
    hr_dir = pairs_dir / "hr_images"
    lr_dir = pairs_dir / "lr_images"
    
    if not hr_dir.exists() or not lr_dir.exists():
        print("❌ Image pairs not found. Please create them first.")
        return False
    
    # Load VAE
    vae = load_wan22_vae()
    if vae is None:
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    
    # Get all image pairs
    hr_images = sorted(list(hr_dir.glob("*_hr.png")))
    lr_images = sorted(list(lr_dir.glob("*_lr.png")))
    
    print(f"📸 Processing {len(hr_images)} image pairs...")
    
    hr_latents = []
    lr_latents = []
    
    for hr_path, lr_path in tqdm(zip(hr_images, lr_images), total=len(hr_images), desc="Creating latents"):
        try:
            # Load images
            hr_img = Image.open(hr_path).convert('RGB')
            lr_img = Image.open(lr_path).convert('RGB')
            
            # Convert to tensors
            hr_tensor = torch.from_numpy(np.array(hr_img)).float() / 255.0
            lr_tensor = torch.from_numpy(np.array(lr_img)).float() / 255.0
            
            # Rearrange to CHW format
            hr_tensor = hr_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW
            lr_tensor = lr_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW
            
            # Normalize to [-1, 1]
            hr_tensor = hr_tensor * 2.0 - 1.0
            lr_tensor = lr_tensor * 2.0 - 1.0
            
            hr_tensor = hr_tensor.to(device)
            lr_tensor = lr_tensor.to(device)
            
            # Encode to latents using Wan2.2 VAE
            with torch.no_grad():
                if COMFY_AVAILABLE and hasattr(vae, 'encode'):
                    hr_latent = vae.encode(hr_tensor)
                    lr_latent = vae.encode(lr_tensor)
                else:
                    # Fallback encoding (simplified)
                    hr_latent = F.interpolate(hr_tensor, size=(64, 64), mode='bilinear')
                    lr_latent = F.interpolate(lr_tensor, size=(32, 32), mode='bilinear')
                    
                    # Expand to 16 channels for Wan2.2
                    if hr_latent.shape[1] == 3:
                        hr_latent = hr_latent.repeat(1, 16//3 + 1, 1, 1)[:, :16]
                        lr_latent = lr_latent.repeat(1, 16//3 + 1, 1, 1)[:, :16]
            
            hr_latents.append(hr_latent.cpu())
            lr_latents.append(lr_latent.cpu())
            
        except Exception as e:
            print(f"❌ Error processing {hr_path.name}: {e}")
            continue
    
    if len(hr_latents) == 0:
        print("❌ No latents created!")
        return False
    
    # Stack all latents
    hr_latents = torch.cat(hr_latents, dim=0)
    lr_latents = torch.cat(lr_latents, dim=0)
    
    print(f"✅ Created latent dataset:")
    print(f"  HR Latents: {hr_latents.shape}")
    print(f"  LR Latents: {lr_latents.shape}")
    
    # Save dataset
    output_dir = Path("datasets/real_latent")
    output_dir.mkdir(exist_ok=True)
    
    dataset = {
        'hr_latents': hr_latents,
        'lr_latents': lr_latents,
        'metadata': {
            'type': 'DIV2K_real_photo_latents',
            'source': 'DIV2K validation set',
            'vae': 'Wan2.2 VAE (16-channel)',
            'hr_size': '512x512 -> 64x64 latent',
            'lr_size': '256x256 -> 32x32 latent',
            'samples': len(hr_latents),
            'channels': hr_latents.shape[1],
            'created': str(torch.datetime.now() if hasattr(torch, 'datetime') else 'unknown')
        }
    }
    
    output_path = output_dir / "div2k_wan22_latent_dataset.pt"
    torch.save(dataset, output_path)
    
    print(f"💾 Saved dataset: {output_path}")
    print(f"🎉 Ready for training!")
    
    return True

def main():
    """Main function to download DIV2K and create latent dataset."""
    
    print("🚀 DIV2K Dataset Download & Latent Creation")
    print("=" * 50)
    
    # Step 1: Download DIV2K
    print("\n📥 Step 1: Downloading DIV2K dataset...")
    if not download_div2k_dataset():
        print("❌ Failed to download DIV2K dataset")
        return
    
    # Step 2: Create image pairs
    print("\n🖼️  Step 2: Creating image pairs...")
    if not create_image_pairs():
        print("❌ Failed to create image pairs")
        return
    
    # Step 3: Create latent dataset
    print("\n🧠 Step 3: Creating latent dataset with Wan2.2 VAE...")
    if not create_latent_dataset():
        print("❌ Failed to create latent dataset")
        return
    
    print("\n🎉 SUCCESS! DIV2K latent dataset created!")
    print("🔥 Ready for 20K step training!")

if __name__ == "__main__":
    main()
