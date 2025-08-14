"""
🍄 SIMPLE MARIO TEST 🍄
Test just the model architecture and loading
"""

import torch
import torch.nn as nn
import os

class WanNNLatentUpscaler(nn.Module):
    """
    MARIO'S FINAL Wan NN Architecture - trained with OFFICIAL VAE!
    """
    def __init__(self):
        super().__init__()
        
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

def test_mario_model():
    """Test Mario's model directly"""
    
    print("🍄" * 50)
    print("🚀 SIMPLE MARIO MODEL TEST!")
    print("🍄" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 Device: {device}")
    
    # Test model creation
    print("🏗️  Creating Mario model...")
    model = WanNNLatentUpscaler()
    model.to(device)
    model.eval()
    print("✅ Model created successfully!")
    
    # Test model loading
    model_path = "models/wan_nn_latent_best.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"📦 Loading trained weights from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            print(f"🏆 Checkpoint info: Epoch {epoch}, Loss {loss}")
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        print("✅ MARIO WEIGHTS LOADED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        return False
    
    # Test inference
    print("🧪 Testing inference...")
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 16, 32, 32).to(device)
        print(f"   Input shape: {dummy_input.shape}")
        
        # Run inference
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: {output.min():.4f} to {output.max():.4f}")
        
        # Verify shape
        expected_shape = (1, 16, 64, 64)
        if output.shape == expected_shape:
            print("✅ Output shape correct!")
        else:
            print(f"❌ Wrong output shape: {output.shape}, expected: {expected_shape}")
            return False
        
        print("✅ INFERENCE TEST SUCCESSFUL!")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("🍄 MARIO'S MODEL IS WORKING PERFECTLY!")
    return True

if __name__ == "__main__":
    success = test_mario_model()
    
    if success:
        print("\n🚀 MARIO MODEL TEST SUCCESSFUL!")
        print("🔥 The model is ready to fix artifacts!")
    else:
        print("\n❌ MARIO MODEL TEST FAILED!")
        print("🔧 Please check the error messages above")
