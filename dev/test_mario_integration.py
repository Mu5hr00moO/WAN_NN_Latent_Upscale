"""
🍄 MARIO INTEGRATION TEST 🍄
Test script to verify Mario's Wan NN integration works correctly
"""

import sys
import os

# Add ComfyUI to path
sys.path.append("../..")
sys.path.append("../../..")

def test_mario_integration():
    """Test if Mario's Wan NN can be imported and used"""
    
    print("🍄" * 50)
    print("🚀 TESTING MARIO'S WAN NN INTEGRATION!")
    print("🍄" * 50)
    
    try:
        # Test import
        print("📦 Testing import...")
        sys.path.append("custom_nodes/denrakeiw_nodes")
        from wan_nn_latent_upscaler import WanNNLatentUpscalerNode
        print("✅ Import successful!")
        
        # Test node creation
        print("🏗️  Testing node creation...")
        node = WanNNLatentUpscalerNode()
        print("✅ Node creation successful!")
        
        # Test input types
        print("🔍 Testing input types...")
        input_types = node.INPUT_TYPES()
        print(f"✅ Input types: {input_types}")
        
        # Test model loading
        print("🍄 Testing Mario model loading...")
        try:
            node.load_model()
            print("✅ MARIO MODEL LOADED SUCCESSFULLY!")
            print(f"   Device: {node.device}")
            print(f"   Model: {type(node.model).__name__}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
        
        # Test with dummy data
        print("🧪 Testing with dummy latent...")
        import torch
        
        # Create dummy latent (batch_size=1, channels=16, height=32, width=32)
        dummy_latent = {
            "samples": torch.randn(1, 16, 32, 32)
        }
        
        try:
            result = node.upscale_latent(dummy_latent, strength=1.0)
            output_latent = result[0]
            output_shape = output_latent["samples"].shape
            
            print(f"✅ MARIO UPSCALING TEST SUCCESSFUL!")
            print(f"   Input shape: {dummy_latent['samples'].shape}")
            print(f"   Output shape: {output_shape}")
            
            # Verify output shape
            expected_shape = (1, 16, 64, 64)
            if output_shape == expected_shape:
                print(f"✅ Output shape correct: {output_shape}")
            else:
                print(f"❌ Output shape wrong: {output_shape}, expected: {expected_shape}")
                return False
                
        except Exception as e:
            print(f"❌ Upscaling test failed: {e}")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("🍄 MARIO'S WAN NN IS READY TO FIX ARTIFACTS!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mario_integration()
    
    if success:
        print("\n🚀 INTEGRATION SUCCESSFUL!")
        print("🍄 You can now use Mario's Wan NN in ComfyUI!")
        print("   1. Restart ComfyUI")
        print("   2. Look for '🍄 Mario's Wan NN Latent Upscaler' in the node menu")
        print("   3. Connect a latent (32x32) and get upscaled latent (64x64)")
        print("   4. Enjoy artifact-free results! 🔥")
    else:
        print("\n❌ INTEGRATION FAILED!")
        print("🔧 Please check the error messages above")
