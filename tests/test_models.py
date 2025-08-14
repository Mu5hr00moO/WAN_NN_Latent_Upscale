"""
Tests for model architectures and functionality
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_resizer import LatentResizer, WanLatentResizer


class TestLatentResizer:
    """Test the basic LatentResizer model (SD1.5/SDXL)"""
    
    def test_model_creation(self):
        """Test that model can be created with correct parameters"""
        model = LatentResizer(4, 4, 128)
        assert model.in_channels == 4
        assert model.out_channels == 4
        assert model.hidden_dim == 128
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        model = LatentResizer(4, 4, 128)
        input_tensor = torch.randn(1, 4, 32, 32)
        output = model(input_tensor)
        
        expected_shape = (1, 4, 64, 64)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_batch_processing(self):
        """Test that model handles different batch sizes"""
        model = LatentResizer(4, 4, 128)
        
        for batch_size in [1, 2, 4, 8]:
            input_tensor = torch.randn(batch_size, 4, 32, 32)
            output = model(input_tensor)
            expected_shape = (batch_size, 4, 64, 64)
            assert output.shape == expected_shape
    
    def test_different_input_sizes(self):
        """Test that model handles different input sizes"""
        model = LatentResizer(4, 4, 128)
        
        # Test various input sizes
        test_sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
        
        for h, w in test_sizes:
            input_tensor = torch.randn(1, 4, h, w)
            output = model(input_tensor)
            expected_shape = (1, 4, h * 2, w * 2)
            assert output.shape == expected_shape
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model"""
        model = LatentResizer(4, 4, 128)
        input_tensor = torch.randn(1, 4, 32, 32, requires_grad=True)
        
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert input_tensor.grad is not None
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))


class TestWanLatentResizer:
    """Test the WanLatentResizer model (Wan2.2/Flux)"""
    
    def test_model_creation(self):
        """Test that Wan model can be created with correct parameters"""
        model = WanLatentResizer(16, 16, 256)
        assert model.in_channels == 16
        assert model.out_channels == 16
        assert model.hidden_dim == 256
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        model = WanLatentResizer(16, 16, 256)
        input_tensor = torch.randn(1, 16, 32, 32)
        output = model(input_tensor)
        
        expected_shape = (1, 16, 64, 64)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_batch_processing(self):
        """Test that Wan model handles different batch sizes"""
        model = WanLatentResizer(16, 16, 256)
        
        for batch_size in [1, 2, 4]:
            input_tensor = torch.randn(batch_size, 16, 32, 32)
            output = model(input_tensor)
            expected_shape = (batch_size, 16, 64, 64)
            assert output.shape == expected_shape
    
    def test_model_parameters(self):
        """Test that model has reasonable number of parameters"""
        model = WanLatentResizer(16, 16, 256)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be less than 10M parameters for efficiency
        assert total_params < 10_000_000, f"Model too large: {total_params} parameters"
        # Should be more than 100K parameters for capacity
        assert total_params > 100_000, f"Model too small: {total_params} parameters"


class TestModelComparison:
    """Test comparisons between models and baselines"""
    
    def test_output_different_from_input(self):
        """Test that model output is different from input (not identity)"""
        model = LatentResizer(4, 4, 128)
        input_tensor = torch.randn(1, 4, 32, 32)
        
        # Upsample input with bilinear for comparison
        upsampled_input = torch.nn.functional.interpolate(
            input_tensor, scale_factor=2, mode='bilinear', align_corners=False
        )
        
        output = model(input_tensor)
        
        # Model output should be different from simple bilinear upsampling
        assert not torch.allclose(output, upsampled_input, atol=1e-3)
    
    def test_model_deterministic(self):
        """Test that model produces deterministic outputs"""
        model = LatentResizer(4, 4, 128)
        model.eval()  # Set to eval mode
        
        input_tensor = torch.randn(1, 4, 32, 32)
        
        # Run twice and compare
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)
        
        assert torch.allclose(output1, output2), "Model should be deterministic in eval mode"
    
    def test_model_training_mode(self):
        """Test that model behaves differently in train vs eval mode"""
        model = WanLatentResizer(16, 16, 256)
        input_tensor = torch.randn(1, 16, 32, 32)
        
        # Test that model can switch between modes
        model.train()
        assert model.training
        
        model.eval()
        assert not model.training


if __name__ == "__main__":
    pytest.main([__file__])
