import pytest
import torch
from models.model import FastMNIST

@pytest.fixture
def model():
    return FastMNIST()

def test_model_structure(model):
    """Test the model structure and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be < 20000"

def test_forward_pass(model):
    """Test the forward pass with a batch of data."""
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 10), f"Expected shape (32, 10), got {output.shape}"
    
    # Check output is probability distribution
    assert torch.allclose(torch.exp(output).sum(dim=1), torch.ones(batch_size), atol=1e-6)

def test_batch_norm_layers(model):
    """Test that batch normalization layers are present."""
    assert hasattr(model, 'bn1'), "Model missing bn1 layer"
    assert hasattr(model, 'bn2'), "Model missing bn2 layer"
    assert hasattr(model, 'bn3'), "Model missing bn3 layer"

def test_dropout_layer(model):
    """Test that dropout layer is present with correct rate."""
    assert hasattr(model, 'dropout'), "Model missing dropout layer"
    dropout_rate = model.dropout.p
    assert dropout_rate == 0.4, f"Expected dropout rate 0.4, got {dropout_rate}"

def test_conv_layers(model):
    """Test convolutional layers configuration."""
    # Test conv1
    assert model.conv1.in_channels == 1, "Conv1 should have 1 input channel"
    assert model.conv1.out_channels == 8, "Conv1 should have 8 output channels"
    
    # Test conv2
    assert model.conv2.in_channels == 8, "Conv2 should have 8 input channels"
    assert model.conv2.out_channels == 16, "Conv2 should have 16 output channels"
    
    # Test conv3
    assert model.conv3.in_channels == 16, "Conv3 should have 16 input channels"
    assert model.conv3.out_channels == 32, "Conv3 should have 32 output channels"

def test_fc_layers(model):
    """Test fully connected layers configuration."""
    assert model.fc1.in_features == 32 * 3 * 3, "FC1 input features incorrect"
    assert model.fc1.out_features == 32, "FC1 output features should be 32"
    assert model.fc2.in_features == 32, "FC2 input features should be 32"
    assert model.fc2.out_features == 10, "FC2 output features should be 10"

def test_gradient_flow(model):
    """Test that gradients can flow through the model."""
    x = torch.randn(1, 1, 28, 28, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
