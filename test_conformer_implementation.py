#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify Conformer implementation without full training
"""

import torch
import numpy as np
from STM08gpu_Conformer_STM_corpus import ConformerClassifier

def test_model_architecture():
    """Test that model can be instantiated and forward pass works"""
    print("="*60)
    print("Testing Conformer Model Architecture")
    print("="*60)
    
    # Model parameters
    input_dim = 20  # frequency bins
    num_classes = 6
    d_model = 128
    
    # Create model
    model = ConformerClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=4,
        ffn_dim=512,
        num_layers=4,
        depthwise_conv_kernel_size=31,
        dropout=0.1
    )
    
    print(f"✓ Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 121  # time steps
    dummy_input = torch.randn(batch_size, input_dim, seq_len)
    
    print(f"\n✓ Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected output shape: ({batch_size}, {num_classes})")
    
    assert output.shape == (batch_size, num_classes), "Output shape mismatch!"
    
    print(f"\n✓ Forward pass successful!")
    print(f"✓ Sample output (logits):\n{output[0]}")
    
    # Test softmax probabilities
    probs = torch.softmax(output, dim=1)
    print(f"\n✓ Sample probabilities (should sum to 1):\n{probs[0]}")
    print(f"✓ Sum of probabilities: {probs[0].sum().item():.6f}")
    
    return True

def test_data_reshaping():
    """Test data reshaping from flattened to 2D"""
    print("\n" + "="*60)
    print("Testing Data Reshaping")
    print("="*60)
    
    # Original dimensions from STM06
    n_freq = 20
    n_time = 121
    flattened_size = n_freq * n_time  # 2420
    
    print(f"Original flattened size: {flattened_size}")
    print(f"Target 2D shape: ({n_freq}, {n_time})")
    
    # Create dummy flattened data
    n_samples = 100
    dummy_data = np.random.randn(n_samples, flattened_size)
    print(f"\n✓ Created dummy data: {dummy_data.shape}")
    
    # Reshape to 2D
    reshaped_data = dummy_data.reshape(n_samples, n_freq, n_time)
    print(f"✓ Reshaped data: {reshaped_data.shape}")
    
    # Verify reshape is reversible
    flattened_back = reshaped_data.reshape(n_samples, -1)
    assert np.allclose(dummy_data, flattened_back), "Reshape not reversible!"
    print(f"✓ Reshape is reversible")
    
    # Test normalization
    means = reshaped_data.mean(axis=(1, 2), keepdims=True)
    stds = reshaped_data.std(axis=(1, 2), keepdims=True)
    normalized = (reshaped_data - means) / (stds + 1e-8)
    
    print(f"\n✓ Normalized data shape: {normalized.shape}")
    print(f"✓ Mean after normalization: {normalized.mean():.6f} (should be ~0)")
    print(f"✓ Std after normalization: {normalized.std():.6f} (should be ~1)")
    
    return True

def test_gpu_availability():
    """Test GPU availability"""
    print("\n" + "="*60)
    print("Testing GPU Availability")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test moving model to GPU
        model = ConformerClassifier(
            input_dim=20,
            num_classes=6,
            d_model=128,
            num_heads=4,
            ffn_dim=512,
            num_layers=4,
            depthwise_conv_kernel_size=31,
            dropout=0.1
        )
        
        device = torch.device('cuda')
        model = model.to(device)
        
        # Test GPU forward pass
        dummy_input = torch.randn(2, 20, 121).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Successfully ran forward pass on GPU")
        print(f"✓ Output device: {output.device}")
        
    else:
        print("⚠ CUDA is not available - will use CPU")
        print("⚠ Training will be slow without GPU")
    
    return True

def test_batch_processing():
    """Test different batch sizes"""
    print("\n" + "="*60)
    print("Testing Batch Processing")
    print("="*60)
    
    model = ConformerClassifier(
        input_dim=20,
        num_classes=6,
        d_model=128,
        num_heads=4,
        ffn_dim=512,
        num_layers=4,
        depthwise_conv_kernel_size=31,
        dropout=0.1
    )
    
    model.eval()
    
    batch_sizes = [1, 8, 32, 128]
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 20, 121)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (batch_size, 6), f"Failed for batch_size={batch_size}"
        print(f"✓ Batch size {batch_size:3d}: output shape {output.shape}")
    
    return True

def main():
    print("\n" + "="*60)
    print("CONFORMER IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    try:
        # Run all tests
        test_model_architecture()
        test_data_reshaping()
        test_gpu_availability()
        test_batch_processing()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now run the full training with:")
        print("  python STM08gpu_Conformer_STM_corpus.py 0")
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
