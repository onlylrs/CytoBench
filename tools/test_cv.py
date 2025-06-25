#!/usr/bin/env python3
"""
Quick test script for cross validation functionality
This creates a minimal test to verify CV works without requiring full training
"""

import os
import sys
import yaml
import torch
import numpy as np

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cell_cls.dataset import build_cv_dataloaders
from backbone.build_model import build_backbone

def create_test_config():
    """Create a minimal test configuration"""
    return {
        'common': {
            'gpu': 0,
            'num_workers': 0,  # Avoid multiprocessing issues in testing
            'seed': 42
        },
        'data': {
            'dataset': 'Herlev'
        },
        'backbone': {
            'name': 'ResNet18',
            'pretrained': False,  # Faster for testing
            'freeze': True
        },
        'model': {
            'feature_dim': 512
        },
        'training': {
            'batch_size': 4,  # Small batch size for testing
            'epochs': 2,      # Very few epochs for testing
            'lr': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'Adam'
        },
        'evaluation': {
            'batch_size': 4,
            'compute_ci': False,  # Disable CI for faster testing
            'cv_enabled': True,
            'cv_folds': 3,        # Small number of folds for testing
            'cv_seed': 42
        },
        'output': {
            'model_dir': 'test_checkpoints',
            'results_dir': 'test_results'
        }
    }

def test_cv_data_loading():
    """Test that cross validation data loading works"""
    print("Testing cross validation data loading...")
    
    config = create_test_config()
    
    try:
        # Test building dataloaders for each fold
        total_samples_per_fold = []
        
        for fold in range(config['evaluation']['cv_folds']):
            print(f"\n  Testing fold {fold + 1}...")
            
            # Build dataloaders for this fold
            dataloaders = build_cv_dataloaders(config, fold)
            
            # Check basic properties
            assert 'train' in dataloaders, "Missing train dataloader"
            assert 'val' in dataloaders, "Missing val dataloader"
            assert 'test' in dataloaders, "Missing test dataloader"
            assert 'num_classes' in dataloaders, "Missing num_classes"
            
            train_size = dataloaders['train_size']
            val_size = dataloaders['val_size']
            test_size = dataloaders['test_size']
            total_size = train_size + val_size + test_size
            
            total_samples_per_fold.append(total_size)
            
            print(f"    Train: {train_size}, Val: {val_size}, Test: {test_size}")
            print(f"    Total: {total_size}, Classes: {dataloaders['num_classes']}")
            
            # Test loading a batch from each dataloader
            if len(dataloaders['train']) > 0:
                train_batch = next(iter(dataloaders['train']))
                images, labels = train_batch
                print(f"    Train batch shape: {images.shape}, Labels: {labels.shape}")
            
            if len(dataloaders['test']) > 0:
                test_batch = next(iter(dataloaders['test']))
                images, labels = test_batch
                print(f"    Test batch shape: {images.shape}, Labels: {labels.shape}")
        
        # Check that total samples are consistent across folds
        if len(set(total_samples_per_fold)) == 1:
            print(f"\n  ✓ Consistent total samples across folds: {total_samples_per_fold[0]}")
        else:
            print(f"\n  ⚠ Warning: Inconsistent total samples: {total_samples_per_fold}")
        
        print("  ✓ Cross validation data loading test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in CV data loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cv_model_creation():
    """Test that models can be created for CV"""
    print("\nTesting cross validation model creation...")
    
    config = create_test_config()
    
    try:
        # Build backbone
        backbone, preprocess = build_backbone(config)
        print(f"  ✓ Backbone created: {config['backbone']['name']}")
        
        # Test with a sample from CV dataloader
        dataloaders = build_cv_dataloaders(config, 0)  # Test with fold 0
        
        # Import here to avoid circular imports
        from model.cell_cls.linear_probe import LinearProbeModel
        
        # Create model
        num_classes = dataloaders['num_classes']
        feature_dim = config['model']['feature_dim']
        freeze_backbone = config['backbone']['freeze']
        dropout_p = config['model'].get('dropout_p', 0.5)

        model = LinearProbeModel(backbone, num_classes, feature_dim, freeze_backbone, dropout_p)
        print(f"  ✓ Model created with {num_classes} classes")
        
        # Test forward pass
        if len(dataloaders['train']) > 0:
            sample_batch = next(iter(dataloaders['train']))
            images, labels = sample_batch
            
            model.eval()
            with torch.no_grad():
                outputs = model(images)
            
            print(f"  ✓ Forward pass successful: {images.shape} -> {outputs.shape}")
            
            # Check output shape
            expected_shape = (images.shape[0], num_classes)
            if outputs.shape == expected_shape:
                print(f"  ✓ Output shape correct: {outputs.shape}")
            else:
                print(f"  ✗ Output shape mismatch: expected {expected_shape}, got {outputs.shape}")
                return False
        
        print("  ✓ Cross validation model creation test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error in CV model creation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Cross Validation Quick Test")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    success_count = 0
    total_tests = 2
    
    # Run tests
    if test_cv_data_loading():
        success_count += 1
    
    if test_cv_model_creation():
        success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✓ All quick tests passed!")
        print("\nYou can now run full cross validation with:")
        print("python tools/train_cv.py --config configs/cell_cls/cross_validation.yaml")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
