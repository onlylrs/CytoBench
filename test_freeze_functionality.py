#!/usr/bin/env python3
"""
Test script to verify backbone freezing functionality in segmentation model
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from model.cell_seg.segmentation_model import build_segmentation_model

def test_freeze_functionality():
    """Test the freeze backbone functionality"""
    print("TESTING BACKBONE FREEZE FUNCTIONALITY")
    print("="*60)
    
    # Test 1: Model without freezing
    print("\n1. Testing model WITHOUT backbone freezing:")
    model_unfrozen = build_segmentation_model(
        'maskrcnn_resnet50_fpn', 
        num_classes=3, 
        pretrained=False, 
        freeze_backbone=False
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model_unfrozen.parameters())
    trainable_params = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable percentage: {100*trainable_params/total_params:.1f}%")
    
    # Test 2: Model with freezing
    print("\n2. Testing model WITH backbone freezing:")
    model_frozen = build_segmentation_model(
        'maskrcnn_resnet50_fpn', 
        num_classes=3, 
        pretrained=False, 
        freeze_backbone=True
    )
    
    # Count parameters
    total_params_frozen = sum(p.numel() for p in model_frozen.parameters())
    trainable_params_frozen = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params_frozen:,}")
    print(f"   Trainable parameters: {trainable_params_frozen:,}")
    print(f"   Trainable percentage: {100*trainable_params_frozen/total_params_frozen:.1f}%")
    
    # Test 3: Verify backbone is actually frozen
    print("\n3. Verifying backbone freeze status:")
    backbone_params = sum(p.numel() for p in model_frozen.backbone.parameters())
    backbone_trainable = sum(p.numel() for p in model_frozen.backbone.parameters() if p.requires_grad)
    
    print(f"   Backbone total parameters: {backbone_params:,}")
    print(f"   Backbone trainable parameters: {backbone_trainable:,}")
    print(f"   Backbone frozen: {'✓' if backbone_trainable == 0 else '✗'}")
    
    # Test 4: Check which components are trainable
    print("\n4. Component-wise trainability analysis:")
    components = {
        'backbone': model_frozen.backbone,
        'rpn': model_frozen.rpn,
        'roi_heads': model_frozen.roi_heads
    }
    
    for comp_name, component in components.items():
        comp_total = sum(p.numel() for p in component.parameters())
        comp_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
        comp_frozen = comp_total - comp_trainable
        
        print(f"   {comp_name}:")
        print(f"     Total: {comp_total:,}")
        print(f"     Trainable: {comp_trainable:,}")
        print(f"     Frozen: {comp_frozen:,}")
        print(f"     Status: {'Frozen' if comp_trainable == 0 else 'Trainable'}")
    
    # Test 5: Test gradient flow
    print("\n5. Testing gradient flow:")
    model_frozen.train()
    
    # Create dummy data
    images = [torch.randn(3, 224, 224)]
    targets = [{
        'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'masks': torch.randint(0, 2, (1, 224, 224), dtype=torch.uint8)
    }]
    
    try:
        # Forward pass
        loss_dict = model_frozen(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients
        backbone_grads = sum(1 for p in model_frozen.backbone.parameters() if p.grad is not None)
        rpn_grads = sum(1 for p in model_frozen.rpn.parameters() if p.grad is not None)
        roi_grads = sum(1 for p in model_frozen.roi_heads.parameters() if p.grad is not None)
        
        print(f"   Backbone parameters with gradients: {backbone_grads}")
        print(f"   RPN parameters with gradients: {rpn_grads}")
        print(f"   ROI heads parameters with gradients: {roi_grads}")
        
        print(f"   Gradient flow test: {'✓ PASSED' if backbone_grads == 0 and (rpn_grads > 0 or roi_grads > 0) else '✗ FAILED'}")
        
    except Exception as e:
        print(f"   ✗ Gradient flow test failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    
    reduction_ratio = (trainable_params - trainable_params_frozen) / trainable_params * 100
    print(f"Parameter reduction: {reduction_ratio:.1f}%")
    print(f"Trainable parameters reduced from {trainable_params:,} to {trainable_params_frozen:,}")
    
    if backbone_trainable == 0:
        print("✓ Backbone successfully frozen")
    else:
        print("✗ Backbone freezing failed")
    
    print("\nFreeze functionality test completed!")

if __name__ == "__main__":
    test_freeze_functionality()
