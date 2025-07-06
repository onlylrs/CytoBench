#!/usr/bin/env python3
"""
Simple test for backbone freezing
"""

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

def test_basic_freeze():
    print("Testing basic freeze functionality...")
    
    # Create model
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Count initial trainable parameters
    initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initial trainable parameters: {initial_trainable:,}")
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Count after freezing
    after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after freeze: {after_freeze:,}")
    
    # Check backbone specifically
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f"Backbone trainable: {backbone_trainable}")
    
    print("✓ Basic freeze test completed")

if __name__ == "__main__":
    test_basic_freeze()
