"""
DINOv2 Feature Extractor Wrapper

This module provides a wrapper for DINOv2 models to ensure compatibility
with the existing backbone system.
"""

import torch
import torch.nn as nn
from .dinov2 import build_model as build_dinov2_model, build_transform as build_dinov2_transform


class DINOv2FeatureExtractor(nn.Module):
    """
    DINOv2 Feature Extractor wrapper for compatibility with existing system
    
    Args:
        model_name: Name of the DINOv2 model ('dinov2_vitl', 'gpfm', 'ccs')
        ckpt_path: Path to the model checkpoint
        device: Device to load the model on
        freeze: Whether to freeze the backbone parameters
    """
    
    def __init__(self, model_name, ckpt_path, device, freeze=True):
        super(DINOv2FeatureExtractor, self).__init__()
        
        self.model_name = model_name
        self.device = device
        self.freeze = freeze
        
        # Build DINOv2 model
        gpu_num = 1  # Single GPU for now
        self.backbone, self.embed_dim = build_dinov2_model(device, gpu_num, model_name, ckpt_path)
        
        # Freeze parameters if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"🔒 Frozen DINOv2 backbone '{model_name}' parameters")
        else:
            print(f"🔓 DINOv2 backbone '{model_name}' parameters are trainable")
    
    def forward(self, x):
        """
        Forward pass through the DINOv2 backbone
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            features: Feature tensor of shape (batch_size, embed_dim)
        """
        # DINOv2 models return features directly
        with torch.set_grad_enabled(not self.freeze):
            features = self.backbone(x)
        
        return features
    
    def get_feature_dim(self):
        """
        Get the feature dimension of the backbone
        
        Returns:
            int: Feature dimension
        """
        return self.embed_dim
    
    def train(self, mode=True):
        """
        Set the module in training mode
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
        """
        super().train(mode)
        
        # If backbone is frozen, keep it in eval mode
        if self.freeze:
            self.backbone.eval()
        else:
            self.backbone.train(mode)
        
        return self
    
    def eval(self):
        """Set the module in evaluation mode"""
        return self.train(False)


def get_dinov2_checkpoint_path(model_name):
    """
    Get the checkpoint path for a DINOv2 model
    
    Args:
        model_name: Name of the DINOv2 model
        
    Returns:
        str: Path to the checkpoint file
    """
    # Define checkpoint paths for DINOv2 models
    dinov2_ckpt_paths = {
        'dinov2_vitl': 'path1.pth',
        'gpfm': '/jhcnas3/Pathology/code/PrePath/models/ckpts/GPFM.pth', 
        'ccs': '/jhcnas4/Cervical/superpod_full/superpod/ckpts/100M/vitl_3.pth'
    }
    
    if model_name not in dinov2_ckpt_paths:
        raise ValueError(f"Unknown DINOv2 model '{model_name}'. Available models: {list(dinov2_ckpt_paths.keys())}")
    
    return dinov2_ckpt_paths[model_name]


def build_dinov2_backbone(model_name, device, freeze=True):
    """
    Build a DINOv2 backbone model
    
    Args:
        model_name: Name of the DINOv2 model
        device: Device to load the model on
        freeze: Whether to freeze the backbone parameters
        
    Returns:
        model: DINOv2FeatureExtractor instance
        preprocess: Preprocessing function
    """
    # Get checkpoint path
    ckpt_path = get_dinov2_checkpoint_path(model_name)
    
    # Build feature extractor
    model = DINOv2FeatureExtractor(model_name, ckpt_path, device, freeze)
    
    # Get preprocessing function
    preprocess = build_dinov2_transform()
    
    return model, preprocess
