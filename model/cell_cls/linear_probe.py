import torch
import torch.nn as nn

class LinearProbeModel(nn.Module):
    """
    Linear probe model for cell classification
    
    Args:
        backbone: Backbone model for feature extraction
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze the backbone parameters
    """
    def __init__(self, backbone, num_classes, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        
        # Get feature dimension from backbone
        # Create a small dummy input to determine feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            
            # Handle case where backbone returns a tuple instead of a tensor
            if isinstance(features, tuple):
                features = features[0]  # Take the first element (usually the features)
            
            feature_dim = features.shape[1]
        
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            logits: Classification logits
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Handle case where backbone returns a tuple instead of a tensor
        if isinstance(features, tuple):
            features = features[0]  # Take the first element (usually the features)
            
        # Apply classifier
        logits = self.classifier(features)
        return logits
