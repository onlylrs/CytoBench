import torch
import torch.nn as nn

class LinearProbeModel(nn.Module):
    """
    Linear probe model for cell classification

    Args:
        backbone: Backbone model for feature extraction
        num_classes: Number of output classes
        feature_dim: Feature dimension (0 for auto-detection)
        freeze_backbone: Whether to freeze the backbone parameters
        dropout_p: Dropout probability (default: 0.5)
    """
    def __init__(self, backbone, num_classes, feature_dim=0, freeze_backbone=False, dropout_p=0.5):
        super().__init__()
        self.backbone = backbone

        # Get feature dimension from backbone
        if feature_dim == 0:
            # Auto-detect feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)

                # Handle case where backbone returns a tuple instead of a tensor
                if isinstance(features, tuple):
                    features = features[0]  # Take the first element (usually the features)

                feature_dim = features.shape[1]

        # Create classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, num_classes)
        )
        
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
