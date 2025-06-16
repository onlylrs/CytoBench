import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet feature extractor that removes the final classification layer
    
    Args:
        pretrained (bool): Whether to use pretrained weights
        model_name (str): Which ResNet model to use (default: 'resnet50')
    """
    def __init__(self, pretrained=True, model_name='resnet50'):
        super(ResNetFeatureExtractor, self).__init__()
        
        # Load the appropriate ResNet model
        if model_name == 'resnet50':
            full_model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            full_model = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            full_model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(full_model.children())[:-1])
        # Add a flatten layer to get a feature vector
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        """
        Extract features from input images
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            features (torch.Tensor): Feature tensor of shape (batch_size, feature_dim)
        """
        # Extract features
        features = self.features(x)
        # Flatten to get a feature vector
        features = self.flatten(features)
        
        return features