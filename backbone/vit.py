import torch
import torch.nn as nn
import torchvision.models as models

class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer (ViT) feature extractor that removes the final classification head
    
    Args:
        pretrained (bool): Whether to use pretrained weights
        model_name (str): Which ViT model to use (default: 'vit_b_16')
    """
    def __init__(self, pretrained=True, model_name='vit_b_16'):
        super(ViTFeatureExtractor, self).__init__()
        
        # Load the appropriate ViT model
        if model_name == 'vit_b_16':
            full_model = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_l_16':
            full_model = models.vit_l_16(pretrained=pretrained)
        elif model_name == 'vit_h_14':
            full_model = models.vit_h_14(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")
        
        # Store the encoder part of the model
        self.encoder = full_model.encoder
        # Store the embedding layer
        self.embedding = full_model.conv_proj
        # Replace the classification head with identity
        self.head = nn.Identity()
        
    def forward(self, x):
        """
        Extract features from input images
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            features (torch.Tensor): Feature tensor of shape (batch_size, feature_dim)
        """
        # Create embeddings
        x = self.embedding(x)
        # Add class token and position embeddings
        x = self.encoder(x)
        # Extract the class token as the feature representation
        features = x[:, 0]
        
        return features