import torch
import torch.nn as nn
import torchvision.models as models

# Check for timm availability for additional ViT models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer (ViT) feature extractor that removes the final classification head

    Args:
        pretrained (bool): Whether to use pretrained weights
        model_name (str): Which ViT model to use (default: 'vit_b_16')
        freeze (bool): Whether to freeze the model parameters
    """
    def __init__(self, pretrained=True, model_name='vit_b_16', freeze=False):
        super(ViTFeatureExtractor, self).__init__()

        # Load the appropriate ViT model
        if model_name == 'vit_b_16':
            full_model = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_b_32':
            full_model = models.vit_b_32(pretrained=pretrained)
        elif model_name == 'vit_l_14':
            # Try torchvision first, fallback to timm
            if hasattr(models, 'vit_l_14'):
                full_model = models.vit_l_14(pretrained=pretrained)
                self._timm_wrapper = False
            elif TIMM_AVAILABLE:
                print("⚠️  torchvision.vit_l_14 not available, using timm as fallback")
                # Initialize timm model and set up attributes
                self._create_timm_vit_l_14(pretrained)
                full_model = None  # Will use timm forward path
            else:
                raise ImportError(
                    "ViT-L-14 is not available in your torchvision version and timm is not installed. "
                    "Please either:\n"
                    "1. Upgrade torchvision: pip install --upgrade torchvision\n"
                    "2. Install timm: pip install timm\n"
                    "3. Use ViT-L-16 instead: backbone.name='ViT-L-16'"
                )
        elif model_name == 'vit_l_16':
            full_model = models.vit_l_16(pretrained=pretrained)
        elif model_name == 'vit_l_32':
            full_model = models.vit_l_32(pretrained=pretrained)
        elif model_name == 'vit_h_14':
            full_model = models.vit_h_14(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}. Supported models: vit_b_16, vit_b_32, vit_l_14, vit_l_16, vit_l_32, vit_h_14")

        # Store the full model components (only for torchvision models)
        if full_model is not None:
            self.conv_proj = full_model.conv_proj
            self.encoder = full_model.encoder
            self.class_token = full_model.class_token
            self.seq_length = full_model.seq_length
            # Get the hidden dimension from the encoder
            self.hidden_dim = full_model.hidden_dim
        # For timm models, attributes are set in _create_timm_vit_l_14

        # Freeze parameters if specified
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def _create_timm_vit_l_14(self, pretrained=True):
        """
        Create ViT-L-14 using timm library as fallback

        Args:
            pretrained: Whether to load pretrained weights

        Returns:
            model: timm ViT model
        """
        # Create timm ViT-L-14 model
        timm_model = timm.create_model(
            'vit_large_patch14_224',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )

        # Set timm-specific attributes for this instance
        self._timm_wrapper = True
        self._timm_model = timm_model

        # Map timm attributes to torchvision-like structure for compatibility
        self.conv_proj = timm_model.patch_embed.proj
        self.class_token = timm_model.cls_token
        self.hidden_dim = timm_model.embed_dim
        self.encoder = timm_model.blocks

        # Define the timm forward function
        def timm_forward(x):
            features = timm_model.forward_features(x)
            return features[:, 0]  # Return class token

        return timm_forward
        
    def forward(self, x):
        """
        Extract features from input images

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)

        Returns:
            features (torch.Tensor): Feature tensor of shape (batch_size, feature_dim)
        """
        # Check if this is a timm wrapper
        if hasattr(self, '_timm_wrapper') and self._timm_wrapper:
            # This is a timm wrapper, use timm forward_features
            features = self._timm_model.forward_features(x)
            return features[:, 0]  # Return class token

        # Original torchvision ViT forward pass
        # Reshape and permute the input tensor
        n, c, h, w = x.shape
        p = self.conv_proj.kernel_size[0]
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Extract the class token as the feature representation
        features = x[:, 0]

        return features