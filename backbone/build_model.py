import os
import torch
import open_clip
from .resnet import ResNetFeatureExtractor
from .vit import ViTFeatureExtractor

def build_backbone(config):
    """
    Build backbone model based on configuration
    
    Args:
        config: Configuration dictionary with backbone settings
        
    Returns:
        model: Backbone model
        preprocess: Preprocessing function for images
    """
    model_name = config['backbone']['name']
    ckpt_root = '/jhcnas2/shared_data/public/others/Cytology_VLP/ckpts'
    
    available_models = [
        'ResNet50', 'CLIP', 'SigLIP', 
        'SigLIP2-ViT-B', 'SigLIP2-ViT-L', 'SigLIP2-ViT-SO400M'
    ]
    
    if model_name not in available_models:
        raise ValueError(f"Model {model_name} not available. Choose from: {available_models}")
    
    # Build model based on name
    if model_name == 'ResNet50':
        # For API consistency, get preprocess from CLIP
        # _, _, preprocess = open_clip.create_model_and_transforms(
        #     'ViT-L-14', 
        #     pretrained=os.path.join(ckpt_root, '2025_04_21-15_22_04-model_ViT-L-14-lr_0.0005-b_64-j_4-p_amp/checkpoints/epoch_40.pt')
        # )
        # # Build ResNet50 feature extractor
        # model = ResNetFeatureExtractor(pretrained=config['backbone']['pretrained'])
        _, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=os.path.join(ckpt_root, '2025_04_21-15_22_04-model_ViT-L-14-lr_0.0005-b_64-j_4-p_amp/checkpoints/epoch_40.pt')) # for API consistency
        # Load ResNet50 and create feature extractor (remove final classification layer)
        full_model = torch.hub.load('pytorch/vision:v0.22.0', 'resnet50', pretrained=True)
        # Create feature extractor by removing the final fc layer
        model = torch.nn.Sequential(*list(full_model.children())[:-1])  # Remove final fc layer
        # Add global average pooling to flatten features
        model.add_module('flatten', torch.nn.Flatten())

        
    elif model_name == 'CLIP':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', 
            pretrained=os.path.join(ckpt_root, '2025_04_21-15_22_04-model_ViT-L-14-lr_0.0005-b_64-j_4-p_amp/checkpoints/epoch_40.pt')
        )
    elif model_name == 'SigLIP':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-16-SigLIP-256', 
            pretrained=os.path.join(ckpt_root, '2025_04_21-19_34_33-model_ViT-L-16-SigLIP-256-lr_0.0005-b_64-j_4-p_amp/checkpoints/epoch_40.pt')
        )
    elif model_name == 'SigLIP2-ViT-B':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:timm/ViT-B-16-SigLIP2-256', 
            pretrained=os.path.join(ckpt_root, '2025_04_21-12_25_13-model_hf-hub:timm-ViT-B-16-SigLIP2-256-lr_0.0005-b_64-j_4-p_amp/checkpoints/epoch_40.pt')
        )
    elif model_name == 'SigLIP2-ViT-L':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:timm/ViT-L-16-SigLIP2-256', 
            pretrained=os.path.join(ckpt_root, '2025_04_20-21_05_53-model_hf-hub:timm-ViT-L-16-SigLIP2-256-lr_0.0005-b_64-j_4-p_amp/checkpoints/epoch_40.pt')
        )
    elif model_name == 'SigLIP2-ViT-SO400M':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:timm/ViT-SO400M-14-SigLIP2', 
            pretrained=os.path.join(ckpt_root, '2025_04_17-11_32_17-model_hf-hub:timm-ViT-SO400M-14-SigLIP2-lr_0.0005-b_64-j_4-p_amp/checkpoints/epoch_40.pt')
        )
    
    return model, preprocess
