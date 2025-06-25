import os
import torch
import open_clip
from torchvision import transforms
from .resnet import ResNetFeatureExtractor
from .vit import ViTFeatureExtractor
from .dinov2_wrapper import build_dinov2_backbone

def get_standard_preprocess():
    """
    Get standard preprocessing function compatible with CLIP/SigLIP models
    This avoids loading a full CLIP model just to get the preprocessing function
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

def get_backbone_feature_dim(backbone_name):
    """
    Get the feature dimension for a given backbone model

    Args:
        backbone_name (str): Name of the backbone model

    Returns:
        int: Feature dimension of the backbone model
    """
    # Feature dimensions for different backbone models
    feature_dims = {
        # ResNet series
        'ResNet18': 512,
        'ResNet34': 512,
        'ResNet50': 2048,
        'ResNet101': 2048,
        'ResNet152': 2048,

        # Vision Transformer series
        'ViT-B-16': 768,
        'ViT-B-32': 768,
        'ViT-L-14': 1024,
        'ViT-L-16': 1024,
        'ViT-L-32': 1024,
        'ViT-H-14': 1280,

        # CLIP and SigLIP models (these may vary based on specific model)
        'CLIP': 768,  # Default for ViT-L-14
        'SigLIP': 768,  # Default for ViT-L-16-SigLIP-256
        'SigLIP2-ViT-B': 768,
        'SigLIP2-ViT-L': 1024,
        'SigLIP2-ViT-SO400M': 1152,  # SO400M specific dimension

        # DINOv2 models
        'dinov2_vitl': 1024,  # DINOv2 ViT-Large
        'gpfm': 1024,         # GPFM model
        'ccs': 1024,          # CCS model
    }

    if backbone_name not in feature_dims:
        raise ValueError(f"Unknown backbone '{backbone_name}'. Available backbones: {list(feature_dims.keys())}")

    return feature_dims[backbone_name]

def auto_set_feature_dim(config):
    """
    Automatically set feature_dim in config based on backbone name

    Args:
        config (dict): Configuration dictionary

    Returns:
        dict: Updated configuration with correct feature_dim
    """
    # Make a copy to avoid modifying the original config
    config = config.copy()

    # Get backbone name
    backbone_name = config['backbone']['name']

    # Get current feature_dim setting
    current_feature_dim = config.get('model', {}).get('feature_dim', 0)

    # If feature_dim is 0, 'auto', or None, set it automatically
    if current_feature_dim in [0, 'auto', None]:
        auto_feature_dim = get_backbone_feature_dim(backbone_name)

        # Ensure model section exists
        if 'model' not in config:
            config['model'] = {}

        config['model']['feature_dim'] = auto_feature_dim
        print(f"🔧 Auto-set feature_dim to {auto_feature_dim} for backbone '{backbone_name}'")
    else:
        # Verify that manually set feature_dim matches the backbone
        expected_feature_dim = get_backbone_feature_dim(backbone_name)
        if current_feature_dim != expected_feature_dim:
            print(f"⚠️  Warning: feature_dim ({current_feature_dim}) doesn't match expected dimension "
                  f"for '{backbone_name}' ({expected_feature_dim}). Using manual setting.")

    return config

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
        'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
        'ViT-B-16', 'ViT-B-32', 'ViT-L-14', 'ViT-L-16', 'ViT-L-32', 'ViT-H-14',
        'CLIP', 'SigLIP',
        'SigLIP2-ViT-B', 'SigLIP2-ViT-L', 'SigLIP2-ViT-SO400M',
        'dinov2_vitl', 'gpfm', 'ccs'
    ]
    
    if model_name not in available_models:
        raise ValueError(f"Model {model_name} not available. Choose from: {available_models}")
    
    # Build model based on name
    if model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
        # Use standard preprocessing function (same as CLIP/SigLIP)
        preprocess = get_standard_preprocess()

        # Build ResNet feature extractor using the ResNetFeatureExtractor class
        resnet_model_name = model_name.lower()  # Convert ResNet50 -> resnet50
        model = ResNetFeatureExtractor(
            pretrained=config['backbone']['pretrained'],
            model_name=resnet_model_name,
            freeze=config['backbone'].get('freeze', False)
        )

    elif model_name in ['ViT-B-16', 'ViT-B-32', 'ViT-L-14', 'ViT-L-16', 'ViT-L-32', 'ViT-H-14']:
        # Check if ViT-L-14 is requested but not available
        if model_name == 'ViT-L-14':
            import torchvision.models as models
            if not hasattr(models, 'vit_l_14'):
                try:
                    import timm
                    print("⚠️  ViT-L-14 not available in torchvision, using timm fallback")
                except ImportError:
                    raise ImportError(
                        f"ViT-L-14 is not available in your torchvision version ({torch.__version__}) "
                        "and timm is not installed. Please either:\n"
                        "1. Upgrade torchvision: pip install --upgrade torchvision\n"
                        "2. Install timm: pip install timm\n"
                        "3. Use ViT-L-16 instead: backbone.name='ViT-L-16'"
                    )
        # Use standard preprocessing function (same as CLIP/SigLIP)
        preprocess = get_standard_preprocess()

        # Build ViT feature extractor using the ViTFeatureExtractor class
        vit_model_name = model_name.lower().replace('-', '_')  # Convert ViT-B-16 -> vit_b_16
        model = ViTFeatureExtractor(
            pretrained=config['backbone']['pretrained'],
            model_name=vit_model_name,
            freeze=config['backbone'].get('freeze', False)
        )


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

    elif model_name in ['dinov2_vitl', 'gpfm', 'ccs']:
        # DINOv2 models
        device = torch.device(f"cuda:{config['common']['gpu']}" if torch.cuda.is_available() else "cpu")
        freeze = config['backbone'].get('freeze', True)

        # Build DINOv2 model using wrapper
        model, preprocess = build_dinov2_backbone(model_name, device, freeze)

        print(f"🔧 Built DINOv2 model '{model_name}' with embed_dim={model.get_feature_dim()}")

    return model, preprocess
