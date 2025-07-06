import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T


def freeze_backbone_parameters(model):
    """
    Freeze backbone parameters for linear probing

    Args:
        model: Mask R-CNN model
    """
    # Freeze backbone (ResNet50 + FPN)
    for param in model.backbone.parameters():
        param.requires_grad = False

    print("🔒 Frozen backbone parameters for linear probing")

    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"   Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")


def unfreeze_backbone_parameters(model):
    """
    Unfreeze backbone parameters for fine-tuning

    Args:
        model: Mask R-CNN model
    """
    # Unfreeze backbone (ResNet50 + FPN)
    for param in model.backbone.parameters():
        param.requires_grad = True

    print("🔓 Unfrozen backbone parameters for fine-tuning")

    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")





def build_segmentation_model(model_name, num_classes, pretrained=True, freeze_backbone=False):
    """
    Build segmentation model based on model name

    Args:
        model_name (str): Name of the model (only 'maskrcnn_resnet50_fpn' supported)
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone parameters for linear probing

    Returns:
        torch.nn.Module: Mask R-CNN segmentation model
    """
    model_name = model_name.lower()
    
    if model_name == 'maskrcnn_resnet50_fpn':
        # Mask R-CNN with ResNet-50 FPN backbone
        if pretrained:
            model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        else:
            model = maskrcnn_resnet50_fpn(weights=None)
        
        # Replace the classifier heads
        # Box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

        # Freeze backbone if requested
        if freeze_backbone:
            freeze_backbone_parameters(model)

        return model
    
    else:
        raise ValueError(f"Unsupported model: {model_name}. Only 'maskrcnn_resnet50_fpn' is supported for segmentation.")


def get_transform(train=False):
    """
    Get data transforms for segmentation
    
    Args:
        train (bool): Whether to apply training transforms
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    transforms = []
    transforms.append(T.ToTensor())
    
    if train:
        # Add training-specific transforms if needed
        # Note: Be careful with transforms that affect masks
        pass
    
    return T.Compose(transforms)


# Test the model builder
if __name__ == '__main__':
    # Test Mask R-CNN model
    model_name = 'maskrcnn_resnet50_fpn'
    num_classes = 3  # Example: background + 2 classes
    
    print("Testing Mask R-CNN segmentation model...")
    
    try:
        model = build_segmentation_model(model_name, num_classes, pretrained=False)
        print(f"✓ {model_name} built successfully")
        
        # Test forward pass
        model.eval()
        dummy_input = [torch.randn(3, 224, 224)]
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful, output keys: {list(output[0].keys())}")
        
    except Exception as e:
        print(f"✗ Error building {model_name}: {e}")
    
    print("Model building test completed!") 