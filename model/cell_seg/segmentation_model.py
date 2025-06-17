import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
import torchvision.transforms as T


class SegmentationModelWrapper(nn.Module):
    """
    Wrapper for segmentation models to ensure consistent output format
    """
    def __init__(self, base_model, model_type):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
    
    def forward(self, images, targets=None):
        if self.training:
            # During training, return losses
            return self.base_model(images, targets)
        else:
            # During inference, ensure mask output
            outputs = self.base_model(images)
            
            # For non-Mask R-CNN models, generate dummy masks from boxes
            if self.model_type != 'maskrcnn' and 'masks' not in outputs[0]:
                for output in outputs:
                    # Create dummy masks from bounding boxes
                    boxes = output['boxes']
                    if len(boxes) > 0:
                        # Get image size (assuming square images for simplicity)
                        # In practice, you might want to pass image size as parameter
                        img_h, img_w = 512, 512  # Default size, should be configurable
                        masks = torch.zeros((len(boxes), img_h, img_w), dtype=torch.uint8, device=boxes.device)
                        
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.int()
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img_w, x2), min(img_h, y2)
                            masks[i, y1:y2, x1:x2] = 1
                        
                        output['masks'] = masks
                    else:
                        # No detections
                        output['masks'] = torch.zeros((0, 512, 512), dtype=torch.uint8, device=boxes.device)
            
            return outputs


def build_segmentation_model(model_name, num_classes, pretrained=True):
    """
    Build segmentation model based on model name
    
    Args:
        model_name (str): Name of the model
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: Segmentation model
    """
    model_name = model_name.lower()
    
    if model_name == 'maskrcnn_resnet50_fpn':
        # Mask R-CNN with ResNet-50 FPN backbone
        model = maskrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Replace the classifier heads
        # Box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        return SegmentationModelWrapper(model, 'maskrcnn')
    
    elif model_name == 'fasterrcnn_resnet50_fpn':
        # Faster R-CNN with ResNet-50 FPN (adapted for segmentation)
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Replace the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        return SegmentationModelWrapper(model, 'fasterrcnn')
    
    elif model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
        # Faster R-CNN with MobileNetV3-Large FPN
        model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
        
        # Replace the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        return SegmentationModelWrapper(model, 'fasterrcnn')
    
    elif model_name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
        # Faster R-CNN with MobileNetV3-Large 320 FPN
        model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained)
        
        # Replace the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        return SegmentationModelWrapper(model, 'fasterrcnn')
    
    elif model_name == 'retinanet_resnet50_fpn':
        # RetinaNet with ResNet-50 FPN (adapted for segmentation)
        model = retinanet_resnet50_fpn(pretrained=pretrained)
        
        # Replace the classification head
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.num_classes = num_classes
        
        # Rebuild classification layers
        cls_logits = nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -torch.log(torch.tensor((1 - 0.01) / 0.01)))
        model.head.classification_head.cls_logits = cls_logits
        
        return SegmentationModelWrapper(model, 'retinanet')
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")


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
    # Test all supported models
    models = [
        'maskrcnn_resnet50_fpn',
        'fasterrcnn_resnet50_fpn',
        'fasterrcnn_mobilenet_v3_large_fpn',
        'fasterrcnn_mobilenet_v3_large_320_fpn',
        'retinanet_resnet50_fpn'
    ]
    
    num_classes = 3  # Example: background + 2 classes
    
    for model_name in models:
        print(f"\nTesting {model_name}...")
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
    
    print("\nModel building tests completed!") 