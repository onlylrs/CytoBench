import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    retinanet_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


class DetectionModel(nn.Module):
    """
    Wrapper for detection models with different architectures
    """
    
    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Args:
            model_name (str): Name of the detection model
            num_classes (int): Number of classes (including background)
            pretrained (bool): Whether to use pretrained weights
        """
        super(DetectionModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Build the model based on the specified architecture
        if model_name == 'fasterrcnn_resnet50_fpn':
            self.model = self._build_fasterrcnn_resnet50_fpn(pretrained)
        elif model_name == 'fasterrcnn_mobilenet_v3_large_fpn':
            self.model = self._build_fasterrcnn_mobilenet_v3_large_fpn(pretrained)
        elif model_name == 'fasterrcnn_mobilenet_v3_large_320_fpn':
            self.model = self._build_fasterrcnn_mobilenet_v3_large_320_fpn(pretrained)
        elif model_name == 'retinanet_resnet50_fpn':
            self.model = self._build_retinanet_resnet50_fpn(pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _build_fasterrcnn_resnet50_fpn(self, pretrained):
        """Build Faster R-CNN with ResNet-50 FPN backbone"""
        if pretrained:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        else:
            model = fasterrcnn_resnet50_fpn(weights=None)
        
        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        return model
    
    def _build_fasterrcnn_mobilenet_v3_large_fpn(self, pretrained):
        """Build Faster R-CNN with MobileNetV3-Large FPN backbone"""
        if pretrained:
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        else:
            model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        
        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        return model
    
    def _build_fasterrcnn_mobilenet_v3_large_320_fpn(self, pretrained):
        """Build Faster R-CNN with MobileNetV3-Large 320 FPN backbone"""
        if pretrained:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
        else:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
        
        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        return model
    
    def _build_retinanet_resnet50_fpn(self, pretrained):
        """Build RetinaNet with ResNet-50 FPN backbone"""
        if pretrained:
            model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
        else:
            model = retinanet_resnet50_fpn(weights=None)
        
        # Replace the classification head
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )
        
        return model
    

    
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Args:
            images (list[Tensor]): Images to be processed
            targets (list[dict], optional): Ground truth boxes and labels
            
        Returns:
            During training: dict with losses
            During inference: list[dict] with predictions
        """
        return self.model(images, targets)
    
    def train(self, mode=True):
        """Set training mode"""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        self.model.eval()
        return self


def build_detection_model(model_name, num_classes, pretrained=True):
    """
    Build detection model
    
    Args:
        model_name (str): Name of the detection model
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        DetectionModel: The detection model
    """
    # Add 1 for background class if not already included
    if model_name.startswith('retinanet'):
        # RetinaNet doesn't have background class
        total_classes = num_classes
    else:
        # Faster R-CNN has background class
        total_classes = num_classes + 1
    
    return DetectionModel(model_name, total_classes, pretrained)


# Available detection models
AVAILABLE_MODELS = [
    'fasterrcnn_resnet50_fpn',
    'fasterrcnn_mobilenet_v3_large_fpn',
    'fasterrcnn_mobilenet_v3_large_320_fpn',
    'retinanet_resnet50_fpn'
]


def get_available_models():
    """Get list of available detection models"""
    return AVAILABLE_MODELS 