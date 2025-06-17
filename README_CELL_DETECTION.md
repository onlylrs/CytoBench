# Cell Detection Implementation

This document describes the cell detection functionality that has been added to the CytoBench framework.

## Overview

The cell detection module supports training and evaluation of various detection models on COCO-style datasets for cell detection tasks. The implementation follows the same structure and patterns as the existing cell classification module.

## Supported Models

The following detection models are supported:

1. **Faster R-CNN ResNet-50 FPN** (`fasterrcnn_resnet50_fpn`)
2. **Faster R-CNN MobileNetV3-Large FPN** (`fasterrcnn_mobilenet_v3_large_fpn`)
3. **Faster R-CNN MobileNetV3-Large 320 FPN** (`fasterrcnn_mobilenet_v3_large_320_fpn`)
4. **RetinaNet ResNet-50 FPN** (`retinanet_resnet50_fpn`)
5. **Mask R-CNN ResNet-50 FPN** (`maskrcnn_resnet50_fpn`)

## Dataset Format

The cell detection module expects datasets in COCO format:

```
dataset_root/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── test/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── train.json
├── val.json
└── test.json
```

### Annotation Format

The JSON files should follow COCO annotation format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "cell_type_1"
    }
  ]
}
```

## Training

### Using the Training Script

```bash
# Train with default Faster R-CNN model
./scripts/train.sh cell_det

# Train with specific config
./scripts/train.sh cell_det configs/cell_det/retinanet.yaml

# Train in background
./scripts/train.sh -b cell_det configs/cell_det/default.yaml
```

### Direct Python Usage

```bash
python tools/train_det.py --config configs/cell_det/default.yaml
```

### Configuration

The configuration files are located in `configs/cell_det/`:

- `default.yaml`: Default configuration with Faster R-CNN ResNet-50 FPN
- `retinanet.yaml`: Configuration for RetinaNet

Key configuration parameters:

```yaml
model:
  name: 'fasterrcnn_resnet50_fpn'  # Model architecture
  pretrained: true                  # Use pretrained weights

training:
  epochs: 50
  batch_size: 4                    # Smaller batch size for detection
  lr: 0.001
  weight_decay: 0.0005
  momentum: 0.9
  score_threshold: 0.5             # Score threshold during training
  nms_threshold: 0.5               # NMS threshold

evaluation:
  batch_size: 2
  compute_ci: true                 # Compute confidence intervals
  n_bootstraps: 1000
  iou_thresholds: [0.5, 0.75]     # IoU thresholds for evaluation
  score_threshold: 0.5             # Score threshold for evaluation
```

## Testing

### Using the Testing Script

```bash
# Test a trained model
./scripts/test.sh cell_det configs/cell_det/default.yaml checkpoints/cell_det/fasterrcnn_resnet50_fpn_dataset.pth
```

### Direct Python Usage

```bash
python tools/test_det.py --config configs/cell_det/default.yaml --checkpoint checkpoints/cell_det/model.pth
```

## Metrics

The detection module computes comprehensive metrics with confidence intervals:

### Primary Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- **mAP@0.75**: Mean Average Precision at IoU threshold 0.75

### Per-Class Metrics
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Aggregate Metrics
- **Macro averages**: Unweighted average across classes
- **Weighted averages**: Weighted by class support (number of ground truth instances)

### Confidence Intervals
All metrics include 95% confidence intervals computed using bootstrap sampling (1000 samples by default).

## Output Files

The training and testing scripts generate several output files:

### Model Checkpoints
- Saved to `checkpoints/cell_det/`
- Best model based on validation mAP is saved

### Results
- Detailed metrics saved to `results/cell_det/`
- Includes formatted text files with all metrics
- CSV and LaTeX tables for easy reporting

### Logs
- Training logs saved to `results/cell_det_[timestamp].log`
- Include training progress, validation metrics, and final test results

## Example Output

```
================================================================================
DETECTION METRICS
================================================================================

mAP@0.5: 65.42% (95% CI: 61.23% - 69.87%)
mAP@0.5:0.95: 45.78%
mAP@0.75: 52.34%

Macro Precision: 68.91% (95% CI: 64.12% - 73.45%)
Macro Recall: 62.87% (95% CI: 58.23% - 67.91%)
Macro F1 Score: 65.73% (95% CI: 60.98% - 70.12%)

Per-class Metrics:
------------------------------------------------------------------------------------------------------------------------
Class                Precision                      Recall                         F1 Score                       Support   
------------------------------------------------------------------------------------------------------------------------
cell_type_1          72.45% (95% CI: 67.23%-77.89%) 65.78% (95% CI: 60.12%-71.34%) 68.97% (95% CI: 63.45%-74.23%) 156       
cell_type_2          65.37% (95% CI: 59.84%-70.92%) 59.96% (95% CI: 54.67%-65.23%) 62.49% (95% CI: 57.51%-67.45%) 142       

Detection Statistics:
------------------------------------------------------------
Class                TP       FP       FN       Total GT   Total Pred  
------------------------------------------------------------
cell_type_1          103      39       53       156        142         
cell_type_2          85       45       57       142        130         
```

## Implementation Details

### Key Components

1. **Dataset Class** (`data/cell_det/dataset.py`):
   - `CellDetDataset`: Handles COCO-style annotations
   - Converts COCO format to PyTorch detection format
   - Supports train/val/test splits

2. **Model Wrapper** (`model/cell_det/detection_model.py`):
   - `DetectionModel`: Unified interface for all detection models
   - Handles model-specific configurations
   - Supports all major detection architectures

3. **Metrics Module** (`evaluation/cell_det/metrics.py`):
   - Comprehensive detection metrics with confidence intervals
   - IoU-based matching between predictions and ground truth
   - COCO-style mAP computation using pycocotools

4. **Training Script** (`tools/train_det.py`):
   - Full training pipeline with validation
   - Learning rate scheduling
   - Best model checkpointing based on validation mAP

5. **Testing Script** (`tools/test_det.py`):
   - Model evaluation on test set
   - Comprehensive metrics computation
   - Results formatting and saving

### Future Extensions

The implementation is designed to support future enhancements:

- **K-fold cross-validation**: Framework ready for k-fold support
- **Additional models**: Easy to add new detection architectures
- **Custom metrics**: Extensible metrics framework
- **Visualization**: Can be extended with detection visualization tools

## Dependencies

The cell detection module requires:

- PyTorch >= 1.8
- torchvision >= 0.9
- pycocotools
- numpy
- tqdm
- PIL (Pillow)

Install with:
```bash
pip install torch torchvision pycocotools numpy tqdm Pillow
``` 