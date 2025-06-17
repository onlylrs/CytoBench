# Cell Segmentation Implementation

This document describes the cell segmentation functionality added to the CytoBench framework.

## Overview

The cell segmentation implementation provides comprehensive support for instance segmentation of cells with:
- **5 segmentation models**: Mask R-CNN ResNet-50 FPN, Faster R-CNN variants, RetinaNet ResNet-50 FPN
- **Comprehensive metrics**: mAP, precision, recall, F1-score, IoU, Dice coefficient, AJI (Aggregated Jaccard Index)
- **Bootstrap confidence intervals**: Statistical significance testing with configurable sample sizes
- **COCO-style datasets**: Support for standard annotation formats with train/val/test splits

## Supported Models

### 1. Mask R-CNN ResNet-50 FPN
- **Name**: `maskrcnn_resnet50_fpn`
- **Description**: True instance segmentation model with dedicated mask prediction head
- **Best for**: High-quality segmentation masks with precise boundaries

### 2. Faster R-CNN ResNet-50 FPN (Adapted)
- **Name**: `fasterrcnn_resnet50_fpn`
- **Description**: Detection model adapted for segmentation by generating masks from bounding boxes
- **Best for**: When bounding box detection is sufficient or as baseline

### 3. Faster R-CNN MobileNetV3-Large FPN
- **Name**: `fasterrcnn_mobilenet_v3_large_fpn`
- **Description**: Lightweight detection model adapted for segmentation
- **Best for**: Resource-constrained environments

### 4. Faster R-CNN MobileNetV3-Large 320 FPN
- **Name**: `fasterrcnn_mobilenet_v3_large_320_fpn`
- **Description**: Compact version for smaller input sizes
- **Best for**: Fast inference on mobile devices

### 5. RetinaNet ResNet-50 FPN (Adapted)
- **Name**: `retinanet_resnet50_fpn`
- **Description**: Single-stage detector adapted for segmentation
- **Best for**: Balanced speed and accuracy

## Dataset Format

The implementation expects COCO-style datasets with the following structure:

```
dataset_root/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── annotations.json
├── val/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── annotations.json
└── test/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── annotations.json
```

### Annotation Format

Each `annotations.json` file should follow the COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 512,
      "height": 512
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "normal"
    },
    {
      "id": 2,
      "name": "abnormal"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]] or {"counts": "...", "size": [h, w]},
      "area": 1234,
      "iscrowd": 0
    }
  ]
}
```

## Evaluation Metrics

### Detection Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5-0.95
- **mAP@0.75**: Mean Average Precision at IoU threshold 0.75
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Segmentation-Specific Metrics
- **IoU (Intersection over Union)**: Overlap between predicted and ground truth masks
- **Dice Coefficient**: 2 * |A ∩ B| / (|A| + |B|)
- **AJI (Aggregated Jaccard Index)**: Instance-level segmentation quality metric

### Statistical Analysis
- **Bootstrap Confidence Intervals**: 95% confidence intervals computed using 1000 bootstrap samples
- **Macro/Weighted Averages**: Both unweighted and sample-size weighted metrics

## Usage

### Training

```bash
# Train Mask R-CNN with default configuration
./scripts/train.sh cell_seg

# Train with specific model configuration
./scripts/train.sh cell_seg configs/cell_seg/fasterrcnn_resnet50_fpn.yaml

# Train in background
./scripts/train.sh -b cell_seg configs/cell_seg/default.yaml
```

### Testing

```bash
# Test trained model
./scripts/test.sh cell_seg configs/cell_seg/default.yaml models/cell_seg/maskrcnn_resnet50_fpn_CRIC.pth

# Test in background
./scripts/test.sh -b cell_seg configs/cell_seg/default.yaml models/cell_seg/maskrcnn_resnet50_fpn_CRIC.pth
```

### Direct Python Usage

```python
# Training
python tools/train_seg.py --config configs/cell_seg/default.yaml

# Testing
python tools/test_seg.py --config configs/cell_seg/default.yaml --checkpoint models/cell_seg/maskrcnn_resnet50_fpn_CRIC.pth
```

## Configuration

### Basic Configuration (`configs/cell_seg/default.yaml`)

```yaml
# Common settings
common:
  seed: 42
  gpu: 0
  num_workers: 4
  disable_progress_bar: false

# Data configuration
data:
  root: './data'
  dataset: 'CRIC'
  organ: 'cervix'

# Model configuration
model:
  name: 'maskrcnn_resnet50_fpn'
  pretrained: true

# Training configuration
training:
  epochs: 50
  batch_size: 4
  lr: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  validation_frequency: 5
  
  lr_scheduler:
    type: 'StepLR'
    step_size: 16
    gamma: 0.1

# Evaluation configuration
evaluation:
  batch_size: 4
  iou_thresholds: [0.5]
  score_threshold: 0.5
  compute_ci: true
  n_bootstraps: 1000
  metrics:
    - 'precision'
    - 'recall'
    - 'f1'
    - 'map'
    - 'iou'
    - 'dice'
    - 'aji'

# Output configuration
output:
  model_dir: './models/cell_seg'
  results_dir: './results/cell_seg'
```

### Key Configuration Options

- **validation_frequency**: How often to run validation (default: every 5 epochs)
- **compute_ci**: Whether to compute bootstrap confidence intervals (default: true)
- **n_bootstraps**: Number of bootstrap samples for CI computation (default: 1000)
- **iou_thresholds**: IoU thresholds for mAP computation (default: [0.5])
- **score_threshold**: Confidence threshold for filtering predictions (default: 0.5)

## Output

### Training Output
- **Model checkpoints**: Saved to `models/cell_seg/`
- **Training logs**: Real-time progress with loss, validation metrics
- **Best model selection**: Based on validation mAP

### Evaluation Results
Results are saved in a comprehensive format:

```
================================================================================
SEGMENTATION METRICS
================================================================================

mAP@0.5: 71.32% (95% CI: 67.55% - 72.85%)
mAP@0.5:0.95: 33.27%
mAP@0.75: 28.33%

Macro Precision: 46.67% (95% CI: 44.53% - 47.78%)
Macro Recall: 48.13% (95% CI: 46.10% - 49.31%)
Macro F1 Score: 47.39% (95% CI: 45.45% - 48.38%)

Weighted Precision: 71.74% (95% CI: 69.19% - 74.76%)
Weighted Recall: 74.13% (95% CI: 71.01% - 76.08%)
Weighted F1 Score: 72.92% (95% CI: 70.29% - 75.23%)

Mean IoU: 65.42% (95% CI: 62.18% - 68.71%)
Mean Dice: 78.91% (95% CI: 75.33% - 82.15%)
AJI Score: 58.73% (95% CI: 55.21% - 62.44%)

Per-class Metrics:
--------------------------------------------------------------------------------------------------------
Class           Precision                 Recall                    F1 Score                  IoU       Dice      Support   
--------------------------------------------------------------------------------------------------------
normal          76.65% (95% CI: 74.17% - 80.91%) 79.60% (95% CI: 75.49% - 81.99%) 78.09% (95% CI: 74.99% - 80.53%) 68.42%    81.23%    1740      
abnormal        63.37% (95% CI: 55.52% - 62.98%) 64.80% (95% CI: 59.84% - 67.61%) 64.08% (95% CI: 59.73% - 64.83%) 62.41%    76.58%    1020      

Segmentation Statistics:
------------------------------------------------------------
Class           TP       FP       FN       Total GT   Total Pred  
------------------------------------------------------------
normal          1385     422      355      1740       1807        
abnormal        661      382      359      1020       1043        
```

## Implementation Details

### Architecture Components

1. **Dataset Loader** (`data/cell_seg/dataset.py`)
   - COCO-style annotation parsing
   - Mask loading and processing
   - Train/val/test split support

2. **Model Builder** (`model/cell_seg/segmentation_model.py`)
   - Unified interface for all models
   - Automatic mask generation for non-segmentation models
   - Pretrained weight loading

3. **Metrics Calculator** (`evaluation/cell_seg/metrics.py`)
   - Comprehensive segmentation metrics
   - Bootstrap confidence intervals
   - Per-class and aggregate statistics

4. **Training Script** (`tools/train_seg.py`)
   - End-to-end training pipeline
   - Validation and best model selection
   - Automatic test set evaluation

5. **Testing Script** (`tools/test_seg.py`)
   - Standalone model evaluation
   - Checkpoint loading and testing

### Key Features

- **Automatic Mask Generation**: Non-segmentation models automatically generate masks from bounding boxes
- **Robust Metrics**: Handles edge cases like empty predictions or no ground truth
- **Statistical Significance**: Bootstrap confidence intervals for all metrics
- **Flexible Configuration**: Easy model and hyperparameter switching
- **Integration**: Seamless integration with existing CytoBench framework

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use gradient accumulation
   - Try smaller models (MobileNet variants)

2. **Slow Bootstrap Evaluation**
   - Reduce `n_bootstraps` in configuration
   - Set `compute_ci: false` for faster evaluation
   - Use smaller validation sets

3. **Poor Segmentation Quality**
   - Use Mask R-CNN for true segmentation
   - Adjust IoU thresholds
   - Check annotation quality

4. **Training Instability**
   - Lower learning rate
   - Increase batch size
   - Use gradient clipping

### Performance Tips

- **Use Mask R-CNN** for best segmentation quality
- **Batch size 4-8** works well for most GPUs
- **Validation every 3-5 epochs** balances speed and monitoring
- **Bootstrap CI computation** can be disabled during development

## Integration with CytoBench

The segmentation implementation follows the same patterns as other CytoBench tasks:

- **Consistent API**: Same command-line interface and configuration format
- **Unified Logging**: Compatible with existing result aggregation
- **Modular Design**: Easy to extend and modify
- **Standard Metrics**: Comparable evaluation across tasks

This implementation provides a solid foundation for cell segmentation research while maintaining compatibility with the broader CytoBench ecosystem. 