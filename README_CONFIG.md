# Configuration System for Linear Probing

This document describes the new configuration system that allows you to organize and manage parameters for linear probing experiments using YAML configuration files.

## Overview

The configuration system provides:
- **Organized parameter management**: Parameters are grouped by functionality (model, dataset, training, evaluation)
- **Easy experiment configuration**: Use YAML files to define different experimental setups
- **Backward compatibility**: All original command-line arguments are still supported
- **Convenient training script**: Use `./scripts/train.sh` to run experiments with configuration files

## Directory Structure

```
cytofm/
├── config/
│   ├── bash_config.yaml           # Default configuration file
│   └── example_custom_config.yaml # Example custom configuration
├── scripts/
│   └── train.sh                   # Training script that uses configuration files
├── main.py                        # Main training script (now supports --config option)
└── README_CONFIG.md               # This documentation
```

## Configuration File Format

Configuration files are organized into functional sections:

### Model Configuration
```yaml
model_config:
  models:                          # List of models to evaluate
    - "ResNet50"
    - "ViT-L"
    - "CLIP"
  gpu: 7                          # GPU ID to use
```

### Dataset Configuration
```yaml
dataset_config:
  datasets:                       # List of datasets to evaluate
    - "Herlev"
    - "HiCervix"
  batch_size: 64                  # Batch size for training/evaluation
  cache_data: false               # Cache images in memory
  num_workers: null               # Number of data loading workers
  disable_progress_bar: false     # Disable progress bars
```

### Training Configuration
```yaml
training_config:
  epochs: 50                      # Number of training epochs
  lr: 0.001                       # Learning rate
```

### Evaluation Configuration
```yaml
evaluation_config:
  compute_ci: true                # Compute confidence intervals
  n_bootstraps: 1000              # Number of bootstrap samples
  metrics:                        # Metrics to compute
    - "accuracy"
    - "auc"
    - "macro_f1"
  force_recompute: false          # Force recomputation of existing results
  kfold: false                    # Use k-fold cross-validation
  k: 3                           # Number of folds
```

## Usage

### Method 1: Using the Training Script (Recommended)

The easiest way to run experiments is using the training script:

```bash
# Use default configuration
./scripts/train.sh

# Use custom configuration file
./scripts/train.sh -c config/example_custom_config.yaml

# Use configuration file with full path
./scripts/train.sh --config /path/to/your/config.yaml

# Show help
./scripts/train.sh --help
```

### Method 2: Direct Python Execution

You can also run the main script directly:

```bash
# Use configuration file
python main.py --config config/bash_config.yaml

# Use command-line arguments (original method)
python main.py --models CLIP SigLIP --datasets Herlev HiCervix --epochs 20
```

## Creating Custom Configurations

1. **Copy the default configuration**:
   ```bash
   cp config/bash_config.yaml config/my_experiment.yaml
   ```

2. **Edit the configuration** to match your experiment needs:
   - Modify model list to test specific models
   - Select subset of datasets
   - Adjust training parameters
   - Configure evaluation settings

3. **Run your experiment**:
   ```bash
   ./scripts/train.sh -c config/my_experiment.yaml
   ```

## Example Configurations

### Quick Testing Configuration
```yaml
model_config:
  models: ["CLIP"]
  gpu: 0

dataset_config:
  datasets: ["Herlev"]
  batch_size: 32

training_config:
  epochs: 10
  lr: 0.001

evaluation_config:
  compute_ci: false
  metrics: ["accuracy"]
```

### Full Evaluation Configuration
```yaml
model_config:
  models: ["ResNet50", "ViT-L", "CLIP", "SigLIP"]
  gpu: 7

dataset_config:
  datasets: ["Herlev", "HiCervix", "JinWooChoi", "FNAC2019"]
  batch_size: 64
  cache_data: true

training_config:
  epochs: 50
  lr: 0.001

evaluation_config:
  compute_ci: true
  n_bootstraps: 1000
  metrics: ["accuracy", "auc", "macro_f1"]
  kfold: true
  k: 5
```

## Benefits

1. **Organization**: Parameters are logically grouped and easy to understand
2. **Reproducibility**: Save exact configurations used for experiments
3. **Flexibility**: Easy to create variations for different experiments
4. **Version Control**: Configuration files can be tracked in git
5. **Documentation**: YAML format is self-documenting with comments

## Backward Compatibility

The original command-line interface is fully preserved. You can still use:

```bash
python main.py --models CLIP --datasets Herlev --epochs 50 --lr 0.001
```

The new `--config` option simply provides an alternative way to specify parameters.