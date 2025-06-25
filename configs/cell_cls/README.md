# Cell Classification Configuration Files

Simple and flexible configuration system for cell classification experiments.

## 📁 Configuration Files

- **`base_config.yaml`** - Main configuration with all options and comments
- **`cross_validation.yaml`** - Enable 5-fold cross validation
- **`quick_test.yaml`** - Fast settings for testing

## 🚀 Usage

### Basic Training
```bash
# Use base config directly
python tools/train.py --config configs/cell_cls/base_config.yaml

# Or modify parameters inline by editing the config file
# Change backbone.name to "ResNet18", "ViT-B-16", "CLIP", etc.
```

### Cross Validation
```bash
python tools/train_cv.py --config configs/cell_cls/cross_validation.yaml
```

### Quick Testing
```bash
python tools/train.py --config configs/cell_cls/quick_test.yaml
```

## ⚙️ Customization

Simply edit `base_config.yaml` or create a new config inheriting from it:

```yaml
# custom_experiment.yaml
_base_: "base_config.yaml"

# Change backbone
backbone:
  name: "ViT-B-16"  # or ResNet18, CLIP, etc.
  freeze: false     # for fine-tuning

# Adjust training
training:
  lr: 0.00001       # lower for fine-tuning
  epochs: 30

# Change dataset
data:
  dataset: "SIPaKMeD"
```

## 📝 Available Options

See `base_config.yaml` for all available parameters with inline comments showing options.
