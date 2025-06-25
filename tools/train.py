import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backbone.build_model import build_backbone, auto_set_feature_dim
from model.cell_cls.linear_probe import LinearProbeModel
from data.cell_cls.dataset import CellClsDataset
from evaluation.cell_cls.metrics import compute_classification_metrics

def apply_config_overrides(config, overrides):
    """
    Apply configuration overrides to the config dictionary

    Args:
        config: Configuration dictionary
        overrides: List of override strings in format "key=value"

    Returns:
        config: Updated configuration dictionary
    """
    for override in overrides:
        if '=' not in override:
            print(f"⚠️  Invalid override format: {override} (should be key=value)")
            continue

        key, value = override.split('=', 1)

        # Parse nested keys (e.g., "common.gpu" -> ["common", "gpu"])
        keys = key.split('.')

        # Navigate to the nested dictionary
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Convert value to appropriate type
        final_key = keys[-1]

        # Try to convert to appropriate type
        if value.lower() in ['true', 'false']:
            current[final_key] = value.lower() == 'true'
        elif value.isdigit():
            current[final_key] = int(value)
        elif value.replace('.', '').isdigit():
            current[final_key] = float(value)
        else:
            current[final_key] = value

        print(f"🔧 Override applied: {key} = {current[final_key]}")

    return config

def generate_experiment_name(backbone_name, dataset_name, timestamp=None):
    """
    Generate consistent experiment name for all files (checkpoints, results, logs)

    Args:
        backbone_name: Name of the backbone model
        dataset_name: Name of the dataset
        timestamp: Optional timestamp string, if None will generate current time

    Returns:
        experiment_name: Formatted name string
    """
    import datetime
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clean backbone name (remove special characters)
    clean_backbone = backbone_name.replace('-', '_').replace('/', '_')

    return f"{clean_backbone}_{dataset_name}_{timestamp}"

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, default='./configs/cell_cls/base_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--override', action='append', default=[],
                       help='Override config values (format: key=value, can be used multiple times)')
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from YAML file with inheritance support
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        config: Merged configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle base configuration inheritance
    if '_base_' in config:
        base_path = config.pop('_base_')
        # Handle relative paths
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        
        # Check if base path exists
        if not os.path.exists(base_path):
            print(f"Warning: Base configuration file not found: {base_path}")
            print(f"Using only the current configuration file: {config_path}")
            return config
            
        # Load base configuration
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configurations (base config is overridden by specific config)
        merged_config = deep_merge(base_config, config)
        return merged_config
    
    return config

def deep_merge(base_dict, override_dict):
    """
    Recursively merge two dictionaries, with values from override_dict taking precedence
    
    Args:
        base_dict: Base dictionary
        override_dict: Dictionary with overrides
        
    Returns:
        merged: Merged dictionary
    """
    merged = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
            
    return merged

def print_config(config):
    """Print all configuration parameters in a formatted way"""
    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")

    # Common settings
    print("📋 Common Settings:")
    common = config.get('common', {})
    print(f"  • Seed: {common.get('seed', 'Not set')}")
    print(f"  • GPU: {common.get('gpu', 'Not set')}")
    print(f"  • Num workers: {common.get('num_workers', 'Not set')}")
    print(f"  • Disable progress bar: {common.get('disable_progress_bar', False)}")

    # Data settings
    print("\n📊 Data Settings:")
    data = config.get('data', {})
    print(f"  • Dataset: {data.get('dataset', 'Not set')}")

    # Model settings
    print("\n🧠 Model Settings:")
    model = config.get('model', {})
    print(f"  • Feature dimension: {model.get('feature_dim', 'Not set')}")
    print(f"  • Dropout probability: {model.get('dropout_p', 'Not set')}")

    # Backbone settings
    print("\n🏗️  Backbone Settings:")
    backbone = config.get('backbone', {})
    print(f"  • Name: {backbone.get('name', 'Not set')}")
    print(f"  • Pretrained: {backbone.get('pretrained', 'Not set')}")
    print(f"  • Freeze: {backbone.get('freeze', 'Not set')}")

    # Training settings
    print("\n🚀 Training Settings:")
    training = config.get('training', {})
    print(f"  • Batch size: {training.get('batch_size', 'Not set')}")
    print(f"  • Epochs: {training.get('epochs', 'Not set')}")
    print(f"  • Learning rate: {training.get('lr', 'Not set')}")
    print(f"  • Weight decay: {training.get('weight_decay', 'Not set')}")
    print(f"  • Optimizer: {training.get('optimizer', 'Not set')}")
    print(f"  • Scheduler: {training.get('scheduler', 'Not set')}")

    # Evaluation settings
    print("\n📈 Evaluation Settings:")
    evaluation = config.get('evaluation', {})
    print(f"  • Batch size: {evaluation.get('batch_size', 'Not set')}")
    print(f"  • Compute CI: {evaluation.get('compute_ci', 'Not set')}")
    print(f"  • Bootstrap samples: {evaluation.get('n_bootstraps', 'Not set')}")
    print(f"  • Cross validation enabled: {evaluation.get('cv_enabled', False)}")
    if evaluation.get('cv_enabled', False):
        print(f"  • CV folds: {evaluation.get('cv_folds', 'Not set')}")
        print(f"  • CV seed: {evaluation.get('cv_seed', 'Not set')}")
    print(f"  • Metrics: {evaluation.get('metrics', 'Not set')}")

    # Output settings
    print("\n💾 Output Settings:")
    output = config.get('output', {})
    print(f"  • Model directory: {output.get('model_dir', 'Not set')}")
    print(f"  • Results directory: {output.get('results_dir', 'Not set')}")

    print(f"{'='*80}\n")

def train(config):
    # Check if cross validation is enabled
    if config.get('evaluation', {}).get('cv_enabled', False):
        print("Cross validation is enabled. Please use train_cv.py for cross validation experiments.")
        print("Example: python tools/train_cv.py --config configs/cell_cls/cross_validation.yaml")
        return

    # Auto-set feature_dim based on backbone name
    config = auto_set_feature_dim(config)

    # Print all configuration parameters
    print_config(config)

    # Set device
    device = torch.device(f"cuda:{config['common']['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate experiment name for consistent file naming
    dataset_name = config['data']['dataset']
    backbone_name = config['backbone']['name']
    experiment_name = generate_experiment_name(backbone_name, dataset_name)
    print(f"📝 Experiment name: {experiment_name}")
    print(f"   Backbone: {backbone_name}")
    print(f"   Dataset: {dataset_name}")

    # Set environment variable for shell scripts to use the same naming
    import os
    os.environ['EXPERIMENT_NAME'] = experiment_name

    # Build backbone model
    backbone, preprocess = build_backbone(config)

    # Build datasets

    # Get root directory if provided in config (fallback)
    dataset_root = None
    if 'root' in config.get('data', {}):
        dataset_root = os.path.join(config['data']['root'], dataset_name)

    train_dataset = CellClsDataset(
        dataset_name=dataset_name,
        preprocess=preprocess,
        split='train',
        root=dataset_root
    )

    val_dataset = CellClsDataset(
        dataset_name=dataset_name,
        preprocess=preprocess,
        split='val',
        root=dataset_root
    )

    test_dataset = CellClsDataset(
        dataset_name=dataset_name,
        preprocess=preprocess,
        split='test',
        root=dataset_root
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['common']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['common']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['common']['num_workers']
    )
    
    # Build model
    num_classes = len(train_dataset.label_dict)
    feature_dim = config['model'].get('feature_dim', 0)  # Use 0 as default to auto-detect
    freeze_backbone = config['backbone'].get('freeze', False)
    # print("test:",freeze_backbone)
    dropout_p = config['model'].get('dropout_p', 0.5)  # Default dropout probability
    model = LinearProbeModel(backbone, num_classes, feature_dim, freeze_backbone, dropout_p)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # Configure optimizer parameters based on freeze setting
    if freeze_backbone:
        # Linear probing: only optimize classifier parameters
        optimizer_params = list(model.classifier.parameters())
        print("🔒 Linear probing mode: Only training classifier parameters")
    else:
        # Fine-tuning: optimize all model parameters
        optimizer_params = list(model.parameters())
        print("🔓 Fine-tuning mode: Training all model parameters")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in optimizer_params if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Verify we have parameters to optimize
    if len(optimizer_params) == 0:
        raise ValueError("No parameters to optimize! Check model configuration.")

    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0

    print(f"\n{'='*80}")
    print(f"Starting training for {config['training']['epochs']} epochs...")
    print(f"Target: Find best validation accuracy and save the corresponding model")
    print(f"{'='*80}")

    for epoch in range(config['training']['epochs']):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            print(f'🎯 New best validation accuracy: {best_val_acc:.2f}% (improved by {improvement:.2f}%) at epoch {best_epoch}')
        else:
            epochs_since_best = (epoch + 1) - best_epoch
            print(f'📊 Current best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch} ({epochs_since_best} epochs ago)')
    
    # Save final model
    print(f"\n{'='*80}")
    print("Training completed! Saving best model...")

    save_dir = os.path.join(config['output']['model_dir'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{experiment_name}.pth")

    # If we have a best model state, use that instead of the final state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Using best model from epoch {best_epoch} with validation accuracy: {best_val_acc:.2f}%")
    else:
        print(f"⚠️  No improvement found, using final model state")

    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}")
    print(f"{'='*80}")
    
    # Test the model
    print("\n" + "="*80)
    print("Evaluating model on test set...")
    
    # Get evaluation parameters from config
    compute_ci = config['evaluation'].get('compute_ci', True)
    n_bootstraps = config['evaluation'].get('n_bootstraps', 1000)
    
    print(f"Computing bootstrap confidence intervals: {compute_ci}")
    if compute_ci:
        print(f"Number of bootstrap samples: {n_bootstraps}")
    print()
    
    # Collect predictions and targets for metrics computation
    all_predictions = []
    all_targets = []
    all_probabilities = []
    test_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            try:
                images = images.to(device)
                labels = labels.squeeze().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Store predictions, targets and probabilities
                all_predictions.append(preds.cpu())
                all_targets.append(labels.cpu())
                all_probabilities.append(probs.cpu())
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory during testing, skipping batch')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_probabilities = torch.cat(all_probabilities)
    
    # Compute average loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Get class names from label dictionary
    label_names = {v: k for k, v in train_dataset.label_dict.items()}
    class_names = [label_names.get(i, f"Class {i}") for i in range(len(train_dataset.label_dict))]
    
    # Compute metrics using the updated function
    metrics = compute_classification_metrics(
        all_targets, 
        all_predictions, 
        all_probabilities,
        class_names=class_names,
        compute_ci=compute_ci,
        n_bootstraps=n_bootstraps
    )
    
    # Add loss to metrics
    metrics['loss'] = avg_test_loss
    
    # Format the results similar to template.log
    formatted_results = "="*80 + "\n"
    formatted_results += "CLASSIFICATION METRICS\n"
    formatted_results += "="*80 + "\n\n"
    
    # Overall metrics with confidence intervals - ordered as requested
    # 1. Accuracy
    if compute_ci and 'accuracy_ci' in metrics:
        acc_lower, acc_upper = metrics['accuracy_ci']
        formatted_results += f"Overall Accuracy: {metrics['accuracy']:.2f}% (95% CI: {acc_lower:.2f}% - {acc_upper:.2f}%)\n"
    else:
        formatted_results += f"Overall Accuracy: {metrics['accuracy']:.2f}%\n"

    # 2. AUC
    if 'auc' in metrics:
        if compute_ci and 'auc_ci' in metrics:
            auc_lower, auc_upper = metrics['auc_ci']
            formatted_results += f"AUC: {metrics['auc']:.2f}% (95% CI: {auc_lower:.2f}% - {auc_upper:.2f}%)\n"
        else:
            formatted_results += f"AUC: {metrics['auc']:.2f}%\n"

    # 3. Weighted Sensitivity
    if 'weighted_sensitivity' in metrics:
        if compute_ci and 'weighted_sensitivity_ci' in metrics:
            ws_lower, ws_upper = metrics['weighted_sensitivity_ci']
            formatted_results += f"Weighted Sensitivity: {metrics['weighted_sensitivity']:.2f}% (95% CI: {ws_lower:.2f}% - {ws_upper:.2f}%)\n"
        else:
            formatted_results += f"Weighted Sensitivity: {metrics['weighted_sensitivity']:.2f}%\n"

    # 4. Weighted Specificity
    if 'weighted_specificity' in metrics:
        if compute_ci and 'weighted_specificity_ci' in metrics:
            wsp_lower, wsp_upper = metrics['weighted_specificity_ci']
            formatted_results += f"Weighted Specificity: {metrics['weighted_specificity']:.2f}% (95% CI: {wsp_lower:.2f}% - {wsp_upper:.2f}%)\n"
        else:
            formatted_results += f"Weighted Specificity: {metrics['weighted_specificity']:.2f}%\n"

    # 5. Weighted F1
    if 'weighted_f1' in metrics:
        if compute_ci and 'weighted_f1_ci' in metrics:
            wf_lower, wf_upper = metrics['weighted_f1_ci']
            formatted_results += f"Weighted F1 Score: {metrics['weighted_f1']:.2f}% (95% CI: {wf_lower:.2f}% - {wf_upper:.2f}%)\n"
        else:
            formatted_results += f"Weighted F1 Score: {metrics['weighted_f1']:.2f}%\n"

    # Other metrics
    if 'macro_precision' in metrics:
        if compute_ci and 'macro_precision_ci' in metrics:
            mp_lower, mp_upper = metrics['macro_precision_ci']
            formatted_results += f"Macro Precision: {metrics['macro_precision']:.2f}% (95% CI: {mp_lower:.2f}% - {mp_upper:.2f}%)\n"
        else:
            formatted_results += f"Macro Precision: {metrics['macro_precision']:.2f}%\n"

    if 'macro_recall' in metrics:
        if compute_ci and 'macro_recall_ci' in metrics:
            mr_lower, mr_upper = metrics['macro_recall_ci']
            formatted_results += f"Macro Recall: {metrics['macro_recall']:.2f}% (95% CI: {mr_lower:.2f}% - {mr_upper:.2f}%)\n"
        else:
            formatted_results += f"Macro Recall: {metrics['macro_recall']:.2f}%\n"

    if 'macro_f1' in metrics:
        if compute_ci and 'macro_f1_ci' in metrics:
            mf_lower, mf_upper = metrics['macro_f1_ci']
            formatted_results += f"Macro F1 Score: {metrics['macro_f1']:.2f}% (95% CI: {mf_lower:.2f}% - {mf_upper:.2f}%)\n"
        else:
            formatted_results += f"Macro F1 Score: {metrics['macro_f1']:.2f}%\n"

    if 'weighted_precision' in metrics:
        if compute_ci and 'weighted_precision_ci' in metrics:
            wp_lower, wp_upper = metrics['weighted_precision_ci']
            formatted_results += f"Weighted Precision: {metrics['weighted_precision']:.2f}% (95% CI: {wp_lower:.2f}% - {wp_upper:.2f}%)\n"
        else:
            formatted_results += f"Weighted Precision: {metrics['weighted_precision']:.2f}%\n"

    if 'weighted_recall' in metrics:
        if compute_ci and 'weighted_recall_ci' in metrics:
            wr_lower, wr_upper = metrics['weighted_recall_ci']
            formatted_results += f"Weighted Recall: {metrics['weighted_recall']:.2f}% (95% CI: {wr_lower:.2f}% - {wr_upper:.2f}%)\n"
        else:
            formatted_results += f"Weighted Recall: {metrics['weighted_recall']:.2f}%\n"

    if 'macro_sensitivity' in metrics:
        if compute_ci and 'macro_sensitivity_ci' in metrics:
            ms_lower, ms_upper = metrics['macro_sensitivity_ci']
            formatted_results += f"Macro Sensitivity: {metrics['macro_sensitivity']:.2f}% (95% CI: {ms_lower:.2f}% - {ms_upper:.2f}%)\n"
        else:
            formatted_results += f"Macro Sensitivity: {metrics['macro_sensitivity']:.2f}%\n"

    if 'macro_specificity' in metrics:
        if compute_ci and 'macro_specificity_ci' in metrics:
            msp_lower, msp_upper = metrics['macro_specificity_ci']
            formatted_results += f"Macro Specificity: {metrics['macro_specificity']:.2f}% (95% CI: {msp_lower:.2f}% - {msp_upper:.2f}%)\n"
        else:
            formatted_results += f"Macro Specificity: {metrics['macro_specificity']:.2f}%\n"
    
    # Per-class metrics
    formatted_results += "\nPer-class Metrics:\n"
    formatted_results += "-" * 180 + "\n"
    formatted_results += f"{'Class':<20} {'Accuracy':<25} {'AUC':<25} {'Sensitivity':<25} {'Specificity':<25} {'F1 Score':<25} {'Support':<10}\n"
    formatted_results += "-" * 180 + "\n"
    
    for i, class_name in enumerate(class_names):
        # 1. Format accuracy with CI
        if compute_ci and 'per_class_accuracy_ci' in metrics:
            acc_lower, acc_upper = metrics['per_class_accuracy_ci'][i]
            acc_str = f"{metrics['per_class_accuracy'][i]:.2f}% ({acc_lower:.1f}-{acc_upper:.1f}%)"
        else:
            acc_str = f"{metrics['per_class_accuracy'][i]:.2f}%"

        # 2. Format AUC with CI
        if 'auc_per_class' in metrics and i < len(metrics['auc_per_class']):
            auc = metrics['auc_per_class'][i]
            if compute_ci and 'auc_per_class_ci' in metrics and i < len(metrics['auc_per_class_ci']):
                auc_lower, auc_upper = metrics['auc_per_class_ci'][i]
                auc_str = f"{auc:.2f}% ({auc_lower:.1f}-{auc_upper:.1f}%)"
            else:
                auc_str = f"{auc:.2f}%"
        else:
            auc_str = "N/A"

        # 3. Format Sensitivity with CI
        if 'sensitivity' in metrics and i < len(metrics['sensitivity']):
            sens = metrics['sensitivity'][i]
            if compute_ci and 'sensitivity_ci' in metrics and i < len(metrics['sensitivity_ci']):
                sens_lower, sens_upper = metrics['sensitivity_ci'][i]
                sens_str = f"{sens:.2f}% ({sens_lower:.1f}-{sens_upper:.1f}%)"
            else:
                sens_str = f"{sens:.2f}%"
        else:
            sens_str = "N/A"

        # 4. Format Specificity with CI
        if 'specificity' in metrics and i < len(metrics['specificity']):
            spec = metrics['specificity'][i]
            if compute_ci and 'specificity_ci' in metrics and i < len(metrics['specificity_ci']):
                spec_lower, spec_upper = metrics['specificity_ci'][i]
                spec_str = f"{spec:.2f}% ({spec_lower:.1f}-{spec_upper:.1f}%)"
            else:
                spec_str = f"{spec:.2f}%"
        else:
            spec_str = "N/A"

        # 5. Format F1 with CI
        f1 = metrics['f1'][i]
        if compute_ci and 'f1_ci' in metrics and i < len(metrics['f1_ci']):
            f1_lower, f1_upper = metrics['f1_ci'][i]
            f1_str = f"{f1:.2f}% ({f1_lower:.1f}-{f1_upper:.1f}%)"
        else:
            f1_str = f"{f1:.2f}%"

        # 6. Support
        support = metrics['support'][i]
        support_str = f"{support}"

        formatted_results += f"{class_name:<20} {acc_str:<25} {auc_str:<25} {sens_str:<25} {spec_str:<25} {f1_str:<25} {support_str:<10}\n"

    
    # Confusion matrix
    formatted_results += "\nConfusion Matrix:\n"
    cm = metrics['confusion_matrix']
    
    # Format confusion matrix header
    header = "      " + " ".join(f"{i:5d}" for i in range(len(class_names)))
    formatted_results += header + "\n"
    formatted_results += "-" * len(header) + "\n"
    
    # Format confusion matrix rows
    for i, row in enumerate(cm):
        formatted_results += f"{i:5d} " + " ".join(f"{cell:5d}" for cell in row) + "\n"
    
    # Print formatted results
    print(formatted_results)
    
    # Save results to file
    results_dir = os.path.join(config['output'].get('results_dir', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    # Add suffix for confidence intervals
    ci_suffix = "_with_ci" if compute_ci else ""
    results_path = os.path.join(results_dir, f"{experiment_name}{ci_suffix}_metrics.txt")
    
    with open(results_path, 'w') as f:
        f.write(formatted_results)
    
    print(f"Results saved to {results_path}")
    
    # Store formatted results in metrics dictionary
    metrics['formatted_results'] = formatted_results
    
    return {
        'model': model,
        'metrics': metrics
    }

def train_epoch(model, data_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Calculate average loss and accuracy
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, data_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:  # 使用传入的data_loader参数
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += loss.item()
    
    return val_loss / len(data_loader), 100 * correct / total

def test(model, data_loader, criterion, device, label_dict):
    """Test the model and compute metrics
    
    Args:
        model: The model to test
        data_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for testing
        label_dict: Dictionary mapping label names to indices
        
    Returns:
        test_loss: Average test loss
        test_acc: Test accuracy
        confusion_matrix: Confusion matrix
        per_class_acc: Per-class accuracy
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # Initialize confusion matrix
    num_classes = len(label_dict)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Testing"):
            try:
                images = images.to(device)
                labels = labels.squeeze().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Store predictions, targets and probabilities
                all_predictions.append(preds.cpu())
                all_targets.append(labels.cpu())
                all_probabilities.append(probs.cpu())
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory during testing, skipping batch')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_probabilities = torch.cat(all_probabilities)
    
    # Compute average loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Get class names from label dictionary
    label_names = {v: k for k, v in train_dataset.label_dict.items()}
    class_names = [label_names.get(i, f"Class {i}") for i in range(len(train_dataset.label_dict))]
    
    # Compute metrics using the updated function
    compute_ci = config['evaluation'].get('compute_ci', True)
    n_bootstraps = config['evaluation'].get('n_bootstraps', 1000)
    
    metrics = compute_classification_metrics(
        all_targets, 
        all_predictions, 
        all_probabilities,
        class_names=class_names,
        compute_ci=compute_ci,
        n_bootstraps=n_bootstraps
    )
    
    # Add loss to metrics
    metrics['loss'] = avg_test_loss
    
    # Print test results
    print(f"\nTest Results:")
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    
    # Print confidence intervals if computed
    if compute_ci and 'accuracy_ci' in metrics:
        acc_lower, acc_upper = metrics['accuracy_ci']
        print(f"Accuracy 95% CI: ({acc_lower:.2f}% - {acc_upper:.2f}%)")
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for class_idx, acc in enumerate(metrics['per_class_accuracy']):
        class_name = class_names[class_idx]
        print(f"  {class_name}: {acc:.2f}%")
        if compute_ci and 'per_class_accuracy_ci' in metrics:
            lower, upper = metrics['per_class_accuracy_ci'][class_idx]
            print(f"    95% CI: ({lower:.2f}% - {upper:.2f}%)")
    
    # Print macro and weighted metrics
    print("\nMacro Metrics:")
    print(f"  Precision: {metrics['macro_precision']:.2f}%")
    print(f"  Recall: {metrics['macro_recall']:.2f}%")
    print(f"  F1 Score: {metrics['macro_f1']:.2f}%")
    if 'macro_auc' in metrics:
        print(f"  AUC: {metrics['macro_auc']:.2f}%")
    
    print("\nWeighted Metrics:")
    print(f"  Precision: {metrics['weighted_precision']:.2f}%")
    print(f"  Recall: {metrics['weighted_recall']:.2f}%")
    print(f"  F1 Score: {metrics['weighted_f1']:.2f}%")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save test results
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(config['output'].get('results_dir', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{config['backbone']['name']}_{dataset_name}_{timestamp}_test_results.txt")
    
    with open(results_path, 'w') as f:
        f.write(f"Test Results for {config['backbone']['name']} on {dataset_name}\n")
        f.write(f"Test Loss: {metrics['loss']:.4f}\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.2f}%\n")
        
        if compute_ci and 'accuracy_ci' in metrics:
            acc_lower, acc_upper = metrics['accuracy_ci']
            f.write(f"Accuracy 95% CI: ({acc_lower:.2f}% - {acc_upper:.2f}%)\n")
        
        f.write("\nPer-class Accuracy:\n")
        for class_idx, acc in enumerate(metrics['per_class_accuracy']):
            class_name = class_names[class_idx]
            f.write(f"  {class_name}: {acc:.2f}%\n")
            if compute_ci and 'per_class_accuracy_ci' in metrics:
                lower, upper = metrics['per_class_accuracy_ci'][class_idx]
                f.write(f"    95% CI: ({lower:.2f}% - {upper:.2f}%)\n")
        
        f.write("\nMacro Metrics:\n")
        f.write(f"  Precision: {metrics['macro_precision']:.2f}%\n")
        f.write(f"  Recall: {metrics['macro_recall']:.2f}%\n")
        f.write(f"  F1 Score: {metrics['macro_f1']:.2f}%\n")
        if 'macro_auc' in metrics:
            f.write(f"  AUC: {metrics['macro_auc']:.2f}%\n")
        
        f.write("\nWeighted Metrics:\n")
        f.write(f"  Precision: {metrics['weighted_precision']:.2f}%\n")
        f.write(f"  Recall: {metrics['weighted_recall']:.2f}%\n")
        f.write(f"  F1 Score: {metrics['weighted_f1']:.2f}%\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']))
    
    print(f"\nTest results saved to {results_path}")
    
    return {
        'model': model,
        'metrics': metrics
    }

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)

    # Apply overrides if provided
    if args.override:
        print("🔧 Applying configuration overrides:")
        config = apply_config_overrides(config, args.override)

    train(config)
