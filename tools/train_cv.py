import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backbone.build_model import build_backbone, auto_set_feature_dim
from model.cell_cls.linear_probe import LinearProbeModel
from data.cell_cls.dataset import build_cv_dataloaders
from evaluation.cell_cls.metrics import compute_classification_metrics

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
    parser = argparse.ArgumentParser(description='Cross Validation Training script')
    parser.add_argument('--config', type=str, default='./configs/cell_cls/default.yaml',
                        help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file with inheritance support"""
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
    """Recursively merge two dictionaries"""
    merged = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
            
    return merged

def train_epoch(model, data_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader, desc="Training", leave=False):
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
        for images, labels in data_loader:
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

def print_config_cv(config, fold):
    """Print configuration for cross validation fold"""
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1} CONFIGURATION")
    print(f"{'='*60}")

    # Key settings for CV
    print("📋 Key Settings:")
    print(f"  • Dataset: {config.get('data', {}).get('dataset', 'Not set')}")
    print(f"  • Backbone: {config.get('backbone', {}).get('name', 'Not set')}")
    print(f"  • Freeze backbone: {config.get('backbone', {}).get('freeze', 'Not set')}")
    print(f"  • Batch size: {config.get('training', {}).get('batch_size', 'Not set')}")
    print(f"  • Learning rate: {config.get('training', {}).get('lr', 'Not set')}")
    print(f"  • Epochs: {config.get('training', {}).get('epochs', 'Not set')}")
    print(f"  • Dropout: {config.get('model', {}).get('dropout_p', 'Not set')}")

    # CV specific settings
    evaluation = config.get('evaluation', {})
    print(f"  • CV folds: {evaluation.get('cv_folds', 'Not set')}")
    print(f"  • CV seed: {evaluation.get('cv_seed', 'Not set')}")
    print(f"  • Current fold: {fold + 1}")
    print(f"{'='*60}")

def train_single_fold(config, fold):
    """Train and evaluate a single fold"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold + 1}")
    print(f"{'='*60}")

    # Auto-set feature_dim based on backbone name
    config = auto_set_feature_dim(config)

    # Print configuration for this fold
    print_config_cv(config, fold)

    # Set device
    device = torch.device(f"cuda:{config['common']['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build backbone model
    backbone, preprocess = build_backbone(config)
    
    # Build dataloaders for this fold
    dataloaders = build_cv_dataloaders(config, fold)
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    print(f"Fold {fold + 1} data sizes:")
    print(f"  Train: {dataloaders['train_size']} samples")
    print(f"  Val: {dataloaders['val_size']} samples")
    print(f"  Test: {dataloaders['test_size']} samples")
    
    # Build model
    num_classes = dataloaders['num_classes']
    feature_dim = config['model'].get('feature_dim', 0)
    freeze_backbone = config['backbone'].get('freeze', False)
    dropout_p = config['model'].get('dropout_p', 0.5)  # Default dropout probability
    model = LinearProbeModel(backbone, num_classes, feature_dim, freeze_backbone, dropout_p)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0

    print(f"Starting training for fold {fold + 1} with {config['training']['epochs']} epochs...")

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
    
    # Load best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Using best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Test the model
    print(f"\nEvaluating Fold {fold + 1} on test set...")
    
    # Collect predictions and targets for metrics computation
    all_predictions = []
    all_targets = []
    all_probabilities = []
    test_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
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
    label_names = {v: k for k, v in dataloaders['label_dict'].items()}
    class_names = [label_names.get(i, f"Class {i}") for i in range(num_classes)]
    
    # Compute metrics
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
    metrics['fold'] = fold
    metrics['best_val_acc'] = best_val_acc
    
    return metrics

def summarize_cv_results(fold_results, config):
    """Summarize cross validation results across all folds"""
    print(f"\n{'='*80}")
    print("CROSS VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")

    dataset_name = config['data']['dataset']
    backbone_name = config['backbone']['name']
    total_folds = len(fold_results)

    # Extract metrics from all folds
    metrics_names = ['accuracy', 'auc', 'macro_f1', 'macro_precision', 'macro_recall']
    fold_metrics = {}

    for metric_name in metrics_names:
        fold_metrics[metric_name] = []
        for fold_result in fold_results:
            if metric_name in fold_result:
                fold_metrics[metric_name].append(fold_result[metric_name])

    # Compute statistics across folds
    summary = {}
    for metric_name, values in fold_metrics.items():
        if values:
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }

    # Print summary
    print(f"Dataset: {dataset_name}")
    print(f"Backbone: {backbone_name}")
    print(f"Number of folds: {total_folds}")
    print(f"Freeze backbone: {config['backbone'].get('freeze', False)}")
    print()

    # Print per-fold results
    print("Per-fold Results:")
    print("-" * 80)
    header = f"{'Fold':<6}"
    for metric_name in metrics_names:
        if metric_name in summary:
            header += f"{metric_name.replace('_', ' ').title():<12}"
    print(header)
    print("-" * 80)

    for i, fold_result in enumerate(fold_results):
        row = f"{i+1:<6}"
        for metric_name in metrics_names:
            if metric_name in fold_result:
                row += f"{fold_result[metric_name]:<12.2f}"
            else:
                row += f"{'N/A':<12}"
        print(row)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 80)

    for metric_name in metrics_names:
        if metric_name in summary:
            stats = summary[metric_name]
            print(f"{metric_name.replace('_', ' ').title():<15} "
                  f"{stats['mean']:<10.2f} {stats['std']:<10.2f} "
                  f"{stats['min']:<10.2f} {stats['max']:<10.2f}")

    # Save results to file
    experiment_name = generate_experiment_name(backbone_name, dataset_name)
    results_dir = config['output'].get('results_dir', 'results/cell_cls')
    os.makedirs(results_dir, exist_ok=True)

    # Set environment variable for consistent naming across scripts
    import os
    os.environ['EXPERIMENT_NAME'] = experiment_name

    # Save detailed results with consistent naming
    results_file = os.path.join(results_dir, f"cv_results_{experiment_name}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'fold_results': fold_results,
            'summary': summary,
            'timestamp': timestamp
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_file}")

    return summary

def main():
    args = parse_args()
    config = load_config(args.config)

    # Check if cross validation is enabled
    if not config.get('evaluation', {}).get('cv_enabled', False):
        print("Cross validation is not enabled in config. Set evaluation.cv_enabled: true")
        return

    # Get cross validation settings
    cv_folds = config['evaluation'].get('cv_folds', 5)
    cv_seed = config['evaluation'].get('cv_seed', 42)

    print(f"Starting {cv_folds}-fold cross validation")
    print(f"Dataset: {config['data']['dataset']}")
    print(f"Backbone: {config['backbone']['name']}")
    print(f"Random seed: {cv_seed}")

    # Set random seeds for reproducibility
    torch.manual_seed(cv_seed)
    np.random.seed(cv_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cv_seed)

    # Train and evaluate each fold
    fold_results = []
    for fold in range(cv_folds):
        try:
            fold_metrics = train_single_fold(config, fold)
            fold_results.append(fold_metrics)
        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")
            continue

    if not fold_results:
        print("No successful folds completed!")
        return

    # Summarize results
    summary = summarize_cv_results(fold_results, config)

    print(f"\nCross validation completed successfully!")
    print(f"Mean accuracy: {summary.get('accuracy', {}).get('mean', 0):.2f}% ± {summary.get('accuracy', {}).get('std', 0):.2f}%")

if __name__ == "__main__":
    main()
