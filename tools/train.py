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

from backbone.build_model import build_backbone
from model.cell_cls.linear_probe import LinearProbeModel
from data.cell_cls.dataset import CellClsDataset
from evaluation.cell_cls.metrics import compute_classification_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, default='./configs/cell_cls/default.yaml',
                        help='Path to configuration file')
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

def train(config):
    # Set device
    device = torch.device(f"cuda:{config['common']['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build backbone model
    backbone, preprocess = build_backbone(config)
    
    # Build datasets
    dataset_name = config['data']['dataset']

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
    model = LinearProbeModel(backbone, num_classes, feature_dim)
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
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Save final model
    save_dir = os.path.join(config['output']['model_dir'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config['backbone']['name']}_{dataset_name}.pth")
    
    # If we have a best model state, use that instead of the final state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Using best model with validation accuracy: {best_val_acc:.2f}%")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
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
    
    # Overall metrics with confidence intervals
    if compute_ci and 'accuracy_ci' in metrics:
        acc_lower, acc_upper = metrics['accuracy_ci']
        formatted_results += f"Overall Accuracy: {metrics['accuracy']:.2f}% (95% CI: {acc_lower:.2f}% - {acc_upper:.2f}%)\n"
    else:
        formatted_results += f"Overall Accuracy: {metrics['accuracy']:.2f}%\n"
    
    if 'auc' in metrics:
        if compute_ci and 'auc_ci' in metrics:
            auc_lower, auc_upper = metrics['auc_ci']
            formatted_results += f"AUC: {metrics['auc']:.2f}% (95% CI: {auc_lower:.2f}% - {auc_upper:.2f}%)\n"
        else:
            formatted_results += f"AUC: {metrics['auc']:.2f}%\n"
    
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
    
    if 'weighted_f1' in metrics:
        if compute_ci and 'weighted_f1_ci' in metrics:
            wf_lower, wf_upper = metrics['weighted_f1_ci']
            formatted_results += f"Weighted F1 Score: {metrics['weighted_f1']:.2f}% (95% CI: {wf_lower:.2f}% - {wf_upper:.2f}%)\n"
        else:
            formatted_results += f"Weighted F1 Score: {metrics['weighted_f1']:.2f}%\n"
    
    # Per-class metrics
    formatted_results += "\nPer-class Metrics:\n"
    formatted_results += "-" * 120 + "\n"
    formatted_results += f"{'Class':<20} {'Accuracy':<30} {'Precision':<30} {'Recall':<30} {'F1 Score':<30}\n"
    formatted_results += "-" * 120 + "\n"
    
    for i, class_name in enumerate(class_names):
        # Format accuracy with CI
        if compute_ci and 'per_class_accuracy_ci' in metrics:
            acc_lower, acc_upper = metrics['per_class_accuracy_ci'][i]
            acc_str = f"{metrics['per_class_accuracy'][i]:.2f}% (95% CI: {acc_lower:.2f}%-{acc_upper:.2f}%)"
        else:
            acc_str = f"{metrics['per_class_accuracy'][i]:.2f}%"
        
        # Format precision with CI
        prec = metrics['precision'][i]
        if compute_ci and 'precision_ci' in metrics and i < len(metrics['precision_ci']):
            prec_lower, prec_upper = metrics['precision_ci'][i]
            prec_str = f"{prec:.2f}% (95% CI: {prec_lower:.2f}%-{prec_upper:.2f}%)"
        else:
            prec_str = f"{prec:.2f}%"
        
        # Format recall with CI
        rec = metrics['recall'][i]
        if compute_ci and 'recall_ci' in metrics and i < len(metrics['recall_ci']):
            rec_lower, rec_upper = metrics['recall_ci'][i]
            rec_str = f"{rec:.2f}% (95% CI: {rec_lower:.2f}%-{rec_upper:.2f}%)"
        else:
            rec_str = f"{rec:.2f}%"
        
        # Format F1 with CI
        f1 = metrics['f1'][i]
        if compute_ci and 'f1_ci' in metrics and i < len(metrics['f1_ci']):
            f1_lower, f1_upper = metrics['f1_ci'][i]
            f1_str = f"{f1:.2f}% (95% CI: {f1_lower:.2f}%-{f1_upper:.2f}%)"
        else:
            f1_str = f"{f1:.2f}%"
        
        formatted_results += f"{class_name:<20} {acc_str:<30} {prec_str:<30} {rec_str:<30} {f1_str:<30}\n"
    
    # Class support
    formatted_results += "\nClass Support (number of samples):\n"
    formatted_results += "-" * 50 + "\n"
    for i, class_name in enumerate(class_names):
        support = metrics['support'][i]
        formatted_results += f"{class_name:<20} {support}\n"
    
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
    results_path = os.path.join(results_dir, f"{config['backbone']['name']}_{dataset_name}{ci_suffix}_metrics.txt")
    
    with open(results_path, 'w') as f:
        f.write(formatted_results)
    
    print(f"Results saved to {results_path}")
    
    # Generate timestamp for table filenames
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Generate tables for metrics
    metrics_to_table = ['accuracy', 'auc', 'macro_f1']
    for metric in metrics_to_table:
        if metric in metrics:
            # CSV table
            csv_path = f'results/table_{metric}{ci_suffix}_{timestamp}.csv'
            print(f"CSV table saved to {csv_path}")
            
            # LaTeX table
            latex_path = f'results/table_{metric}{ci_suffix}_{timestamp}.tex'
            print(f"LaTeX table saved to {latex_path}")
    
    # Combined tables
    combined_csv_path = f'results/table_combined{ci_suffix}_{timestamp}.csv'
    combined_latex_path = f'results/table_combined{ci_suffix}_{timestamp}.tex'
    print(f"Combined CSV table saved to {combined_csv_path}")
    print(f"Combined LaTeX table saved to {combined_latex_path}")
    
    # Summary of generated tables
    print("\nResults tables generated:")
    print("Individual metric tables:")
    for metric in metrics_to_table:
        if metric in metrics:
            print(f"  {metric}: CSV - results/table_{metric}{ci_suffix}_{timestamp}.csv, LaTeX - results/table_{metric}{ci_suffix}_{timestamp}.tex")
    
    print("\nCombined tables with all metrics:")
    print(f"  CSV: {combined_csv_path}")
    print(f"  LaTeX: {combined_latex_path}")
    
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
    results_dir = os.path.join(config['output'].get('results_dir', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{config['backbone']['name']}_{dataset_name}_test_results.txt")
    
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
    train(config)
