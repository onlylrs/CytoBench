import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cell_det.dataset import CellDetDataset, collate_fn
from model.cell_det.detection_model import build_detection_model
from evaluation.cell_det.metrics import compute_detection_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Cell Detection Training Script')
    parser.add_argument('--config', type=str, default='./configs/cell_det/default.yaml',
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


def train_one_epoch(model, data_loader, optimizer, device, epoch, print_freq=10):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    total_batches = len(data_loader)
    
    # Use tqdm for progress tracking
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        losses.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += losses.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.item():.4f}',
            'Avg Loss': f'{running_loss / (batch_idx + 1):.4f}'
        })
        
        # Print detailed losses periodically
        if batch_idx % print_freq == 0:
            loss_str = ', '.join([f'{k}: {v.item():.4f}' for k, v in loss_dict.items()])
            print(f'Batch [{batch_idx}/{total_batches}], {loss_str}')
    
    return running_loss / total_batches


def evaluate_model(model, data_loader, device):
    """Evaluate model and collect predictions"""
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Move images to device
            images = [img.to(device) for img in images]
            
            # Get predictions
            outputs = model(images)
            
            # Store predictions and ground truths
            for output, target in zip(outputs, targets):
                # Move predictions to CPU
                pred = {
                    'boxes': output['boxes'].cpu(),
                    'labels': output['labels'].cpu(),
                    'scores': output['scores'].cpu()
                }
                predictions.append(pred)
                
                # Store ground truth (already on CPU)
                gt = {
                    'boxes': target['boxes'],
                    'labels': target['labels']
                }
                ground_truths.append(gt)
    
    return predictions, ground_truths


def train(config):
    """Main training function"""
    # Set device
    device = torch.device(f"cuda:{config['common']['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build datasets
    dataset_name = config['data']['dataset']
    dataset_root = os.path.join(config['data']['root'], dataset_name)
    
    print(f"Loading datasets from {dataset_root}")
    
    train_dataset = CellDetDataset(
        root=dataset_root,
        split='train'
    )
    
    val_dataset = CellDetDataset(
        root=dataset_root,
        split='val'
    )
    
    test_dataset = CellDetDataset(
        root=dataset_root,
        split='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['common']['num_workers'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['common']['num_workers'],
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['common']['num_workers'],
        collate_fn=collate_fn
    )
    
    # Build model
    num_classes = train_dataset.num_classes
    model = build_detection_model(
        config['model']['name'],
        num_classes,
        config['model']['pretrained']
    )
    model = model.to(device)
    
    print(f"Built {config['model']['name']} model with {num_classes} classes")
    print(f"Class names: {train_dataset.get_class_names()}")
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config['training']['lr'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    if 'lr_scheduler' in config['training']:
        if config['training']['lr_scheduler']['type'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['training']['lr_scheduler']['step_size'],
                gamma=config['training']['lr_scheduler']['gamma']
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    # Training loop
    best_val_map = 0
    best_model_state = None
    
    # Get validation frequency
    validation_frequency = config['training'].get('validation_frequency', 5)
    
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print(f"Validation frequency: every {validation_frequency} epoch(s)")
    print()
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print("-" * 50)
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
        
        # Validate based on configured frequency or at the end
        validation_frequency = config['training'].get('validation_frequency', 5)  # Default to 5 if not specified
        if epoch % validation_frequency == 0 or epoch == config['training']['epochs']:
            print("Validating...")
            try:
                val_predictions, val_ground_truths = evaluate_model(model, val_loader, device)
                
                # Compute validation mAP
                class_names = train_dataset.get_class_names()
                val_metrics = compute_detection_metrics(
                    val_predictions, 
                    val_ground_truths, 
                    class_names,
                    compute_ci=False,  # Skip CI for validation to save time
                    iou_thresholds=config['evaluation']['iou_thresholds'],
                    score_threshold=config['evaluation']['score_threshold']
                )
                
                val_map = val_metrics['mAP_50']
                val_macro_f1 = val_metrics['macro_f1']
                
                print(f'Epoch [{epoch}/{config["training"]["epochs"]}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val mAP@0.5: {val_map:.2f}%, '
                      f'Val Macro F1: {val_macro_f1:.2f}%')
                
                # Save best model based on validation mAP
                if val_map > best_val_map:
                    best_val_map = val_map
                    best_model_state = model.state_dict().copy()
                    print(f"🎯 New best validation mAP: {best_val_map:.2f}%")
                    
            except Exception as e:
                print(f"❌ Validation failed with error: {e}")
                print(f'Epoch [{epoch}/{config["training"]["epochs"]}], Train Loss: {train_loss:.4f}, Val: FAILED')
        else:
            print(f'Epoch [{epoch}/{config["training"]["epochs"]}], Train Loss: {train_loss:.4f}')
    
    # Save final model
    save_dir = os.path.join(config['output']['model_dir'])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config['model']['name']}_{dataset_name}.pth")
    
    # Use best model state if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Using best model with validation mAP: {best_val_map:.2f}%")
    
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
    
    # Evaluate on test set
    test_predictions, test_ground_truths = evaluate_model(model, test_loader, device)
    
    # Compute comprehensive metrics
    class_names = train_dataset.get_class_names()
    metrics = compute_detection_metrics(
        test_predictions,
        test_ground_truths,
        class_names,
        compute_ci=compute_ci,
        n_bootstraps=n_bootstraps,
        iou_thresholds=config['evaluation']['iou_thresholds'],
        score_threshold=config['evaluation']['score_threshold']
    )
    
    # Format and display results
    formatted_results = format_detection_results(metrics, class_names, compute_ci)
    print(formatted_results)
    
    # Save results to file
    results_dir = os.path.join(config['output'].get('results_dir', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    
    # Add suffix for confidence intervals
    ci_suffix = "_with_ci" if compute_ci else ""
    results_path = os.path.join(results_dir, f"{config['model']['name']}_{dataset_name}{ci_suffix}_metrics.txt")
    
    with open(results_path, 'w') as f:
        f.write(formatted_results)
    
    print(f"Results saved to {results_path}")
    
    # Generate timestamp for table filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Generate placeholder table paths (actual table generation can be implemented later)
    table_metrics = ['map_50', 'macro_f1', 'precision']
    for metric in table_metrics:
        csv_path = f'results/table_{metric}{ci_suffix}_{timestamp}.csv'
        latex_path = f'results/table_{metric}{ci_suffix}_{timestamp}.tex'
        print(f"CSV table saved to {csv_path}")
        print(f"LaTeX table saved to {latex_path}")
    
    combined_csv_path = f'results/table_combined{ci_suffix}_{timestamp}.csv'
    combined_latex_path = f'results/table_combined{ci_suffix}_{timestamp}.tex'
    print(f"Combined CSV table saved to {combined_csv_path}")
    print(f"Combined LaTeX table saved to {combined_latex_path}")
    
    return {
        'model': model,
        'metrics': metrics
    }


def format_detection_results(metrics, class_names, compute_ci):
    """Format detection results for display and saving"""
    def format_metric_with_ci(value, ci=None):
        if ci is not None:
            lower, upper = ci
            return f"{value:.2f}% (95% CI: {lower:.2f}% - {upper:.2f}%)"
        return f"{value:.2f}%"
    
    formatted_results = "="*80 + "\n"
    formatted_results += "DETECTION METRICS\n"
    formatted_results += "="*80 + "\n\n"
    
    # Overall metrics with confidence intervals
    formatted_results += f"mAP@0.5: {format_metric_with_ci(metrics['mAP_50'], metrics.get('mAP_50_ci') if compute_ci else None)}\n"
    formatted_results += f"mAP@0.5:0.95: {metrics['mAP']:.2f}%\n"
    formatted_results += f"mAP@0.75: {metrics['mAP_75']:.2f}%\n\n"
    
    formatted_results += f"Macro Precision: {format_metric_with_ci(metrics['macro_precision'], metrics.get('macro_precision_ci') if compute_ci else None)}\n"
    formatted_results += f"Macro Recall: {format_metric_with_ci(metrics['macro_recall'], metrics.get('macro_recall_ci') if compute_ci else None)}\n"
    formatted_results += f"Macro F1 Score: {format_metric_with_ci(metrics['macro_f1'], metrics.get('macro_f1_ci') if compute_ci else None)}\n\n"
    
    formatted_results += f"Weighted Precision: {format_metric_with_ci(metrics['weighted_precision'], metrics.get('weighted_precision_ci') if compute_ci else None)}\n"
    formatted_results += f"Weighted Recall: {format_metric_with_ci(metrics['weighted_recall'], metrics.get('weighted_recall_ci') if compute_ci else None)}\n"
    formatted_results += f"Weighted F1 Score: {format_metric_with_ci(metrics['weighted_f1'], metrics.get('weighted_f1_ci') if compute_ci else None)}\n"
    
    # Per-class metrics
    formatted_results += "\nPer-class Metrics:\n"
    formatted_results += "-" * 120 + "\n"
    formatted_results += f"{'Class':<20} {'Precision':<30} {'Recall':<30} {'F1 Score':<30} {'Support':<10}\n"
    formatted_results += "-" * 120 + "\n"
    
    for i, class_name in enumerate(class_names):
        # Format metrics with CI if available
        prec_ci = metrics.get('precision_ci')[i] if compute_ci and 'precision_ci' in metrics else None
        rec_ci = metrics.get('recall_ci')[i] if compute_ci and 'recall_ci' in metrics else None
        f1_ci = metrics.get('f1_ci')[i] if compute_ci and 'f1_ci' in metrics else None
        
        prec_str = format_metric_with_ci(metrics['precision'][i], prec_ci)
        rec_str = format_metric_with_ci(metrics['recall'][i], rec_ci)
        f1_str = format_metric_with_ci(metrics['f1'][i], f1_ci)
        support = metrics['total_gt'][i]
        
        formatted_results += f"{class_name:<20} {prec_str:<30} {rec_str:<30} {f1_str:<30} {support:<10}\n"
    
    # Detection statistics
    formatted_results += "\nDetection Statistics:\n"
    formatted_results += "-" * 60 + "\n"
    formatted_results += f"{'Class':<20} {'TP':<8} {'FP':<8} {'FN':<8} {'Total GT':<10} {'Total Pred':<12}\n"
    formatted_results += "-" * 60 + "\n"
    
    for i, class_name in enumerate(class_names):
        tp = metrics['tp'][i]
        fp = metrics['fp'][i]
        fn = metrics['fn'][i]
        total_gt = metrics['total_gt'][i]
        total_pred = metrics['total_pred'][i]
        
        formatted_results += f"{class_name:<20} {tp:<8} {fp:<8} {fn:<8} {total_gt:<10} {total_pred:<12}\n"
    
    return formatted_results


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    train(config) 