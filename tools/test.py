import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backbone.build_model import build_backbone
from model.cell_cls.linear_probe import LinearProbeModel
from data.cell_cls.dataset import CellClsDataset
from evaluation.cell_cls.metrics import compute_classification_metrics

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--config', type=str, default='./configs/cell_cls/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_model_and_data(config, checkpoint_path, device):
    """Setup model and data loaders"""
    # Build backbone model
    backbone, preprocess = build_backbone(config)
    
    # Build test dataset
    dataset_name = config['data']['dataset']
    dataset_root = os.path.join(config['data']['root'], dataset_name)
    
    test_dataset = CellClsDataset(
        root=dataset_root,
        preprocess=preprocess,
        split='test'
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['common']['num_workers']
    )
    
    # Build model
    num_classes = len(test_dataset.label_dict)
    feature_dim = config['model'].get('feature_dim', 0)  # Use 0 as default to auto-detect
    model = LinearProbeModel(backbone, num_classes, feature_dim)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Get class names from label dictionary
    label_names = {v: k for k, v in test_dataset.label_dict.items()}
    class_names = [label_names.get(i, f"Class {i}") for i in range(len(test_dataset.label_dict))]
    
    return model, test_loader, class_names

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and collect predictions"""
    criterion = torch.nn.CrossEntropyLoss()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    test_loss = 0.0
    
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
    
    return all_targets, all_predictions, all_probabilities, avg_test_loss

def format_metric_with_ci(value, ci=None):
    """Format a metric with confidence interval if available"""
    if ci is not None:
        lower, upper = ci
        return f"{value:.2f}% (95% CI: {lower:.2f}% - {upper:.2f}%)"
    return f"{value:.2f}%"

def format_results(metrics, class_names, compute_ci):
    """Format evaluation results for display and saving"""
    formatted_results = "="*80 + "\n"
    formatted_results += "CLASSIFICATION METRICS\n"
    formatted_results += "="*80 + "\n\n"
    
    # Overall metrics with confidence intervals
    formatted_results += f"Overall Accuracy: {format_metric_with_ci(metrics['accuracy'], metrics.get('accuracy_ci') if compute_ci else None)}\n"
    
    if 'auc' in metrics:
        formatted_results += f"AUC: {format_metric_with_ci(metrics['auc'], metrics.get('auc_ci') if compute_ci else None)}\n"
    
    if 'macro_precision' in metrics:
        formatted_results += f"Macro Precision: {format_metric_with_ci(metrics['macro_precision'], metrics.get('macro_precision_ci') if compute_ci else None)}\n"
    
    if 'macro_recall' in metrics:
        formatted_results += f"Macro Recall: {format_metric_with_ci(metrics['macro_recall'], metrics.get('macro_recall_ci') if compute_ci else None)}\n"
    
    if 'macro_f1' in metrics:
        formatted_results += f"Macro F1 Score: {format_metric_with_ci(metrics['macro_f1'], metrics.get('macro_f1_ci') if compute_ci else None)}\n"
    
    # Per-class metrics
    formatted_results += "\nPer-class Metrics:\n"
    formatted_results += "-" * 120 + "\n"
    formatted_results += f"{'Class':<20} {'Accuracy':<30} {'Precision':<30} {'Recall':<30} {'F1 Score':<30}\n"
    formatted_results += "-" * 120 + "\n"
    
    for i, class_name in enumerate(class_names):
        # Format metrics with CI if available
        acc_ci = metrics.get('per_class_accuracy_ci')[i] if compute_ci and 'per_class_accuracy_ci' in metrics else None
        prec_ci = metrics.get('precision_ci')[i] if compute_ci and 'precision_ci' in metrics and i < len(metrics.get('precision_ci', [])) else None
        rec_ci = metrics.get('recall_ci')[i] if compute_ci and 'recall_ci' in metrics and i < len(metrics.get('recall_ci', [])) else None
        f1_ci = metrics.get('f1_ci')[i] if compute_ci and 'f1_ci' in metrics and i < len(metrics.get('f1_ci', [])) else None
        
        acc_str = format_metric_with_ci(metrics['per_class_accuracy'][i], acc_ci)
        prec_str = format_metric_with_ci(metrics['precision'][i], prec_ci)
        rec_str = format_metric_with_ci(metrics['recall'][i], rec_ci)
        f1_str = format_metric_with_ci(metrics['f1'][i], f1_ci)
        
        formatted_results += f"{class_name:<20} {acc_str:<30} {prec_str:<30} {rec_str:<30} {f1_str:<30}\n"
    
    # Confusion matrix
    formatted_results += "\nConfusion Matrix:\n"
    formatted_results += str(metrics['confusion_matrix'])
    
    return formatted_results

def save_results(formatted_results, config, checkpoint_path, compute_ci):
    """Save formatted results to file"""
    results_dir = os.path.join(config['output'].get('results_dir', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract model name from checkpoint path
    checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
    
    # Add suffix for confidence intervals
    ci_suffix = "_with_ci" if compute_ci else ""
    results_path = os.path.join(results_dir, f"{checkpoint_name}{ci_suffix}_test_metrics.txt")
    
    with open(results_path, 'w') as f:
        f.write(formatted_results)
    
    print(f"Results saved to {results_path}")

def test(config, checkpoint_path):
    """Test a trained model"""
    # Set device
    device = torch.device(f"cuda:{config['common']['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup model and data
    model, test_loader, class_names = setup_model_and_data(config, checkpoint_path, device)
    
    # Get evaluation parameters from config
    compute_ci = config['evaluation'].get('compute_ci', True)
    n_bootstraps = config['evaluation'].get('n_bootstraps', 1000)
    
    print(f"Computing bootstrap confidence intervals: {compute_ci}")
    if compute_ci:
        print(f"Number of bootstrap samples: {n_bootstraps}")
    
    # Evaluate model
    print("\n" + "="*80)
    print("Evaluating model on test set...")
    targets, predictions, probabilities, test_loss = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_classification_metrics(
        targets, 
        predictions, 
        probabilities,
        class_names=class_names,
        compute_ci=compute_ci,
        n_bootstraps=n_bootstraps
    )
    
    # Add loss to metrics
    metrics['loss'] = test_loss
    
    # Format and display results
    formatted_results = format_results(metrics, class_names, compute_ci)
    print(formatted_results)
    
    # Save results to file
    save_results(formatted_results, config, checkpoint_path, compute_ci)
    
    return metrics

def main():
    """Main function"""
    args = parse_args()
    config = load_config(args.config)
    test(config, args.checkpoint)

if __name__ == '__main__':
    main()
