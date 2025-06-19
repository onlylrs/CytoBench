import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cell_det.dataset import CellDetDataset, collate_fn
from model.cell_det.detection_model import build_detection_model
from evaluation.cell_det.metrics import compute_detection_metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cell Detection Testing Script')
    parser.add_argument('--config', type=str, default='./configs/cell_det/default.yaml',
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
    # Build test dataset
    dataset_name = config['data']['dataset']
    dataset_root = os.path.join(config['data']['root'], dataset_name)
    
    test_dataset = CellDetDataset(
        root=dataset_root,
        split='test'
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['common']['num_workers'],
        collate_fn=collate_fn
    )
    
    # Build model
    num_classes = test_dataset.num_classes
    model = build_detection_model(
        config['model']['name'],
        num_classes,
        config['model']['pretrained']
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Get class names
    class_names = test_dataset.get_class_names()
    
    return model, test_loader, class_names


def evaluate_model(model, data_loader, device):
    """Evaluate model on test set and collect predictions"""
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Testing"):
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


def format_metric_with_ci(value, ci=None):
    """Format a metric with confidence interval if available"""
    if ci is not None:
        lower, upper = ci
        return f"{value:.2f}% (95% CI: {lower:.2f}% - {upper:.2f}%)"
    return f"{value:.2f}%"


def format_results(metrics, class_names, compute_ci):
    """Format evaluation results for display and saving"""
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
    """Test a trained detection model"""
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
    predictions, ground_truths = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_detection_metrics(
        predictions, 
        ground_truths, 
        class_names,
        compute_ci=compute_ci,
        n_bootstraps=n_bootstraps,
        iou_thresholds=config['evaluation']['iou_thresholds'],
        score_threshold=config['evaluation']['score_threshold']
    )
    
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