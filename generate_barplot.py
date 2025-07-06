import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import glob
import re
import os

def parse_filename(filename):
    """Extract model and dataset from filename"""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    
    if basename.startswith('ccs_'):
        model = 'CytoFMv1'
        dataset = parts[1]
    elif basename.startswith('ResNet50_'):
        model = 'ResNet50'
        dataset = parts[1]
    elif basename.startswith('ViT_L_16_'):
        model = 'ViT-L/16'
        dataset = parts[3]
    else:
        model = parts[0]
        dataset = parts[1]
    
    return model, dataset

def convert_percent_to_decimal(percent_str):
    """Convert percentage string to decimal"""
    try:
        value = float(percent_str)
        return value/100
    except:
        return np.nan

def parse_metrics_file(filepath):
    """Parse metrics from a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract Overall Accuracy
    match = re.search(r'Overall Accuracy: ([\d.]+)% \(95% CI: ([\d.]+)% - ([\d.]+)%\)', content)
    if match:
        metrics['overall_accuracy'] = convert_percent_to_decimal(match.group(1))
    
    # Extract AUC
    match = re.search(r'AUC: ([\d.]+)% \(95% CI: ([\d.]+)% - ([\d.]+)%\)', content)
    if match:
        metrics['auc'] = convert_percent_to_decimal(match.group(1))
    
    # Extract Weighted F1 Score
    match = re.search(r'Weighted F1 Score: ([\d.]+)% \(95% CI: ([\d.]+)% - ([\d.]+)%\)', content)
    if match:
        metrics['weighted_f1'] = convert_percent_to_decimal(match.group(1))
    
    return metrics

def clean_dataset_name(dataset):
    """Clean dataset name for better display"""
    if dataset == "C":
        return "C-NMC-2019"
    elif dataset == "WBC":
        return "Raabin-WBC"
    elif dataset == "OCPap":
        return "UFSC-OCPap"
    return dataset

def load_data():
    """Load all data from the three directories"""
    file_patterns = [
        'results/cell_cls/ccs/ccs_*_with_ci_metrics.txt',
        'results/cell_cls/R50_frozen/ResNet50_*_with_ci_metrics.txt',
        'results/cell_cls/VITL_unfrozen/ViT_L_16_*_with_ci_metrics.txt'
    ]
    
    results_dict = {}
    
    for pattern in file_patterns:
        files = glob.glob(pattern)
        for filepath in files:
            model, dataset = parse_filename(filepath)
            metrics = parse_metrics_file(filepath)
            
            key = f"{model}_{dataset}"
            
            if key in results_dict:
                current_timestamp = filepath.split('_')[-3] + '_' + filepath.split('_')[-2]
                existing_timestamp = results_dict[key]['filepath'].split('_')[-3] + '_' + results_dict[key]['filepath'].split('_')[-2]
                
                if current_timestamp > existing_timestamp:
                    results_dict[key] = {
                        'model': model,
                        'dataset': dataset,
                        'metrics': metrics,
                        'filepath': filepath
                    }
            else:
                results_dict[key] = {
                    'model': model,
                    'dataset': dataset,
                    'metrics': metrics,
                    'filepath': filepath
                }
    
    return [{'model': v['model'], 'dataset': v['dataset'], 'metrics': v['metrics']} 
            for v in results_dict.values()]

def create_comparison_barplot():
    """Create a comprehensive bar plot comparing all models across all datasets"""
    # Load data
    results = load_data()
    
    # Organize data by dataset
    dataset_groups = {}
    for result in results:
        dataset = result['dataset']
        if dataset not in dataset_groups:
            dataset_groups[dataset] = {}
        dataset_groups[dataset][result['model']] = result['metrics']
    
    # Prepare data for plotting
    datasets = sorted(dataset_groups.keys())
    models = ['CytoFMv1', 'ResNet50', 'ViT-L/16']
    metrics = ['overall_accuracy', 'auc', 'weighted_f1']
    metric_names = ['Overall Accuracy', 'AUC', 'Weighted F1']
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(20, 18))
    fig.suptitle('Model Performance Comparison Across All Datasets', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[metric_idx]
        
        # Prepare data for this metric
        x_pos = np.arange(len(datasets))
        width = 0.25
        
        for model_idx, model in enumerate(models):
            values = []
            for dataset in datasets:
                if model in dataset_groups[dataset] and metric in dataset_groups[dataset][model]:
                    values.append(dataset_groups[dataset][model][metric])
                else:
                    values.append(0)
            
            bars = ax.bar(x_pos + model_idx * width, values, width, 
                         label=model, color=colors[model_idx], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Customize subplot
        ax.set_xlabel('Datasets', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontweight='bold')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([clean_dataset_name(d) for d in datasets], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('model_comparison_barplot.png', dpi=300, bbox_inches='tight')
    plt.savefig('model_comparison_barplot.pdf', bbox_inches='tight')
    print("Bar plot saved as 'model_comparison_barplot.png' and 'model_comparison_barplot.pdf'")
    
    return fig

def create_single_metric_barplot(metric='overall_accuracy', metric_name='Overall Accuracy'):
    """Create a single metric bar plot for better readability"""
    # Load data
    results = load_data()

    # Organize data by dataset
    dataset_groups = {}
    for result in results:
        dataset = result['dataset']
        if dataset not in dataset_groups:
            dataset_groups[dataset] = {}
        dataset_groups[dataset][result['model']] = result['metrics']

    # Prepare data for plotting
    datasets = sorted(dataset_groups.keys())
    models = ['CytoFMv1', 'ResNet50', 'ViT-L/16']

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))

    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange

    # Prepare data
    x_pos = np.arange(len(datasets))
    width = 0.25

    for model_idx, model in enumerate(models):
        values = []
        for dataset in datasets:
            if model in dataset_groups[dataset] and metric in dataset_groups[dataset][model]:
                values.append(dataset_groups[dataset][model][metric])
            else:
                values.append(0)

        bars = ax.bar(x_pos + model_idx * width, values, width,
                     label=model, color=colors[model_idx], alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9, rotation=90)

    # Customize plot
    ax.set_xlabel('Datasets', fontweight='bold', fontsize=14)
    ax.set_ylabel(metric_name, fontweight='bold', fontsize=14)
    ax.set_title(f'{metric_name} Comparison Across All Datasets', fontweight='bold', fontsize=16)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([clean_dataset_name(d) for d in datasets], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    filename = f'model_comparison_{metric}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(f'model_comparison_{metric}.pdf', bbox_inches='tight')
    print(f"Single metric bar plot saved as '{filename}'")

    return fig

if __name__ == "__main__":
    # Create comprehensive plot
    create_comparison_barplot()

    # Create individual metric plots
    create_single_metric_barplot('overall_accuracy', 'Overall Accuracy')
    create_single_metric_barplot('auc', 'AUC')
    create_single_metric_barplot('weighted_f1', 'Weighted F1 Score')

    plt.show()
