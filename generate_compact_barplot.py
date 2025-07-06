import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

def create_compact_all_datasets_plot():
    """Create a compact plot showing all datasets with all three metrics"""
    # Load data
    results = load_data()
    
    # Create DataFrame for easier manipulation
    data_rows = []
    for result in results:
        dataset = clean_dataset_name(result['dataset'])
        model = result['model']
        metrics = result['metrics']
        
        for metric_key, metric_name in [('overall_accuracy', 'Overall Accuracy'), 
                                       ('auc', 'AUC'), 
                                       ('weighted_f1', 'Weighted F1')]:
            if metric_key in metrics:
                data_rows.append({
                    'Dataset': dataset,
                    'Model': model,
                    'Metric': metric_name,
                    'Value': metrics[metric_key]
                })
    
    df = pd.DataFrame(data_rows)
    
    # Create the plot
    plt.figure(figsize=(24, 16))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create the grouped bar plot
    ax = sns.barplot(data=df, x='Dataset', y='Value', hue='Model', 
                     palette=['#2E86AB', '#A23B72', '#F18F01'],
                     alpha=0.8)
    
    # Customize the plot
    plt.title('Model Performance Comparison Across All Datasets\n(Overall Accuracy, AUC, and Weighted F1)', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Datasets', fontsize=14, fontweight='bold')
    plt.ylabel('Performance Score', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add legend
    plt.legend(title='Model', fontsize=12, title_fontsize=12, loc='upper right')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(0, 1.1)
    
    # Add value labels on bars (optional, might be crowded)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.3f', rotation=90, fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('compact_all_datasets_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('compact_all_datasets_comparison.pdf', bbox_inches='tight')
    print("Compact comparison plot saved as 'compact_all_datasets_comparison.png' and 'compact_all_datasets_comparison.pdf'")
    
    return plt.gcf()

def create_heatmap_comparison():
    """Create a heatmap showing model performance across datasets"""
    # Load data
    results = load_data()
    
    # Create pivot table for heatmap
    data_rows = []
    for result in results:
        dataset = clean_dataset_name(result['dataset'])
        model = result['model']
        if 'overall_accuracy' in result['metrics']:
            data_rows.append({
                'Dataset': dataset,
                'Model': model,
                'Overall_Accuracy': result['metrics']['overall_accuracy']
            })
    
    df = pd.DataFrame(data_rows)
    pivot_df = df.pivot(index='Dataset', columns='Model', values='Overall_Accuracy')
    
    # Create heatmap
    plt.figure(figsize=(12, 20))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlBu_r', center=0.5, 
                fmt='.3f', cbar_kws={'label': 'Overall Accuracy'})
    
    plt.title('Model Performance Heatmap (Overall Accuracy)', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Dataset', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('model_performance_heatmap.pdf', bbox_inches='tight')
    print("Heatmap saved as 'model_performance_heatmap.png' and 'model_performance_heatmap.pdf'")
    
    return plt.gcf()

if __name__ == "__main__":
    # Create compact comparison plot
    create_compact_all_datasets_plot()
    
    # Create heatmap
    create_heatmap_comparison()
    
    plt.show()
