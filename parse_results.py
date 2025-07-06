import os
import re
import glob

def parse_filename(filename):
    """Extract model and dataset from filename"""
    # Remove path and extension
    basename = os.path.basename(filename)
    # Remove timestamp and extension
    parts = basename.split('_')

    if basename.startswith('ccs_'):
        model = 'CytoFMv1'
        dataset = parts[1]
    elif basename.startswith('ResNet50_'):
        model = 'ResNet50'
        dataset = parts[1]
    elif basename.startswith('ViT_L_16_'):
        model = 'ViT-L/16'
        dataset = parts[3]  # ViT_L_16_DatasetName
    else:
        model = parts[0]
        dataset = parts[1]

    return model, dataset

def convert_percent_to_decimal(percent_str):
    """Convert percentage string to decimal with 3 decimal places"""
    try:
        value = float(percent_str)
        return f"{value/100:.3f}"
    except:
        return "N/A"

def parse_metrics_file(filepath):
    """Parse metrics from a single file"""
    with open(filepath, 'r') as f:
        content = f.read()

    metrics = {}

    # Extract Overall Accuracy
    match = re.search(r'Overall Accuracy: ([\d.]+)% \(95% CI: ([\d.]+)% - ([\d.]+)%\)', content)
    if match:
        main_val = convert_percent_to_decimal(match.group(1))
        ci_low = convert_percent_to_decimal(match.group(2))
        ci_high = convert_percent_to_decimal(match.group(3))
        metrics['overall_accuracy'] = f"{main_val} ({ci_low}-{ci_high})"

    # Extract AUC
    match = re.search(r'AUC: ([\d.]+)% \(95% CI: ([\d.]+)% - ([\d.]+)%\)', content)
    if match:
        main_val = convert_percent_to_decimal(match.group(1))
        ci_low = convert_percent_to_decimal(match.group(2))
        ci_high = convert_percent_to_decimal(match.group(3))
        metrics['auc'] = f"{main_val} ({ci_low}-{ci_high})"

    # Extract Weighted F1 Score
    match = re.search(r'Weighted F1 Score: ([\d.]+)% \(95% CI: ([\d.]+)% - ([\d.]+)%\)', content)
    if match:
        main_val = convert_percent_to_decimal(match.group(1))
        ci_low = convert_percent_to_decimal(match.group(2))
        ci_high = convert_percent_to_decimal(match.group(3))
        metrics['weighted_f1'] = f"{main_val} ({ci_low}-{ci_high})"

    # Extract Macro Sensitivity
    match = re.search(r'Macro Sensitivity: ([\d.]+)% \(95% CI: ([\d.]+)% - ([\d.]+)%\)', content)
    if match:
        main_val = convert_percent_to_decimal(match.group(1))
        ci_low = convert_percent_to_decimal(match.group(2))
        ci_high = convert_percent_to_decimal(match.group(3))
        metrics['macro_sensitivity'] = f"{main_val} ({ci_low}-{ci_high})"

    # Extract Macro Specificity
    match = re.search(r'Macro Specificity: ([\d.]+)% \(95% CI: ([\d.]+)% - ([\d.]+)%\)', content)
    if match:
        main_val = convert_percent_to_decimal(match.group(1))
        ci_low = convert_percent_to_decimal(match.group(2))
        ci_high = convert_percent_to_decimal(match.group(3))
        metrics['macro_specificity'] = f"{main_val} ({ci_low}-{ci_high})"

    return metrics

def clean_dataset_name(dataset):
    """Clean dataset name for better display"""
    # Handle special cases
    if dataset == "C":
        return "C-NMC-2019"
    elif dataset == "WBC":
        return "Raabin-WBC"
    elif dataset == "OCPap":
        return "UFSC-OCPap"
    return dataset

def generate_latex_table(results):
    """Generate LaTeX table organized by dataset"""
    latex = """\\begin{table*}[htbp]
\\centering
\\caption{Classification Results Summary Organized by Dataset}
\\label{tab:classification_results}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{|l|l|c|c|c|c|c|}
\\hline
\\textbf{Dataset} & \\textbf{Model} & \\textbf{Overall Accuracy} & \\textbf{AUC} & \\textbf{Weighted F1} & \\textbf{Macro Sensitivity} & \\textbf{Macro Specificity} \\\\
\\hline
"""

    # Group results by dataset
    dataset_groups = {}
    for result in results:
        dataset = result['dataset']
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        dataset_groups[dataset].append(result)

    # Sort datasets alphabetically
    sorted_datasets = sorted(dataset_groups.keys())

    for dataset in sorted_datasets:
        clean_dataset = clean_dataset_name(dataset)

        # Sort models within each dataset (CytoFMv1, ResNet50, ViT-L/16)
        model_order = {'CytoFMv1': 0, 'ResNet50': 1, 'ViT-L/16': 2}
        dataset_results = sorted(dataset_groups[dataset],
                               key=lambda x: model_order.get(x['model'], 3))

        for i, result in enumerate(dataset_results):
            # Only show dataset name for the first model
            dataset_cell = clean_dataset if i == 0 else ""

            latex += f"{dataset_cell} & {result['model']} & {result['metrics'].get('overall_accuracy', 'N/A')} & {result['metrics'].get('auc', 'N/A')} & {result['metrics'].get('weighted_f1', 'N/A')} & {result['metrics'].get('macro_sensitivity', 'N/A')} & {result['metrics'].get('macro_specificity', 'N/A')} \\\\\n"
            latex += "\\hline\n"

    latex += """\\end{tabular}%
}
\\end{table*}
"""

    return latex

def generate_compact_latex_table(results):
    """Generate a compact LaTeX table organized by dataset with CI values"""
    latex = """\\begin{table*}[htbp]
\\centering
\\caption{Classification Results Summary (Organized by Dataset)}
\\label{tab:classification_results_compact}
\\footnotesize
\\begin{tabular}{|l|l|c|c|c|c|c|}
\\hline
\\textbf{Dataset} & \\textbf{Model} & \\textbf{Accuracy} & \\textbf{AUC} & \\textbf{Weighted F1} & \\textbf{Macro Sens.} & \\textbf{Macro Spec.} \\\\
\\hline
"""

    # Group results by dataset
    dataset_groups = {}
    for result in results:
        dataset = result['dataset']
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        dataset_groups[dataset].append(result)

    # Sort datasets alphabetically
    sorted_datasets = sorted(dataset_groups.keys())

    for dataset in sorted_datasets:
        clean_dataset = clean_dataset_name(dataset)

        # Sort models within each dataset (CytoFMv1, ResNet50, ViT-L/16)
        model_order = {'CytoFMv1': 0, 'ResNet50': 1, 'ViT-L/16': 2}
        dataset_results = sorted(dataset_groups[dataset],
                               key=lambda x: model_order.get(x['model'], 3))

        for i, result in enumerate(dataset_results):
            # Only show dataset name for the first model
            dataset_cell = clean_dataset if i == 0 else ""

            # Keep full CI values
            acc = result['metrics'].get('overall_accuracy', 'N/A')
            auc = result['metrics'].get('auc', 'N/A')
            f1 = result['metrics'].get('weighted_f1', 'N/A')
            sens = result['metrics'].get('macro_sensitivity', 'N/A')
            spec = result['metrics'].get('macro_specificity', 'N/A')

            latex += f"{dataset_cell} & {result['model']} & {acc} & {auc} & {f1} & {sens} & {spec} \\\\\n"
            latex += "\\hline\n"

    latex += """\\end{tabular}
\\end{table*}
"""

    return latex

def main():
    # Get all txt files from all three directories
    file_patterns = [
        'results/cell_cls/ccs/ccs_*_with_ci_metrics.txt',
        'results/cell_cls/R50_frozen/ResNet50_*_with_ci_metrics.txt',
        'results/cell_cls/VITL_unfrozen/ViT_L_16_*_with_ci_metrics.txt'
    ]

    # Dictionary to store results and handle duplicates
    results_dict = {}

    for pattern in file_patterns:
        files = glob.glob(pattern)
        for filepath in files:
            model, dataset = parse_filename(filepath)
            metrics = parse_metrics_file(filepath)

            # Create a unique key for model-dataset combination
            key = f"{model}_{dataset}"

            # If we already have this combination, keep the one with later timestamp
            if key in results_dict:
                # Extract timestamp from filename
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

    # Convert dictionary back to list
    results = [{'model': v['model'], 'dataset': v['dataset'], 'metrics': v['metrics']}
               for v in results_dict.values()]
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results)

    # Save to file
    with open('table.tex', 'w') as f:
        f.write(latex_table)

    # Also generate a compact version
    compact_table = generate_compact_latex_table(results)
    with open('table_compact.tex', 'w') as f:
        f.write(compact_table)

    print(f"Generated LaTeX table with {len(results)} entries")
    print("Saved to table.tex and table_compact.tex")

if __name__ == "__main__":
    main()
