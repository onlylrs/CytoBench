import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, confusion_matrix

def bootstrap_ci(data, metric_func, n_bootstraps=1000, ci=95):
    """
    Compute bootstrap confidence intervals for a metric
    
    Args:
        data: Tuple of (predictions, targets)
        metric_func: Function to compute the metric
        n_bootstraps: Number of bootstrap samples
        ci: Confidence interval percentage
        
    Returns:
        lower_bound: Lower bound of the confidence interval
        upper_bound: Upper bound of the confidence interval
    """
    predictions, targets = data
    n_samples = len(predictions)
    
    # Compute bootstrap samples
    bootstrap_metrics = []
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_preds = predictions[indices]
        bootstrap_targets = targets[indices]
        
        # Compute metric on bootstrap sample
        bootstrap_metric = metric_func(bootstrap_targets, bootstrap_preds)
        bootstrap_metrics.append(bootstrap_metric)
    
    # Compute confidence interval
    alpha = (100 - ci) / 2 / 100
    lower_bound = np.percentile(bootstrap_metrics, alpha * 100)
    upper_bound = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
    
    return lower_bound, upper_bound

def compute_classification_metrics(y_true, y_pred, y_score=None, class_names=None, compute_ci=True, n_bootstraps=1000):
    """
    Compute classification metrics with bootstrap confidence intervals
    
    Args:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        y_score (array-like, optional): Prediction scores/probabilities
        class_names (list, optional): List of class names
        compute_ci (bool, optional): Whether to compute confidence intervals
        n_bootstraps (int, optional): Number of bootstrap samples
        
    Returns:
        metrics (dict): Dictionary of metrics
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_score is not None and isinstance(y_score, torch.Tensor):
        y_score = y_score.cpu().numpy()
    
    # Ensure predictions and targets are 1D arrays
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    # Get number of classes
    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    metrics['confusion_matrix'] = cm
    
    # Compute overall accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    metrics['accuracy'] = accuracy
    
    # Compute confidence interval for accuracy if requested
    if compute_ci:
        acc_func = lambda y_true, y_pred: accuracy_score(y_true, y_pred) * 100
        acc_lower, acc_upper = bootstrap_ci((y_pred, y_true), acc_func, n_bootstraps)
        metrics['accuracy_ci'] = (acc_lower, acc_upper)
    
    # Compute per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        mask = (y_true == i)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == i) * 100
        else:
            class_acc = 0.0
        per_class_acc.append(class_acc)
    
    metrics['per_class_accuracy'] = np.array(per_class_acc)
    
    # Compute per-class confidence intervals if requested
    if compute_ci:
        per_class_ci = []
        for class_idx in range(num_classes):
            # Filter data for this class
            class_mask = (y_true == class_idx)
            if np.sum(class_mask) > 0:
                class_targets = y_true[class_mask]
                class_preds = y_pred[class_mask]
                
                # Define per-class accuracy function
                def class_acc_func(y_true, y_pred):
                    return np.mean(y_pred == class_idx) * 100
                
                # Compute CI
                lower, upper = bootstrap_ci((class_preds, class_targets), class_acc_func, n_bootstraps)
                per_class_ci.append((lower, upper))
            else:
                per_class_ci.append((0, 0))
        
        metrics['per_class_accuracy_ci'] = per_class_ci
    
    # Compute precision, recall, F1 score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), average=None, zero_division=0
    )
    
    metrics['precision'] = precision * 100
    metrics['recall'] = recall * 100
    metrics['f1'] = f1 * 100
    metrics['support'] = support
    
    # Compute confidence intervals for precision, recall, F1 if requested
    if compute_ci:
        # Precision CI
        precision_ci = []
        for class_idx in range(num_classes):
            def precision_func(y_true, y_pred):
                p, _, _, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[class_idx], average=None, zero_division=0
                )
                return p[0] * 100
            
            lower, upper = bootstrap_ci((y_pred, y_true), precision_func, n_bootstraps)
            precision_ci.append((lower, upper))
        
        metrics['precision_ci'] = precision_ci
        
        # Recall CI
        recall_ci = []
        for class_idx in range(num_classes):
            def recall_func(y_true, y_pred):
                _, r, _, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[class_idx], average=None, zero_division=0
                )
                return r[0] * 100
            
            lower, upper = bootstrap_ci((y_pred, y_true), recall_func, n_bootstraps)
            recall_ci.append((lower, upper))
        
        metrics['recall_ci'] = recall_ci
        
        # F1 CI
        f1_ci = []
        for class_idx in range(num_classes):
            def f1_func(y_true, y_pred):
                _, _, f, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[class_idx], average=None, zero_division=0
                )
                return f[0] * 100
            
            lower, upper = bootstrap_ci((y_pred, y_true), f1_func, n_bootstraps)
            f1_ci.append((lower, upper))
        
        metrics['f1_ci'] = f1_ci
    
    # Compute macro and weighted averages
    metrics['macro_precision'] = np.mean(metrics['precision'])
    metrics['macro_recall'] = np.mean(metrics['recall'])
    metrics['macro_f1'] = np.mean(metrics['f1'])
    
    metrics['weighted_precision'] = np.average(metrics['precision'], weights=support)
    metrics['weighted_recall'] = np.average(metrics['recall'], weights=support)
    metrics['weighted_f1'] = np.average(metrics['f1'], weights=support)
    
    # Compute confidence intervals for macro and weighted averages if requested
    if compute_ci:
        # Macro precision CI
        def macro_precision_func(y_true, y_pred):
            p, _, _, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=range(num_classes), average='macro', zero_division=0
            )
            return p * 100
        
        lower, upper = bootstrap_ci((y_pred, y_true), macro_precision_func, n_bootstraps)
        metrics['macro_precision_ci'] = (lower, upper)
        
        # Macro recall CI
        def macro_recall_func(y_true, y_pred):
            _, r, _, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=range(num_classes), average='macro', zero_division=0
            )
            return r * 100
        
        lower, upper = bootstrap_ci((y_pred, y_true), macro_recall_func, n_bootstraps)
        metrics['macro_recall_ci'] = (lower, upper)
        
        # Macro F1 CI
        def macro_f1_func(y_true, y_pred):
            _, _, f, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=range(num_classes), average='macro', zero_division=0
            )
            return f * 100
        
        lower, upper = bootstrap_ci((y_pred, y_true), macro_f1_func, n_bootstraps)
        metrics['macro_f1_ci'] = (lower, upper)
        
        # Weighted precision CI
        def weighted_precision_func(y_true, y_pred):
            p, _, _, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=range(num_classes), average='weighted', zero_division=0
            )
            return p * 100
        
        lower, upper = bootstrap_ci((y_pred, y_true), weighted_precision_func, n_bootstraps)
        metrics['weighted_precision_ci'] = (lower, upper)
        
        # Weighted recall CI
        def weighted_recall_func(y_true, y_pred):
            _, r, _, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=range(num_classes), average='weighted', zero_division=0
            )
            return r * 100
        
        lower, upper = bootstrap_ci((y_pred, y_true), weighted_recall_func, n_bootstraps)
        metrics['weighted_recall_ci'] = (lower, upper)
        
        # Weighted F1 CI
        def weighted_f1_func(y_true, y_pred):
            _, _, f, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=range(num_classes), average='weighted', zero_division=0
            )
            return f * 100
        
        lower, upper = bootstrap_ci((y_pred, y_true), weighted_f1_func, n_bootstraps)
        metrics['weighted_f1_ci'] = (lower, upper)
    
    # Compute AUC if scores are provided
    if y_score is not None:
        try:
            # For binary classification
            if num_classes == 2:
                # Ensure y_score is for the positive class
                if y_score.ndim > 1 and y_score.shape[1] == 2:
                    y_score = y_score[:, 1]
                metrics['auc'] = roc_auc_score(y_true, y_score) * 100
                
                # Compute confidence interval for AUC if requested
                if compute_ci:
                    def auc_func(y_true, prob_samples):
                        try:
                            return roc_auc_score(y_true, prob_samples) * 100
                        except:
                            return 50.0  # Default to random chance
                    
                    auc_lower, auc_upper = bootstrap_ci((y_score, y_true), auc_func, n_bootstraps)
                    metrics['auc_ci'] = (auc_lower, auc_upper)
            
            # For multi-class classification
            elif y_score.ndim > 1:
                # Compute AUC for each class (one-vs-rest)
                auc_per_class = []
                auc_ci_per_class = []
                
                for i in range(num_classes):
                    # Create binary labels for this class
                    binary_y_true = (y_true == i).astype(int)
                    class_score = y_score[:, i]
                    
                    try:
                        class_auc = roc_auc_score(binary_y_true, class_score) * 100
                        auc_per_class.append(class_auc)
                        
                        # Compute CI for this class AUC if requested
                        if compute_ci:
                            def class_auc_func(y_true, prob_samples):
                                try:
                                    return roc_auc_score(y_true, prob_samples) * 100
                                except:
                                    return 50.0
                            
                            lower, upper = bootstrap_ci((class_score, binary_y_true), class_auc_func, n_bootstraps)
                            auc_ci_per_class.append((lower, upper))
                    except:
                        auc_per_class.append(50.0)  # Default to random chance
                        if compute_ci:
                            auc_ci_per_class.append((50.0, 50.0))
                
                metrics['auc_per_class'] = np.array(auc_per_class)
                if compute_ci:
                    metrics['auc_per_class_ci'] = auc_ci_per_class
                
                # Compute macro-average AUC
                metrics['macro_auc'] = np.mean(auc_per_class)
                metrics['auc'] = metrics['macro_auc']  # Use macro AUC as the main AUC metric
                
                # Compute CI for macro AUC
                if compute_ci and auc_ci_per_class:
                    lower_bounds = [ci[0] for ci in auc_ci_per_class]
                    upper_bounds = [ci[1] for ci in auc_ci_per_class]
                    auc_lower = np.mean(lower_bounds)
                    auc_upper = np.mean(upper_bounds)
                    metrics['auc_ci'] = (auc_lower, auc_upper)
                    metrics['macro_auc_ci'] = (auc_lower, auc_upper)
        except Exception as e:
            print(f"Error computing AUC: {e}")
            metrics['auc'] = 0.0
    
    # Add class names if provided
    if class_names is not None:
        metrics['class_names'] = class_names
    
    return metrics
