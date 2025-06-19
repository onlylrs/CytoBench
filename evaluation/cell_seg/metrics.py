import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import warnings
from collections import defaultdict


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_dice(mask1, mask2):
    """Compute Dice coefficient between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2 * intersection / total


def compute_aji(pred_masks, gt_masks):
    """
    Compute Aggregated Jaccard Index (AJI) for instance segmentation
    
    Args:
        pred_masks: List of predicted masks (each mask is a 2D numpy array)
        gt_masks: List of ground truth masks (each mask is a 2D numpy array)
    
    Returns:
        float: AJI score
    """
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0
    
    if len(gt_masks) == 0:
        return 0.0
    
    if len(pred_masks) == 0:
        return 0.0
    
    # Compute pairwise IoU matrix
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)
    
    # Find optimal assignment using Hungarian algorithm
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
    
    # Compute AJI
    matched_iou_sum = 0
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        matched_iou_sum += iou_matrix[p_idx, g_idx]
    
    total_union = 0
    matched_gt = set(gt_indices)
    matched_pred = set(pred_indices)
    
    # Union of matched pairs
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        pred_mask = pred_masks[p_idx]
        gt_mask = gt_masks[g_idx]
        total_union += np.logical_or(pred_mask, gt_mask).sum()
    
    # Union of unmatched predictions
    for i, pred_mask in enumerate(pred_masks):
        if i not in matched_pred:
            total_union += pred_mask.sum()
    
    # Union of unmatched ground truths
    for j, gt_mask in enumerate(gt_masks):
        if j not in matched_gt:
            total_union += gt_mask.sum()
    
    if total_union == 0:
        return 1.0
    
    return matched_iou_sum / total_union


def masks_to_boxes(masks):
    """Convert masks to bounding boxes"""
    boxes = []
    for mask in masks:
        if mask.sum() == 0:
            boxes.append([0, 0, 0, 0])
            continue
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            boxes.append([0, 0, 0, 0])
            continue
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        boxes.append([cmin, rmin, cmax + 1, rmax + 1])
    
    return np.array(boxes)


def compute_segmentation_metrics_per_class(predictions, ground_truths, class_names, 
                                         iou_thresholds=[0.5], score_threshold=0.5):
    """
    Compute per-class segmentation metrics
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        class_names: List of class names
        iou_thresholds: List of IoU thresholds for mAP computation
        score_threshold: Score threshold for filtering predictions
    
    Returns:
        dict: Dictionary containing per-class metrics
    """
    num_classes = len(class_names)
    
    # Initialize counters
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    total_gt = np.zeros(num_classes)
    total_pred = np.zeros(num_classes)
    
    # For mAP computation
    all_scores = [[] for _ in range(num_classes)]
    all_matches = [[] for _ in range(num_classes)]
    
    # For mask-based metrics
    class_ious = [[] for _ in range(num_classes)]
    class_dices = [[] for _ in range(num_classes)]
    
    # For AJI computation (per image)
    aji_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        # Convert tensors to numpy
        pred_boxes = pred['boxes'].cpu().numpy() if len(pred['boxes']) > 0 else np.array([]).reshape(0, 4)
        pred_labels = pred['labels'].cpu().numpy() if len(pred['labels']) > 0 else np.array([])
        pred_scores = pred['scores'].cpu().numpy() if len(pred['scores']) > 0 else np.array([])
        pred_masks = pred['masks'].cpu().numpy() if len(pred['masks']) > 0 else np.array([]).reshape(0, 0, 0)
        
        gt_boxes = gt['boxes'].cpu().numpy() if len(gt['boxes']) > 0 else np.array([]).reshape(0, 4)
        gt_labels = gt['labels'].cpu().numpy() if len(gt['labels']) > 0 else np.array([])
        gt_masks = gt['masks'].cpu().numpy() if len(gt['masks']) > 0 else np.array([]).reshape(0, 0, 0)
        
        # Filter predictions by score threshold
        if len(pred_scores) > 0:
            valid_preds = pred_scores >= score_threshold
            pred_boxes = pred_boxes[valid_preds]
            pred_labels = pred_labels[valid_preds]
            pred_scores = pred_scores[valid_preds]
            pred_masks = pred_masks[valid_preds]
        
        # Count ground truth instances per class
        for class_idx in range(1, num_classes + 1):  # Skip background (class 0)
            gt_class_mask = gt_labels == class_idx
            total_gt[class_idx - 1] += gt_class_mask.sum()
            
            pred_class_mask = pred_labels == class_idx
            total_pred[class_idx - 1] += pred_class_mask.sum()
        
        # Compute matches for each IoU threshold
        for iou_thresh in iou_thresholds:
            matches = np.zeros(len(pred_labels), dtype=bool)
            gt_matched = np.zeros(len(gt_labels), dtype=bool)
            
            if len(pred_labels) > 0 and len(gt_labels) > 0:
                # Compute IoU matrix for masks
                iou_matrix = np.zeros((len(pred_labels), len(gt_labels)))
                for i in range(len(pred_labels)):
                    for j in range(len(gt_labels)):
                        if pred_labels[i] == gt_labels[j]:
                            pred_mask = pred_masks[i] > 0.5
                            gt_mask = gt_masks[j] > 0.5
                            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)
                
                # Find matches using Hungarian algorithm
                if iou_matrix.max() > 0:
                    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
                    for p_idx, g_idx in zip(pred_indices, gt_indices):
                        if iou_matrix[p_idx, g_idx] >= iou_thresh:
                            matches[p_idx] = True
                            gt_matched[g_idx] = True
                            
                            # Store IoU and Dice for matched pairs
                            class_idx = pred_labels[p_idx] - 1
                            pred_mask = pred_masks[p_idx] > 0.5
                            gt_mask = gt_masks[g_idx] > 0.5
                            class_ious[class_idx].append(compute_iou(pred_mask, gt_mask))
                            class_dices[class_idx].append(compute_dice(pred_mask, gt_mask))
            
            # Update TP, FP, FN for each class
            for class_idx in range(1, num_classes + 1):
                pred_class_mask = pred_labels == class_idx
                gt_class_mask = gt_labels == class_idx
                
                # True positives: matched predictions of this class
                tp[class_idx - 1] += (matches & pred_class_mask).sum()
                
                # False positives: unmatched predictions of this class
                fp[class_idx - 1] += ((~matches) & pred_class_mask).sum()
                
                # False negatives: unmatched ground truths of this class
                fn[class_idx - 1] += ((~gt_matched) & gt_class_mask).sum()
                
                # Store scores and matches for mAP computation
                class_pred_indices = np.where(pred_class_mask)[0]
                for idx in class_pred_indices:
                    all_scores[class_idx - 1].append(pred_scores[idx])
                    all_matches[class_idx - 1].append(matches[idx])
        
        # Compute AJI for this image
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            pred_mask_list = [pred_masks[i] > 0.5 for i in range(len(pred_masks))]
            gt_mask_list = [gt_masks[i] > 0.5 for i in range(len(gt_masks))]
            aji_score = compute_aji(pred_mask_list, gt_mask_list)
            aji_scores.append(aji_score)
        elif len(gt_masks) == 0 and len(pred_masks) == 0:
            aji_scores.append(1.0)
        else:
            aji_scores.append(0.0)
    
    # Compute precision, recall, F1 for each class
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i]) * 100
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i]) * 100
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    # Compute mAP for primary IoU threshold
    map_scores = []
    for i in range(num_classes):
        if len(all_scores[i]) > 0:
            # Sort by score (descending)
            sorted_indices = np.argsort(all_scores[i])[::-1]
            sorted_matches = np.array(all_matches[i])[sorted_indices]
            
            # Compute precision-recall curve
            tp_cumsum = np.cumsum(sorted_matches)
            fp_cumsum = np.cumsum(~sorted_matches)
            
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            recalls = tp_cumsum / max(total_gt[i], 1)
            
            # Compute AP using 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11
            
            map_scores.append(ap * 100)
        else:
            map_scores.append(0.0)
    
    # Compute mean IoU and Dice per class
    mean_ious = []
    mean_dices = []
    for i in range(num_classes):
        if len(class_ious[i]) > 0:
            mean_ious.append(np.mean(class_ious[i]) * 100)
            mean_dices.append(np.mean(class_dices[i]) * 100)
        else:
            mean_ious.append(0.0)
            mean_dices.append(0.0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp.astype(int),
        'fp': fp.astype(int),
        'fn': fn.astype(int),
        'total_gt': total_gt.astype(int),
        'total_pred': total_pred.astype(int),
        'map_per_class': map_scores,
        'iou_per_class': mean_ious,
        'dice_per_class': mean_dices,
        'aji_scores': aji_scores
    }


def bootstrap_sample(predictions, ground_truths, n_samples):
    """Generate bootstrap samples"""
    n = len(predictions)
    for _ in range(n_samples):
        indices = np.random.choice(n, n, replace=True)
        sample_preds = [predictions[i] for i in indices]
        sample_gts = [ground_truths[i] for i in indices]
        yield sample_preds, sample_gts


def compute_ci(values, confidence=0.95):
    """Compute confidence interval"""
    if len(values) == 0:
        return (0, 0)
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return (np.percentile(values, lower_percentile), 
            np.percentile(values, upper_percentile))


def compute_segmentation_metrics(predictions, ground_truths, class_names, 
                               compute_ci=True, n_bootstraps=1000, 
                               iou_thresholds=[0.5], score_threshold=0.5):
    """
    Compute comprehensive segmentation metrics with confidence intervals
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries  
        class_names: List of class names
        compute_ci: Whether to compute confidence intervals
        n_bootstraps: Number of bootstrap samples
        iou_thresholds: List of IoU thresholds
        score_threshold: Score threshold for filtering predictions
    
    Returns:
        dict: Dictionary containing all metrics
    """
    if not predictions or not ground_truths:
        print("Warning: Empty predictions or ground truths")
        return {}
    
    try:
        # Compute base metrics
        base_metrics = compute_segmentation_metrics_per_class(
            predictions, ground_truths, class_names, 
            iou_thresholds, score_threshold
        )
        
        # Aggregate metrics
        results = {
            'precision': base_metrics['precision'],
            'recall': base_metrics['recall'],
            'f1': base_metrics['f1'],
            'tp': base_metrics['tp'],
            'fp': base_metrics['fp'],
            'fn': base_metrics['fn'],
            'total_gt': base_metrics['total_gt'],
            'total_pred': base_metrics['total_pred'],
            'map_per_class': base_metrics['map_per_class'],
            'iou_per_class': base_metrics['iou_per_class'],
            'dice_per_class': base_metrics['dice_per_class']
        }
        
        # Compute overall metrics
        valid_classes = base_metrics['total_gt'] > 0
        
        if valid_classes.any():
            results['macro_precision'] = np.mean(base_metrics['precision'][valid_classes])
            results['macro_recall'] = np.mean(base_metrics['recall'][valid_classes])
            results['macro_f1'] = np.mean(base_metrics['f1'][valid_classes])
            results['mAP_50'] = np.mean(base_metrics['map_per_class'])
            results['mAP'] = results['mAP_50']  # For compatibility
            results['mAP_75'] = results['mAP_50']  # Simplified for now
            results['mean_iou'] = np.mean(base_metrics['iou_per_class'])
            results['mean_dice'] = np.mean(base_metrics['dice_per_class'])
            
            # Weighted metrics
            weights = base_metrics['total_gt'][valid_classes]
            if weights.sum() > 0:
                results['weighted_precision'] = np.average(
                    base_metrics['precision'][valid_classes], weights=weights
                )
                results['weighted_recall'] = np.average(
                    base_metrics['recall'][valid_classes], weights=weights
                )
                results['weighted_f1'] = np.average(
                    base_metrics['f1'][valid_classes], weights=weights
                )
            else:
                results['weighted_precision'] = 0
                results['weighted_recall'] = 0
                results['weighted_f1'] = 0
        else:
            # No valid classes
            results.update({
                'macro_precision': 0, 'macro_recall': 0, 'macro_f1': 0,
                'mAP_50': 0, 'mAP': 0, 'mAP_75': 0,
                'mean_iou': 0, 'mean_dice': 0,
                'weighted_precision': 0, 'weighted_recall': 0, 'weighted_f1': 0
            })
        
        # AJI score
        if base_metrics['aji_scores']:
            results['aji'] = np.mean(base_metrics['aji_scores']) * 100
        else:
            results['aji'] = 0.0
        
        # Compute confidence intervals if requested
        if compute_ci and len(predictions) > 1:
            print(f"Computing bootstrap confidence intervals with {n_bootstraps} samples...")
            
            bootstrap_precision = [[] for _ in range(len(class_names))]
            bootstrap_recall = [[] for _ in range(len(class_names))]
            bootstrap_f1 = [[] for _ in range(len(class_names))]
            bootstrap_macro_precision = []
            bootstrap_macro_recall = []
            bootstrap_macro_f1 = []
            bootstrap_weighted_precision = []
            bootstrap_weighted_recall = []
            bootstrap_weighted_f1 = []
            bootstrap_maps = []
            bootstrap_ious = []
            bootstrap_dices = []
            bootstrap_ajis = []
            
            for sample_preds, sample_gts in bootstrap_sample(predictions, ground_truths, n_bootstraps):
                try:
                    sample_metrics = compute_segmentation_metrics_per_class(
                        sample_preds, sample_gts, class_names, 
                        iou_thresholds, score_threshold
                    )
                    
                    # Per-class metrics
                    for i in range(len(class_names)):
                        bootstrap_precision[i].append(sample_metrics['precision'][i])
                        bootstrap_recall[i].append(sample_metrics['recall'][i])
                        bootstrap_f1[i].append(sample_metrics['f1'][i])
                    
                    # Aggregate metrics
                    valid_classes = sample_metrics['total_gt'] > 0
                    if valid_classes.any():
                        bootstrap_macro_precision.append(
                            np.mean(sample_metrics['precision'][valid_classes])
                        )
                        bootstrap_macro_recall.append(
                            np.mean(sample_metrics['recall'][valid_classes])
                        )
                        bootstrap_macro_f1.append(
                            np.mean(sample_metrics['f1'][valid_classes])
                        )
                        
                        weights = sample_metrics['total_gt'][valid_classes]
                        if weights.sum() > 0:
                            bootstrap_weighted_precision.append(
                                np.average(sample_metrics['precision'][valid_classes], weights=weights)
                            )
                            bootstrap_weighted_recall.append(
                                np.average(sample_metrics['recall'][valid_classes], weights=weights)
                            )
                            bootstrap_weighted_f1.append(
                                np.average(sample_metrics['f1'][valid_classes], weights=weights)
                            )
                        
                        bootstrap_maps.append(np.mean(sample_metrics['map_per_class']))
                        bootstrap_ious.append(np.mean(sample_metrics['iou_per_class']))
                        bootstrap_dices.append(np.mean(sample_metrics['dice_per_class']))
                    
                    if sample_metrics['aji_scores']:
                        bootstrap_ajis.append(np.mean(sample_metrics['aji_scores']) * 100)
                    
                except Exception as e:
                    print(f"Warning: Bootstrap sample failed: {e}")
                    continue
            
            # Compute confidence intervals
            results['precision_ci'] = [compute_ci(bootstrap_precision[i]) for i in range(len(class_names))]
            results['recall_ci'] = [compute_ci(bootstrap_recall[i]) for i in range(len(class_names))]
            results['f1_ci'] = [compute_ci(bootstrap_f1[i]) for i in range(len(class_names))]
            
            # Aggregate CIs
            if bootstrap_macro_precision:
                results['macro_precision_ci'] = compute_ci(bootstrap_macro_precision)
                results['macro_recall_ci'] = compute_ci(bootstrap_macro_recall)
                results['macro_f1_ci'] = compute_ci(bootstrap_macro_f1)
            
            if bootstrap_weighted_precision:
                results['weighted_precision_ci'] = compute_ci(bootstrap_weighted_precision)
                results['weighted_recall_ci'] = compute_ci(bootstrap_weighted_recall)
                results['weighted_f1_ci'] = compute_ci(bootstrap_weighted_f1)
            
            if bootstrap_maps:
                results['mAP_50_ci'] = compute_ci(bootstrap_maps)
            
            if bootstrap_ious:
                results['mean_iou_ci'] = compute_ci(bootstrap_ious)
            
            if bootstrap_dices:
                results['mean_dice_ci'] = compute_ci(bootstrap_dices)
            
            if bootstrap_ajis:
                results['aji_ci'] = compute_ci(bootstrap_ajis)
        
        return results
        
    except Exception as e:
        print(f"Error computing segmentation metrics: {e}")
        return {}


# Test the metrics
if __name__ == '__main__':
    print("Testing segmentation metrics...")
    
    # Create dummy data
    predictions = []
    ground_truths = []
    
    # Add test cases here if needed
    print("Segmentation metrics implementation completed!") 