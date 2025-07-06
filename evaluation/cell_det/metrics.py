import torch
import numpy as np
from collections import defaultdict
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
import tempfile
import os


def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
        
    Returns:
        iou: Intersection over Union
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_predictions_to_ground_truth(pred_boxes, pred_labels, pred_scores, 
                                    gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes based on IoU threshold
    
    Args:
        pred_boxes: Predicted bounding boxes [N, 4]
        pred_labels: Predicted labels [N]
        pred_scores: Prediction scores [N]
        gt_boxes: Ground truth boxes [M, 4]
        gt_labels: Ground truth labels [M]
        iou_threshold: IoU threshold for matching
        
    Returns:
        matches: List of (pred_idx, gt_idx, iou) for matched pairs
        unmatched_preds: List of unmatched prediction indices
        unmatched_gts: List of unmatched ground truth indices
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            if pred_labels[i] == gt_labels[j]:  # Same class
                iou_matrix[i, j] = compute_iou(pred_box, gt_box)
    
    # Find matches using greedy assignment
    matches = []
    used_gt = set()
    used_pred = set()
    
    # Sort predictions by score (descending)
    if len(pred_scores) == 0:
        sorted_indices = []
    else:
        sorted_indices = np.argsort(pred_scores)[::-1]
    
    for pred_idx in sorted_indices:
        best_gt_idx = -1
        best_iou = 0
        
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in used_gt:
                continue
            
            iou = iou_matrix[pred_idx, gt_idx]
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            matches.append((pred_idx, best_gt_idx, best_iou))
            used_pred.add(pred_idx)
            used_gt.add(best_gt_idx)
    
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in used_pred]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in used_gt]
    
    return matches, unmatched_preds, unmatched_gts


def compute_detection_metrics_per_class(predictions, ground_truths, class_names, 
                                       iou_thresholds=[0.5], score_threshold=0.5):
    """
    Compute detection metrics per class
    
    Args:
        predictions: List of predictions for each image
        ground_truths: List of ground truths for each image
        class_names: List of class names
        iou_thresholds: List of IoU thresholds
        score_threshold: Score threshold for predictions
        
    Returns:
        metrics: Dictionary with per-class metrics
    """
    num_classes = len(class_names)
    metrics = {}
    
    for iou_thresh in iou_thresholds:
        # Initialize counters for each class
        tp_per_class = defaultdict(int)
        fp_per_class = defaultdict(int)
        fn_per_class = defaultdict(int)
        total_gt_per_class = defaultdict(int)
        total_pred_per_class = defaultdict(int)
        
        # Process each image
        for pred, gt in zip(predictions, ground_truths):
            # Filter predictions by score threshold
            valid_pred_mask = pred['scores'] >= score_threshold
            pred_boxes = pred['boxes'][valid_pred_mask]
            pred_labels = pred['labels'][valid_pred_mask]
            pred_scores = pred['scores'][valid_pred_mask]
            
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']
            
            # Convert to numpy arrays for easier handling
            if len(pred_boxes) > 0:
                pred_boxes = pred_boxes.numpy()
                pred_labels = pred_labels.numpy()
                pred_scores = pred_scores.numpy()
            else:
                pred_boxes = np.array([]).reshape(0, 4)
                pred_labels = np.array([])
                pred_scores = np.array([])
            
            if len(gt_boxes) > 0:
                gt_boxes = gt_boxes.numpy()
                gt_labels = gt_labels.numpy()
            else:
                gt_boxes = np.array([]).reshape(0, 4)
                gt_labels = np.array([])
            
            # Count ground truth instances per class
            for label in gt_labels:
                label_val = label.item() if hasattr(label, 'item') else int(label)
                total_gt_per_class[label_val] += 1
            
            # Count predictions per class
            for label in pred_labels:
                label_val = label.item() if hasattr(label, 'item') else int(label)
                total_pred_per_class[label_val] += 1
            
            # Match predictions to ground truth
            matches, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
                pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh
            )
            
            # Count TP, FP, FN per class
            matched_pred_indices = set()
            matched_gt_indices = set()
            
            for pred_idx, gt_idx, iou in matches:
                pred_label = pred_labels[pred_idx].item() if hasattr(pred_labels[pred_idx], 'item') else int(pred_labels[pred_idx])
                tp_per_class[pred_label] += 1
                matched_pred_indices.add(pred_idx)
                matched_gt_indices.add(gt_idx)
            
            # False positives (unmatched predictions)
            for pred_idx in unmatched_preds:
                pred_label = pred_labels[pred_idx].item() if hasattr(pred_labels[pred_idx], 'item') else int(pred_labels[pred_idx])
                fp_per_class[pred_label] += 1
            
            # False negatives (unmatched ground truths)
            for gt_idx in unmatched_gts:
                gt_label = gt_labels[gt_idx].item() if hasattr(gt_labels[gt_idx], 'item') else int(gt_labels[gt_idx])
                fn_per_class[gt_label] += 1
        
        # Compute metrics per class
        # Note: Labels are offset by +1 (actual labels are [1, 2, 3, ...] for classes [0, 1, 2, ...])
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for class_idx in range(num_classes):
            label_val = class_idx + 1  # Actual label value (offset by +1 due to background class)
            tp = tp_per_class[label_val]
            fp = fp_per_class[label_val]
            fn = fn_per_class[label_val]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_per_class.append(precision * 100)  # Convert to percentage
            recall_per_class.append(recall * 100)
            f1_per_class.append(f1 * 100)
        
        metrics[f'iou_{iou_thresh}'] = {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'tp': [tp_per_class[i + 1] for i in range(num_classes)],  # offset by +1
            'fp': [fp_per_class[i + 1] for i in range(num_classes)],  # offset by +1
            'fn': [fn_per_class[i + 1] for i in range(num_classes)],  # offset by +1
            'total_gt': [total_gt_per_class[i + 1] for i in range(num_classes)],  # offset by +1
            'total_pred': [total_pred_per_class[i + 1] for i in range(num_classes)]  # offset by +1
        }
    
    return metrics


def compute_map_coco(predictions, ground_truths, class_names):
    """
    Compute COCO-style mAP using pycocotools
    
    Args:
        predictions: List of predictions for each image
        ground_truths: List of ground truths for each image
        class_names: List of class names
        
    Returns:
        map_results: Dictionary with mAP results
    """
    # Create temporary COCO format annotations
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for i, name in enumerate(class_names):
        coco_gt["categories"].append({
            "id": i + 1,  # COCO categories start from 1
            "name": name
        })
    
    # Add images and annotations
    ann_id = 1
    for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Add image
        coco_gt["images"].append({
            "id": img_id,
            "width": 640,  # Placeholder
            "height": 640   # Placeholder
        })
        
        # Add ground truth annotations
        if len(gt['boxes']) > 0:
            for box, label in zip(gt['boxes'], gt['labels']):
                x1, y1, x2, y2 = box.tolist()
                label_val = label.item() if hasattr(label, 'item') else int(label)
                coco_gt["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": label_val,  # Labels are already offset (+1) to account for background
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, width, height]
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0
                })
                ann_id += 1
    
    # Create COCO predictions
    coco_preds = []
    for img_id, pred in enumerate(predictions):
        if len(pred['boxes']) > 0:
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                x1, y1, x2, y2 = box.tolist()
                label_val = label.item() if hasattr(label, 'item') else int(label)
                score_val = score.item() if hasattr(score, 'item') else float(score)
                coco_preds.append({
                    "image_id": img_id,
                    "category_id": label_val,  # Labels are already offset (+1) to account for background
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format
                    "score": score_val
                })
    
    # Save to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_gt, f)
        gt_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_preds, f)
        pred_file = f.name
    
    try:
        # Check if we have any predictions
        if len(coco_preds) == 0:
            # No predictions, return zero mAP
            map_results = {
                'mAP': 0.0,
                'mAP_50': 0.0,
                'mAP_75': 0.0,
                'mAP_small': 0.0,
                'mAP_medium': 0.0,
                'mAP_large': 0.0,
            }
        else:
            # Load COCO ground truth
            coco_gt_api = COCO(gt_file)
            
            # Load predictions
            coco_pred_api = coco_gt_api.loadRes(pred_file)
            
            # Evaluate
            coco_eval = COCOeval(coco_gt_api, coco_pred_api, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract results
            map_results = {
                'mAP': coco_eval.stats[0] * 100,  # mAP @ IoU=0.50:0.95
                'mAP_50': coco_eval.stats[1] * 100,  # mAP @ IoU=0.50
                'mAP_75': coco_eval.stats[2] * 100,  # mAP @ IoU=0.75
                'mAP_small': coco_eval.stats[3] * 100,  # mAP for small objects
                'mAP_medium': coco_eval.stats[4] * 100,  # mAP for medium objects
                'mAP_large': coco_eval.stats[5] * 100,  # mAP for large objects
            }
        
    except Exception as e:
        print(f"Warning: COCO evaluation failed: {e}")
        # Return zero mAP if evaluation fails
        map_results = {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0,
        }
    finally:
        # Clean up temporary files
        if os.path.exists(gt_file):
            os.unlink(gt_file)
        if os.path.exists(pred_file):
            os.unlink(pred_file)
    
    return map_results


def bootstrap_sample(predictions, ground_truths, n_samples=1000):
    """
    Generate bootstrap samples
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        n_samples: Number of bootstrap samples
        
    Yields:
        Tuple of (sampled_predictions, sampled_ground_truths)
    """
    n_images = len(predictions)
    
    for _ in range(n_samples):
        # Sample with replacement
        indices = np.random.choice(n_images, size=n_images, replace=True)
        
        sampled_preds = [predictions[i] for i in indices]
        sampled_gts = [ground_truths[i] for i in indices]
        
        yield sampled_preds, sampled_gts


def compute_detection_metrics(predictions, ground_truths, class_names, 
                            compute_ci=True, n_bootstraps=1000, 
                            iou_thresholds=[0.5], score_threshold=0.5):
    """
    Compute comprehensive detection metrics with confidence intervals
    
    Args:
        predictions: List of predictions for each image
        ground_truths: List of ground truths for each image
        class_names: List of class names
        compute_ci: Whether to compute confidence intervals
        n_bootstraps: Number of bootstrap samples
        iou_thresholds: List of IoU thresholds
        score_threshold: Score threshold for predictions
        
    Returns:
        metrics: Dictionary with all metrics
    """
    # Compute base metrics
    base_metrics = compute_detection_metrics_per_class(
        predictions, ground_truths, class_names, iou_thresholds, score_threshold
    )
    
    # Compute mAP
    map_results = compute_map_coco(predictions, ground_truths, class_names)
    
    # Combine metrics for primary IoU threshold (0.5)
    primary_iou = iou_thresholds[0]
    primary_metrics = base_metrics[f'iou_{primary_iou}']
    
    # Compute macro averages
    macro_precision = np.mean(primary_metrics['precision'])
    macro_recall = np.mean(primary_metrics['recall'])
    macro_f1 = np.mean(primary_metrics['f1'])
    
    # Compute weighted averages (weighted by number of ground truth instances)
    total_gt = sum(primary_metrics['total_gt'])
    if total_gt > 0:
        weights = [gt / total_gt for gt in primary_metrics['total_gt']]
        weighted_precision = sum(p * w for p, w in zip(primary_metrics['precision'], weights))
        weighted_recall = sum(r * w for r, w in zip(primary_metrics['recall'], weights))
        weighted_f1 = sum(f * w for f, w in zip(primary_metrics['f1'], weights))
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0
    
    # Initialize results
    results = {
        'precision': primary_metrics['precision'],
        'recall': primary_metrics['recall'],
        'f1': primary_metrics['f1'],
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'tp': primary_metrics['tp'],
        'fp': primary_metrics['fp'],
        'fn': primary_metrics['fn'],
        'total_gt': primary_metrics['total_gt'],
        'total_pred': primary_metrics['total_pred'],
        **map_results
    }
    
    # Compute confidence intervals if requested
    if compute_ci and len(predictions) > 1:
        print(f"Computing confidence intervals with {n_bootstraps} bootstrap samples...")
        
        # Bootstrap sampling
        bootstrap_precisions = []
        bootstrap_recalls = []
        bootstrap_f1s = []
        bootstrap_macro_precisions = []
        bootstrap_macro_recalls = []
        bootstrap_macro_f1s = []
        bootstrap_weighted_precisions = []
        bootstrap_weighted_recalls = []
        bootstrap_weighted_f1s = []
        bootstrap_maps = []
        
        for sample_preds, sample_gts in bootstrap_sample(predictions, ground_truths, n_bootstraps):
            # Compute metrics for this bootstrap sample
            sample_metrics = compute_detection_metrics_per_class(
                sample_preds, sample_gts, class_names, [primary_iou], score_threshold
            )
            sample_primary = sample_metrics[f'iou_{primary_iou}']
            
            # Compute mAP for this sample
            try:
                sample_map = compute_map_coco(sample_preds, sample_gts, class_names)
                bootstrap_maps.append(sample_map['mAP_50'])
            except:
                bootstrap_maps.append(0.0)
            
            # Store per-class metrics
            bootstrap_precisions.append(sample_primary['precision'])
            bootstrap_recalls.append(sample_primary['recall'])
            bootstrap_f1s.append(sample_primary['f1'])
            
            # Compute and store macro averages
            bootstrap_macro_precisions.append(np.mean(sample_primary['precision']))
            bootstrap_macro_recalls.append(np.mean(sample_primary['recall']))
            bootstrap_macro_f1s.append(np.mean(sample_primary['f1']))
            
            # Compute weighted averages
            sample_total_gt = sum(sample_primary['total_gt'])
            if sample_total_gt > 0:
                sample_weights = [gt / sample_total_gt for gt in sample_primary['total_gt']]
                bootstrap_weighted_precisions.append(
                    sum(p * w for p, w in zip(sample_primary['precision'], sample_weights))
                )
                bootstrap_weighted_recalls.append(
                    sum(r * w for r, w in zip(sample_primary['recall'], sample_weights))
                )
                bootstrap_weighted_f1s.append(
                    sum(f * w for f, w in zip(sample_primary['f1'], sample_weights))
                )
            else:
                bootstrap_weighted_precisions.append(0.0)
                bootstrap_weighted_recalls.append(0.0)
                bootstrap_weighted_f1s.append(0.0)
        
        # Compute confidence intervals (2.5th and 97.5th percentiles)
        def compute_ci(values):
            return np.percentile(values, [2.5, 97.5])
        
        # Per-class CIs
        bootstrap_precisions = np.array(bootstrap_precisions)
        bootstrap_recalls = np.array(bootstrap_recalls)
        bootstrap_f1s = np.array(bootstrap_f1s)
        
        results['precision_ci'] = [compute_ci(bootstrap_precisions[:, i]) for i in range(len(class_names))]
        results['recall_ci'] = [compute_ci(bootstrap_recalls[:, i]) for i in range(len(class_names))]
        results['f1_ci'] = [compute_ci(bootstrap_f1s[:, i]) for i in range(len(class_names))]
        
        # Macro CIs
        results['macro_precision_ci'] = compute_ci(bootstrap_macro_precisions)
        results['macro_recall_ci'] = compute_ci(bootstrap_macro_recalls)
        results['macro_f1_ci'] = compute_ci(bootstrap_macro_f1s)
        
        # Weighted CIs
        results['weighted_precision_ci'] = compute_ci(bootstrap_weighted_precisions)
        results['weighted_recall_ci'] = compute_ci(bootstrap_weighted_recalls)
        results['weighted_f1_ci'] = compute_ci(bootstrap_weighted_f1s)
        
        # mAP CI
        results['mAP_50_ci'] = compute_ci(bootstrap_maps)
    
    return results 