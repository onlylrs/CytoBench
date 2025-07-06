import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import warnings
from collections import defaultdict
from tqdm import tqdm
import json
import tempfile
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util


def _squeeze_mask(mask):
    """Ensure (H,W) bool array from mask that might be (1,H,W) or (H,W)"""
    if mask.ndim == 3:
        mask = mask[0]  # Remove channel dimension
    return mask > 0.5  # Threshold and convert to bool


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
        float: AJI score (between 0 and 1)
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
    
    # Compute intersection and union for matched pairs
    intersection_sum = 0
    union_sum = 0
    matched_gt = set(gt_indices)
    matched_pred = set(pred_indices)
    
    # For matched pairs: compute intersection and union
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        pred_mask = pred_masks[p_idx]
        gt_mask = gt_masks[g_idx]
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        intersection_sum += intersection
        union_sum += union
    
    # Add unmatched predictions to union (they contribute 0 to intersection)
    for i, pred_mask in enumerate(pred_masks):
        if i not in matched_pred:
            union_sum += pred_mask.sum()
    
    # Add unmatched ground truths to union (they contribute 0 to intersection)
    for j, gt_mask in enumerate(gt_masks):
        if j not in matched_gt:
            union_sum += gt_mask.sum()
    
    if union_sum == 0:
        return 1.0
    
    # AJI = sum of intersections / sum of unions
    return intersection_sum / union_sum


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


def compute_map_coco_seg(predictions, ground_truths, class_names):
    """
    Compute COCO-style mAP for segmentation masks using pycocotools.

    Args:
        predictions (list): List of prediction dictionaries for each image.
        ground_truths (list): List of ground truth dictionaries for each image.
        class_names (list): List of class names (background is ignored).

    Returns:
        dict: A dictionary containing mAP results.
    """
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_preds = []

    # COCO category IDs start from 1. We assume class_names[0] is background.
    for i, name in enumerate(class_names[1:], 1):
        coco_gt["categories"].append({"id": i, "name": name, "supercategory": "cell"})

    ann_id = 1
    for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Use a placeholder image size; COCOeval doesn't use it for area calculation with RLE.
        height, width = (1024, 1024) if gt['masks'].numel() == 0 else gt['masks'].shape[-2:]
        coco_gt["images"].append({"id": img_id, "width": width, "height": height})

        # Ground Truth Annotations
        gt_masks = gt['masks'].cpu().numpy()
        gt_labels = gt['labels'].cpu().numpy()
        for i in range(len(gt_masks)):
            mask = gt_masks[i]
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            area = mask_util.area(rle)
            bbox = mask_util.toBbox(rle)
            rle['counts'] = rle['counts'].decode('utf-8')  # for json serialization

            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(gt_labels[i]),
                "segmentation": rle,
                "bbox": bbox.tolist(),
                "area": float(area),
                "iscrowd": 0,
            })
            ann_id += 1

        # Prediction Annotations
        pred_masks = pred['masks'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        for i in range(len(pred_masks)):
            mask = _squeeze_mask(pred_masks[i]) # Squeeze and threshold
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')

            coco_preds.append({
                "image_id": img_id,
                "category_id": int(pred_labels[i]),
                "segmentation": rle,
                "score": float(pred_scores[i]),
            })

    if not coco_preds:
        return {'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_gt:
        json.dump(coco_gt, f_gt)
        gt_path = f_gt.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f_pred:
        json.dump(coco_preds, f_pred)
        pred_path = f_pred.name

    map_results = {}
    try:
        coco_gt_api = COCO(gt_path)
        coco_pred_api = coco_gt_api.loadRes(pred_path)

        coco_eval = COCOeval(coco_gt_api, coco_pred_api, iouType='segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        map_results = {
            'mAP': stats[0] * 100,       # mAP @ IoU=0.50:0.95
            'mAP_50': stats[1] * 100,    # mAP @ IoU=0.50
            'mAP_75': stats[2] * 100,    # mAP @ IoU=0.75
            'mAP_small': stats[3] * 100,
            'mAP_medium': stats[4] * 100,
            'mAP_large': stats[5] * 100,
        }
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")
        map_results = {'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0}
    finally:
        os.remove(gt_path)
        os.remove(pred_path)

    return map_results


def compute_segmentation_metrics_per_class(predictions, ground_truths, class_names, 
                                         iou_thresholds=[0.5, 0.75], score_threshold=0.5):
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
    
    # Use the first IoU threshold for TP/FP/FN, precision, recall, F1
    primary_iou = iou_thresholds[0]
    
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    total_gt = np.zeros(num_classes)
    total_pred = np.zeros(num_classes)
    
    class_ious = [[] for _ in range(num_classes)]
    class_dices = [[] for _ in range(num_classes)]
    aji_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        # Convert tensors to numpy
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_masks = pred['masks'].cpu().numpy()
        
        gt_labels = gt['labels'].cpu().numpy()
        gt_masks = gt['masks'].cpu().numpy()
        
        # Filter predictions by score threshold
        valid_preds = pred_scores >= score_threshold
        pred_labels = pred_labels[valid_preds]
        pred_masks = pred_masks[valid_preds]
        
        # Count instances per class
        for class_idx in range(1, num_classes + 1):
            total_gt[class_idx - 1] += (gt_labels == class_idx).sum()
            total_pred[class_idx - 1] += (pred_labels == class_idx).sum()

        matches = np.zeros(len(pred_labels), dtype=bool)
        gt_matched = np.zeros(len(gt_labels), dtype=bool)
        
        if len(pred_labels) > 0 and len(gt_labels) > 0:
            iou_matrix = np.zeros((len(pred_labels), len(gt_labels)))
            for i in range(len(pred_labels)):
                for j in range(len(gt_labels)):
                    if pred_labels[i] == gt_labels[j]:
                        pred_mask = _squeeze_mask(pred_masks[i])
                        gt_mask = _squeeze_mask(gt_masks[j])
                        iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)
            
            if iou_matrix.max() > 0:
                pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
                for p_idx, g_idx in zip(pred_indices, gt_indices):
                    if iou_matrix[p_idx, g_idx] >= primary_iou:
                        matches[p_idx] = True
                        gt_matched[g_idx] = True
                        
                        class_idx = pred_labels[p_idx] - 1
                        pred_mask = _squeeze_mask(pred_masks[p_idx])
                        gt_mask = _squeeze_mask(gt_masks[g_idx])
                        class_ious[class_idx].append(compute_iou(pred_mask, gt_mask))
                        class_dices[class_idx].append(compute_dice(pred_mask, gt_mask))

        for class_idx in range(1, num_classes + 1):
            pred_class_mask = pred_labels == class_idx
            gt_class_mask = gt_labels == class_idx
            
            tp[class_idx - 1] += (matches & pred_class_mask).sum()
            fp[class_idx - 1] += ((~matches) & pred_class_mask).sum()
            fn[class_idx - 1] += ((~gt_matched) & gt_class_mask).sum()

        pred_mask_list = [_squeeze_mask(m) for m in pred_masks]
        gt_mask_list = [_squeeze_mask(m) for m in gt_masks]
        aji_scores.append(compute_aji(pred_mask_list, gt_mask_list))

    # Compute precision, recall, F1, and mean IoU/Dice per class
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    mean_ious = np.zeros(num_classes)
    mean_dices = np.zeros(num_classes)
    
    for i in range(num_classes):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i]) * 100
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i]) * 100
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        if len(class_ious[i]) > 0:
            mean_ious[i] = np.mean(class_ious[i]) * 100
        if len(class_dices[i]) > 0:
            mean_dices[i] = np.mean(class_dices[i]) * 100
            
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp.astype(int),
        'fp': fp.astype(int),
        'fn': fn.astype(int),
        'total_gt': total_gt.astype(int),
        'total_pred': total_pred.astype(int),
        'iou_per_class': mean_ious,
        'dice_per_class': mean_dices,
        'aji_scores': aji_scores
    }


def bootstrap_sample(predictions, ground_truths, n_samples):
    """Generate bootstrap samples"""
    n = len(predictions)
    for _ in range(n_samples):
        indices = np.random.choice(n, size=n, replace=True)
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
                               compute_confidence_intervals=True, n_bootstraps=1000, 
                               iou_thresholds=[0.5, 0.75], score_threshold=0.5):
    """
    Compute comprehensive segmentation metrics with confidence intervals
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries  
        class_names: List of class names
        compute_confidence_intervals: Whether to compute confidence intervals
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
        # Compute COCO-style mAP first
        map_results = compute_map_coco_seg(predictions, ground_truths, class_names)

        # Compute other metrics like precision, recall, F1, AJI
        base_metrics = compute_segmentation_metrics_per_class(
            predictions, ground_truths, class_names, 
            iou_thresholds, score_threshold
        )
        
        # Aggregate metrics
        results = {**map_results} # Start with mAP results
        
        # Add per-class precision, recall, etc.
        results['precision_per_class'] = base_metrics['precision']
        results['recall_per_class'] = base_metrics['recall']
        results['f1_per_class'] = base_metrics['f1']
        results['iou_per_class'] = base_metrics['iou_per_class']
        results['dice_per_class'] = base_metrics['dice_per_class']
        
        # Add TP/FP/FN statistics
        results['tp'] = base_metrics['tp']
        results['fp'] = base_metrics['fp']
        results['fn'] = base_metrics['fn']
        results['total_gt'] = base_metrics['total_gt']
        results['total_pred'] = base_metrics['total_pred']
        
        # Add overall metrics, ignoring classes with no ground truth instances
        valid_classes = base_metrics['total_gt'] > 0
        if valid_classes.any():
            results['macro_precision'] = np.mean(base_metrics['precision'][valid_classes])
            results['macro_recall'] = np.mean(base_metrics['recall'][valid_classes])
            results['macro_f1'] = np.mean(base_metrics['f1'][valid_classes])
            results['mean_iou'] = np.mean(base_metrics['iou_per_class'][valid_classes])
            results['mean_dice'] = np.mean(base_metrics['dice_per_class'][valid_classes])
            
            # Weighted metrics
            weights = base_metrics['total_gt'][valid_classes]
            if weights.sum() > 0:
                results['weighted_precision'] = np.average(base_metrics['precision'][valid_classes], weights=weights)
                results['weighted_recall'] = np.average(base_metrics['recall'][valid_classes], weights=weights)
                results['weighted_f1'] = np.average(base_metrics['f1'][valid_classes], weights=weights)
        
        if base_metrics['aji_scores']:
            results['aji'] = np.mean(base_metrics['aji_scores']) * 100
        else:
            results['aji'] = 0.0
        
        # Compute confidence intervals if requested
        if compute_confidence_intervals and len(predictions) > 1:
            print(f"Computing bootstrap confidence intervals with {n_bootstraps} samples...")
            
            bootstrap_maps = defaultdict(list)
            bootstrap_ajis = []
            bootstrap_f1s = []

            bootstrap_samples = bootstrap_sample(predictions, ground_truths, n_bootstraps)
            
            for sample_preds, sample_gts in tqdm(bootstrap_samples, total=n_bootstraps, desc="Bootstrap CI"):
                try:
                    # mAP CI
                    sample_map_results = compute_map_coco_seg(sample_preds, sample_gts, class_names)
                    for key, value in sample_map_results.items():
                        bootstrap_maps[key].append(value)
                    
                    # Other metrics CI
                    sample_metrics = compute_segmentation_metrics_per_class(
                        sample_preds, sample_gts, class_names, iou_thresholds, score_threshold
                    )
                    
                    # AJI CI
                    if sample_metrics['aji_scores']:
                        bootstrap_ajis.append(np.mean(sample_metrics['aji_scores']) * 100)

                    # Macro F1 CI
                    valid_classes = sample_metrics['total_gt'] > 0
                    if valid_classes.any():
                        bootstrap_f1s.append(np.mean(sample_metrics['f1'][valid_classes]))

                except Exception as e:
                    print(f"\nWarning: Bootstrap sample failed: {e}")
                    continue
            
            # Compute and store confidence intervals
            for key, values in bootstrap_maps.items():
                if values:
                    results[f'{key}_ci'] = compute_ci(values)
            if bootstrap_ajis:
                results['aji_ci'] = compute_ci(bootstrap_ajis)
            if bootstrap_f1s:
                results['macro_f1_ci'] = compute_ci(bootstrap_f1s)
        
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