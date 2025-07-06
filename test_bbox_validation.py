#!/usr/bin/env python3
"""
Test script to validate bounding box validation in the dataset
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from torch.utils.data import DataLoader
from data.cell_det.dataset import CellDetDataset, collate_fn

def test_dataset_with_invalid_bbox():
    """Test the dataset with potentially invalid bounding boxes"""
    
    # Test configuration - adjust path as needed
    dataset_root = '/jhcnas3/Cervical/CervicalData_NEW/Processed_Data/PATCH_DATA/Coco/coco_all'
    
    try:
        # Create dataset
        print("Creating dataset...")
        dataset = CellDetDataset(
            root=dataset_root,
            split='train'
        )
        
        print(f"Dataset loaded successfully with {len(dataset)} images")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Class names: {dataset.get_class_names()}")
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Use 0 to avoid multiprocessing issues during testing
            collate_fn=collate_fn
        )
        
        # Test a few batches
        print("\nTesting first few batches...")
        for batch_idx, (images, targets) in enumerate(data_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  Images: {len(images)} images")
            
            for i, target in enumerate(targets):
                boxes = target['boxes']
                labels = target['labels']
                
                print(f"  Image {i + 1}: {len(boxes)} objects")
                
                # Validate all boxes have positive dimensions
                if len(boxes) > 0:
                    widths = boxes[:, 2] - boxes[:, 0]  # x2 - x1
                    heights = boxes[:, 3] - boxes[:, 1]  # y2 - y1
                    
                    min_width = widths.min().item()
                    min_height = heights.min().item()
                    
                    print(f"    Min width: {min_width:.2f}, Min height: {min_height:.2f}")
                    
                    if min_width <= 0 or min_height <= 0:
                        print(f"    ERROR: Found invalid box dimensions!")
                        print(f"    Boxes: {boxes}")
                        return False
                    else:
                        print(f"    All boxes valid ✓")
            
            # Test first 5 batches
            if batch_idx >= 4:
                break
                
        print("\n✅ All tested batches have valid bounding boxes!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing bounding box validation...")
    success = test_dataset_with_invalid_bbox()
    
    if success:
        print("\n🎉 Test passed! Dataset handles bounding boxes correctly.")
    else:
        print("\n💥 Test failed! There are still issues with bounding boxes.")
    
    sys.exit(0 if success else 1) 