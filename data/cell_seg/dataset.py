import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pycocotools import mask as coco_mask
import torchvision.transforms as transforms


class CellSegDataset(Dataset):
    """
    COCO-style dataset for cell segmentation
    
    Expected directory structure:
    dataset_root/
    ├── train/           # folder with images
    ├── val/             # folder with images
    ├── test/            # folder with images
    ├── train.json       # annotations for train split
    ├── val.json         # annotations for val split
    └── test.json        # annotations for test split
    """
    
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root (str): Root directory of the dataset
            split (str): Dataset split ('train', 'val', 'test')
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # Paths
        self.split_dir = os.path.join(root, split)
        self.images_dir = self.split_dir  # Images are directly in the split folder
        self.annotations_file = os.path.join(root, f'{split}.json')  # JSON files are at root level
        
        # Load annotations
        self._load_annotations()
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    
    def _load_annotations(self):
        """Load COCO-style annotations"""
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)
        
        # Get list of image IDs that have annotations
        self.image_ids = list(self.image_annotations.keys())
        
        # Create class name mapping
        self.class_names = ['__background__'] + [cat['name'] for cat in sorted(self.categories.values(), key=lambda x: x['id'])]
        self.num_classes = len(self.class_names)
        
        print(f"Loaded {len(self.image_ids)} images for {self.split} split")
        print(f"Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        annotations = self.image_annotations[image_id]
        
        # Load image
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Get image dimensions
        width, height = image.size
        
        # Process annotations
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowds = []
        
        for ann in annotations:
            # Bounding box
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            
            # Label (category_id)
            labels.append(ann['category_id'])
            
            # Segmentation mask
            if isinstance(ann['segmentation'], list):
                # Polygon format
                rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
                mask = coco_mask.decode(rle)
                if len(mask.shape) == 3:
                    mask = mask.sum(axis=2)
                mask = (mask > 0).astype(np.uint8)
            else:
                # RLE format
                if isinstance(ann['segmentation']['counts'], list):
                    rle = coco_mask.frPyObjects([ann['segmentation']], height, width)[0]
                else:
                    rle = ann['segmentation']
                mask = coco_mask.decode(rle)
            
            masks.append(mask)
            
            # Area and iscrowd
            areas.append(ann.get('area', w * h))
            iscrowds.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': iscrowds
        }
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def get_class_names(self):
        """Get list of class names"""
        return self.class_names[1:]  # Exclude background class
    
    def get_image_info(self, idx):
        """Get image information"""
        image_id = self.image_ids[idx]
        return self.images[image_id]


def collate_fn(batch):
    """Custom collate function for segmentation data loader"""
    images, targets = zip(*batch)
    
    # Images are already tensors, just stack them
    images = list(images)
    
    # Targets are dictionaries, keep them as list
    targets = list(targets)
    
    return images, targets


# Test the dataset
if __name__ == '__main__':
    # Example usage
    dataset_root = '/path/to/your/dataset'
    dataset = CellSegDataset(
        root=dataset_root,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class names: {dataset.get_class_names()}")
    
    # Get a sample
    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Number of objects: {len(target['boxes'])}")
        print(f"Mask shape: {target['masks'].shape}") 