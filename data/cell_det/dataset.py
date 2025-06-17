import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import numpy as np


class CellDetDataset(Dataset):
    """
    Cell Detection Dataset for COCO-style annotations
    
    Dataset structure:
    dataset_root/
    ├── train/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── val/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── test/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── train.json
    ├── val.json
    └── test.json
    """
    
    def __init__(self, root, task_organ, preprocess=None, split='train'):
        """
        Args:
            root (str): Root directory of the dataset
            task_organ (str): Organ type (e.g., 'cervix', 'breast', etc.)
            preprocess (callable, optional): Preprocessing function for images
            split (str): Dataset split ('train', 'val', 'test')
        """
        self.root = root
        self.task_organ = task_organ
        self.split = split
        self.preprocess = preprocess
        
        # Paths
        self.img_dir = os.path.join(root, split)
        self.ann_file = os.path.join(root, f'{split}.json')
        
        # Check if files exist
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")
        
        # Load COCO annotations
        self.coco = COCO(self.ann_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_dict = {cat['id']: cat['name'] for cat in self.categories}
        self.num_classes = len(self.categories)
        
        # Create label mapping (COCO category IDs might not be continuous)
        self.coco_to_label = {cat_id: idx for idx, cat_id in enumerate(sorted(self.category_dict.keys()))}
        self.label_to_coco = {idx: cat_id for cat_id, idx in self.coco_to_label.items()}
        
        print(f"Loaded {split} split with {len(self.image_ids)} images and {self.num_classes} classes")
        print(f"Categories: {list(self.category_dict.values())}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Preprocessed image
            target (dict): Dictionary containing:
                - boxes (Tensor[N, 4]): Bounding boxes in (x1, y1, x2, y2) format
                - labels (Tensor[N]): Class labels for each box
                - image_id (int): Image ID
                - area (Tensor[N]): Area of each box
                - iscrowd (Tensor[N]): Whether each box is crowd
        """
        image_id = self.image_ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Convert annotations to tensors
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Convert COCO bbox format [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Convert COCO category ID to label index
            coco_cat_id = ann['category_id']
            label_idx = self.coco_to_label[coco_cat_id]
            labels.append(label_idx)
            
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        # Apply preprocessing
        if self.preprocess is not None:
            image = self.preprocess(image)
        else:
            # Default preprocessing if none provided
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image)
        
        return image, target
    
    def get_class_names(self):
        """Get list of class names in order"""
        class_names = []
        for idx in range(self.num_classes):
            coco_cat_id = self.label_to_coco[idx]
            class_names.append(self.category_dict[coco_cat_id])
        return class_names
    
    def get_image_info(self, idx):
        """Get image information"""
        image_id = self.image_ids[idx]
        return self.coco.loadImgs(image_id)[0]


def collate_fn(batch):
    """
    Custom collate function for detection datasets
    Since each image can have different numbers of objects,
    we need to handle variable-length targets
    """
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets) 