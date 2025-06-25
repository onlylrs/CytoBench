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
    
    Supports two dataset structures:
    
    Structure 1 (Standard):
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
    
    Structure 2 (Alternative):
    dataset_root/
    ├── A/
    │   ├── image1.jpg
    │   └── ...
    ├── B/
    │   ├── image2.jpg
    │   └── ...
    ├── C/
    │   └── ...
    └── annotations.json  (or any json file)
    
    In Structure 2, the JSON file contains image names like "A/image1.jpg"
    """
    
    def __init__(self, root, preprocess=None, split='train', annotation_file=None, annotations_dir=None):
        """
        Args:
            root (str): Root directory of the dataset
            preprocess (callable, optional): Preprocessing function for images
            split (str): Dataset split ('train', 'val', 'test') - only used for Structure 1
            annotation_file (str, optional): Path to annotation file - if provided, 
                                           overrides the default split-based annotation file
            annotations_dir (str, optional): Directory containing annotation files, relative to root.
                                           If not provided, looks in root directory.
        """
        self.root = root
        self.split = split
        self.preprocess = preprocess
        self.annotations_dir = annotations_dir
        
        # Determine dataset structure and set paths
        self._detect_structure(annotation_file)
        
        # Check if files exist
        if self.use_split_dirs and not os.path.exists(self.img_dir):
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
        
        structure_type = "split-based" if self.use_split_dirs else "folder-based"
        annotations_location = f"annotations_dir: {self.annotations_dir}" if self.annotations_dir else "root directory"
        print(f"Loaded {structure_type} dataset with {len(self.image_ids)} images and {self.num_classes} classes")
        print(f"Annotations loaded from {annotations_location}")
        print(f"Categories: {list(self.category_dict.values())}")
        
    def _detect_structure(self, annotation_file=None):
        """
        Detect dataset structure and set appropriate paths
        """
        # Determine annotation directory
        if self.annotations_dir is not None:
            ann_dir = os.path.join(self.root, self.annotations_dir)
        else:
            ann_dir = self.root
            
        if annotation_file is not None:
            # Custom annotation file provided - use folder-based structure
            if os.path.isabs(annotation_file):
                self.ann_file = annotation_file
            else:
                self.ann_file = os.path.join(ann_dir, annotation_file)
            self.use_split_dirs = False
            self.img_dir = self.root
        else:
            # Check if split-based structure exists
            split_ann_file = os.path.join(ann_dir, f'{self.split}.json')
            split_img_dir = os.path.join(self.root, self.split)
            
            if os.path.exists(split_ann_file):
                # Split JSON file exists, now check directory structure
                if os.path.exists(split_img_dir):
                    # Use split-based structure (Structure 1): has both split dirs and split JSONs
                    self.ann_file = split_ann_file
                    self.img_dir = split_img_dir
                    self.use_split_dirs = True
                else:
                    # Use folder-based structure (Structure 2): has split JSONs but arbitrary dirs
                    self.ann_file = split_ann_file
                    self.img_dir = self.root
                    self.use_split_dirs = False
                    print(f"Using folder-based structure with split JSON: {self.split}.json")
            else:
                # Try to find any JSON file and use folder-based structure
                if os.path.exists(ann_dir):
                    json_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
                else:
                    json_files = []
                    
                if not json_files:
                    raise FileNotFoundError(f"No annotation files found in {ann_dir}")
                
                # Use the first JSON file found
                self.ann_file = os.path.join(ann_dir, json_files[0])
                self.img_dir = self.root
                self.use_split_dirs = False
                
                print(f"Using folder-based structure with annotation file: {json_files[0]}")

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
        
        if self.use_split_dirs:
            # Structure 1: images are in split-specific directories
            img_path = os.path.join(self.img_dir, img_info['file_name'])
        else:
            # Structure 2: images are in subdirectories specified in file_name
            img_path = os.path.join(self.root, img_info['file_name'])
        
        # Verify image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
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


def detect_dataset_structure(root_dir, annotations_dir=None):
    """
    Detect the structure of a dataset directory
    
    Args:
        root_dir (str): Root directory of the dataset
        annotations_dir (str, optional): Directory containing annotation files, relative to root
        
    Returns:
        dict: Information about the dataset structure
    """
    structure_info = {
        'type': None,
        'splits': [],
        'json_files': [],
        'subdirectories': []
    }
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    # Determine annotation directory
    if annotations_dir is not None:
        ann_dir = os.path.join(root_dir, annotations_dir)
        if not os.path.exists(ann_dir):
            raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")
    else:
        ann_dir = root_dir
    
    # Get all items in root directory
    items = os.listdir(root_dir)
    
    # Find JSON files in annotation directory
    ann_items = os.listdir(ann_dir)
    json_files = [f for f in ann_items if f.endswith('.json')]
    structure_info['json_files'] = json_files
    
    # Find subdirectories
    subdirs = [d for d in items if os.path.isdir(os.path.join(root_dir, d))]
    structure_info['subdirectories'] = subdirs
    
    # Check for split-based structure
    split_dirs = [d for d in subdirs if d in ['train', 'val', 'test']]
    split_jsons = [f for f in json_files if f.replace('.json', '') in ['train', 'val', 'test']]
    
    if split_jsons:
        if split_dirs:
            structure_info['type'] = 'split-based'
            structure_info['splits'] = split_dirs
        else:
            structure_info['type'] = 'folder-based-with-splits'
            structure_info['splits'] = [f.replace('.json', '') for f in split_jsons]
    else:
        structure_info['type'] = 'folder-based'
    
    return structure_info


def create_dataset_for_structure(root_dir, preprocess=None, split='train', annotation_file=None, annotations_dir=None):
    """
    Create a CellDetDataset instance automatically detecting the appropriate structure
    
    Args:
        root_dir (str): Root directory of the dataset
        preprocess (callable, optional): Preprocessing function for images
        split (str): Dataset split for split-based structure
        annotation_file (str, optional): Specific annotation file for folder-based structure
        annotations_dir (str, optional): Directory containing annotation files, relative to root
        
    Returns:
        CellDetDataset: Dataset instance
        dict: Structure information
    """
    structure_info = detect_dataset_structure(root_dir, annotations_dir=annotations_dir)
    
    if structure_info['type'] == 'split-based':
        print(f"Detected split-based structure with splits: {structure_info['splits']}")
        dataset = CellDetDataset(root_dir, preprocess=preprocess, split=split, annotations_dir=annotations_dir)
    elif structure_info['type'] == 'folder-based-with-splits':
        print(f"Detected folder-based structure with split JSON files: {structure_info['splits']}")
        dataset = CellDetDataset(root_dir, preprocess=preprocess, split=split, annotations_dir=annotations_dir)
    else:
        print(f"Detected folder-based structure with subdirectories: {structure_info['subdirectories']}")
        if annotation_file is None and structure_info['json_files']:
            annotation_file = structure_info['json_files'][0]
            print(f"Using annotation file: {annotation_file}")
        dataset = CellDetDataset(root_dir, preprocess=preprocess, annotation_file=annotation_file, annotations_dir=annotations_dir)
    
    return dataset, structure_info 