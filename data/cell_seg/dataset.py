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
    
    Supports two dataset structures:
    
    Structure 1 (Standard):
    dataset_root/
    ├── train/           # folder with images
    ├── val/             # folder with images
    ├── test/            # folder with images
    ├── train.json       # annotations for train split
    ├── val.json         # annotations for val split
    └── test.json        # annotations for test split
    
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
    
    def __init__(self, root, split='train', transform=None, annotation_file=None, annotations_dir=None, category_id_offset=1):
        """
        Args:
            root (str): Root directory of the dataset
            split (str): Dataset split ('train', 'val', 'test') - only used for Structure 1
            transform (callable, optional): Optional transform to be applied on images
            annotation_file (str, optional): Path to annotation file - if provided, 
                                           overrides the default split-based annotation file
            annotations_dir (str, optional): Directory containing annotation files, relative to root.
                                           If not provided, looks in root directory.
            category_id_offset (int): Offset to add to category IDs to map dataset classes to model classes.
                                    Set to 1 if dataset categories start from 0, set to 0 if they start from 1.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.annotations_dir = annotations_dir
        self.category_id_offset = category_id_offset
        
        # Determine dataset structure and set paths
        self._detect_structure(annotation_file)
        
        # Load annotations
        self._load_annotations()
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    
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
                self.annotations_file = annotation_file
            else:
                self.annotations_file = os.path.join(ann_dir, annotation_file)
            self.use_split_dirs = False
            self.images_dir = self.root
        else:
            # Check if split-based structure exists
            split_ann_file = os.path.join(ann_dir, f'{self.split}.json')
            split_img_dir = os.path.join(self.root, self.split)
            
            if os.path.exists(split_ann_file):
                # Split JSON file exists, now check directory structure
                if os.path.exists(split_img_dir):
                    # Use split-based structure (Structure 1): has both split dirs and split JSONs
                    self.annotations_file = split_ann_file
                    self.split_dir = split_img_dir
                    self.images_dir = self.split_dir  # Images are directly in the split folder
                    self.use_split_dirs = True
                else:
                    # Use folder-based structure (Structure 2): has split JSONs but arbitrary dirs
                    self.annotations_file = split_ann_file
                    self.images_dir = self.root
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
                self.annotations_file = os.path.join(ann_dir, json_files[0])
                self.images_dir = self.root
                self.use_split_dirs = False
                
                print(f"Using folder-based structure with annotation file: {json_files[0]}")

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
        
        structure_type = "split-based" if self.use_split_dirs else "folder-based"
        annotations_location = f"annotations_dir: {self.annotations_dir}" if self.annotations_dir else "root directory"
        print(f"Loaded {structure_type} dataset with {len(self.image_ids)} images")
        print(f"Annotations loaded from {annotations_location}")
        print(f"Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        annotations = self.image_annotations[image_id]
        
        # Load image
        if self.use_split_dirs:
            # Structure 1: images are in split-specific directories
            image_path = os.path.join(self.images_dir, image_info['file_name'])
        else:
            # Structure 2: images are in subdirectories specified in file_name
            image_path = os.path.join(self.root, image_info['file_name'])
        
        # Verify image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
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
            
            # Validate bounding box dimensions
            if w <= 0 or h <= 0:
                print(f"Warning: Skipping invalid bounding box with w={w}, h={h} for image_id={image_id}")
                continue
            
            # Ensure minimum box size (at least 1 pixel)
            if w < 1.0:
                w = 1.0
            if h < 1.0:
                h = 1.0
            
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Final validation to ensure box has positive area
            if x2 <= x1 or y2 <= y1:
                print(f"Warning: Skipping invalid bounding box coordinates [{x1}, {y1}, {x2}, {y2}] for image_id={image_id}")
                continue
                
            boxes.append([x1, y1, x2, y2])  # Convert to [x1, y1, x2, y2]
            
            # Label (category_id) - apply offset to map dataset classes to model classes
            # Background is class 0, actual objects start from class 1
            labels.append(ann['category_id'] + self.category_id_offset)
            
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


def create_dataset_for_structure(root_dir, transform=None, split='train', annotation_file=None, annotations_dir=None):
    """
    Create a CellSegDataset instance automatically detecting the appropriate structure
    
    Args:
        root_dir (str): Root directory of the dataset
        transform (callable, optional): Transform function for images
        split (str): Dataset split for split-based structure
        annotation_file (str, optional): Specific annotation file for folder-based structure
        annotations_dir (str, optional): Directory containing annotation files, relative to root
        
    Returns:
        CellSegDataset: Dataset instance
        dict: Structure information
    """
    structure_info = detect_dataset_structure(root_dir, annotations_dir=annotations_dir)
    
    if structure_info['type'] == 'split-based':
        print(f"Detected split-based structure with splits: {structure_info['splits']}")
        dataset = CellSegDataset(root_dir, split=split, transform=transform, annotations_dir=annotations_dir)
    elif structure_info['type'] == 'folder-based-with-splits':
        print(f"Detected folder-based structure with split JSON files: {structure_info['splits']}")
        dataset = CellSegDataset(root_dir, split=split, transform=transform, annotations_dir=annotations_dir)
    else:
        print(f"Detected folder-based structure with subdirectories: {structure_info['subdirectories']}")
        if annotation_file is None and structure_info['json_files']:
            annotation_file = structure_info['json_files'][0]
            print(f"Using annotation file: {annotation_file}")
        dataset = CellSegDataset(root_dir, transform=transform, annotation_file=annotation_file, annotations_dir=annotations_dir)
    
    return dataset, structure_info


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