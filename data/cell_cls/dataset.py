import os
import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import json
import numpy as np

class CellClsDataset(Dataset):
    """
    Dataset for cell classification
    
    Args:
        root (str): Root directory of the dataset
        task_organ (str): Organ type for the task (e.g., 'cervix')
        preprocess (callable): Preprocessing function for images
        split (str): Dataset split ('train', 'val', or 'test')
    """
    def __init__(self, root, task_organ, preprocess, split='train'):
        self.root = root
        self.task_organ = task_organ
        self.preprocess = preprocess
        self.split = split
        
        # Set task type
        self.task_type = "cls"
        
        # Load dataset metadata
        self.metadata_path = os.path.join(root, f"{self.task_type}/{split}.txt")
        if os.path.exists(self.metadata_path):
            self.metadata = self._load_metadata_from_file()
        else:
            # If metadata file doesn't exist, scan the directory structure
            self.metadata = self._create_metadata()
            
        # Get class labels
        self.label_dict = self._get_label_dict()
        
    def _load_metadata_from_file(self):
        """Load metadata from split file"""
        metadata = []
        with open(self.metadata_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_path = os.path.join(self.root, parts[0])
                label = parts[-1]
                metadata.append({
                    'image_path': image_path,
                    'label': label
                })
            else:
                print(f"Warning: Invalid line format in {self.metadata_path}: {line.strip()}")
                
        return metadata
        
    def _create_metadata(self):
        """Create metadata by scanning directory structure"""
        metadata = []
        
        # Get class directories
        class_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        
        for class_dir in class_dirs:
            class_path = os.path.join(self.root, class_dir)
            
            # Check if this directory has split subdirectories
            split_path = os.path.join(class_path, self.split)
            if os.path.exists(split_path):
                # If split subdirectory exists, use it
                image_dir = split_path
            else:
                # Otherwise use the class directory directly
                image_dir = class_path
                
            # Get all image files
            image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            
            for img_file in image_files:
                metadata.append({
                    'image_path': os.path.join(image_dir, img_file),
                    'label': class_dir
                })
                
        return metadata
    
    def _get_label_dict(self):
        """Create a dictionary mapping class names to indices"""
        labels = sorted(list(set([item['label'] for item in self.metadata])))
        return {label: idx for idx, label in enumerate(labels)}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        item = self.metadata[idx]
        
        # Load and preprocess image
        image_path = item['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image)
        
        # Get label index
        label = self.label_dict[item['label']]
        
        return image, label

def build_cell_cls_dataloaders(config):
    """
    Build dataloaders for cell classification tasks
    
    Args:
        config (dict): Configuration dictionary containing dataset settings
        
    Returns:
        dict: Dictionary of dataloaders for each dataset
    """
    from torch.utils.data import DataLoader
    from backbone.build_model import build_backbone
    
    # Build backbone to get preprocessing function
    _, preprocess = build_backbone(config)
    
    # Get dataset names from config or use default list
    dataset_names = config.get('data', {}).get('datasets', 
                              ['Herlev', 'HiCervix', 'JinWooChoi', 'FNAC2019', 
                               'LDCC', 'Sipakmed', 'Barcelona', 'BCI', 'BCFCI', 'BMCC'])
    
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    dataloaders = {}
    
    for dataset_name in dataset_names:
        dataset_root = os.path.join(config['data']['root'], dataset_name)
        organ = config['data'].get('organ', 'cervix')  # Default to cervix if not specified
        
        # Create datasets for each split
        train_dataset = CellClsDataset(
            root=dataset_root,
            task_organ=organ,
            preprocess=preprocess,
            split='train'
        )
        
        val_dataset = CellClsDataset(
            root=dataset_root,
            task_organ=organ,
            preprocess=preprocess,
            split='val'
        )
        
        test_dataset = CellClsDataset(
            root=dataset_root,
            task_organ=organ,
            preprocess=preprocess,
            split='test'
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['common']['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['common']['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['common']['num_workers'],
            pin_memory=True
        )
        
        # Store dataloaders for this dataset
        dataloaders[dataset_name] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'num_classes': len(train_dataset.label_dict)
        }
    
    return dataloaders
